import torch
import torch.distributed as dist
from flash_attn.flash_attn_interface import _flash_attn_forward, _flash_attn_backward
import torch.distributed
# from .utils import RingCommNew, RingCommOld, RingComm
from ring_flash_attn.utils import update_out_and_lse_old as update_out_and_lse
from ring_flash_attn.utils import print_rank_0
from comm_lib.comm_utils import A2AComm

def deepspeed_ulysses_attn_forward(
    process_groups,
    q_s: torch.Tensor,    # [mbs, S / sp, Nh, D]
    k_s: torch.Tensor,
    v_s: torch.Tensor,
    softmax_scale,
    dropout_p=0,
    causal=True,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
    PROC_INFO=None,
    opt=False,
):
    a2a_comm = A2AComm(process_groups, PROC_INFO)
    q_h = a2a_comm.all_to_all(q_s, -2, -3)    # [mbs, S, Nh / sp, D]
    k_h = a2a_comm.all_to_all(k_s, -2, -3)
    v_h = a2a_comm.all_to_all(v_s, -2, -3)
    
    def forward(q, k, v, causal):
        block_out, _, _, _, _, block_lse, _, _ = _flash_attn_forward(
            q,
            k,
            v,
            dropout_p,
            softmax_scale,
            causal=causal,
            window_size=window_size,
            alibi_slopes=alibi_slopes,
            return_softmax=True and dropout_p > 0,
        )
        return block_out, block_lse
    # Compute
    forward(q_s, k_s, v_s, causal)  # [TODO]
    
    out_s = a2a_comm.all_to_all(out_h, -3, -2)  # [mbs, S / sp, Nh, D]
    
    return q_h, k_h, v_h, out_h, softmax_lse_h, out_s

def deepspeed_ulysses_attn_backward(
    process_group,
    dout_s,
    q_h,
    k_h,
    v_h,
    out_h,
    softmax_lse,
    softmax_scale=None,
    dropout_p=0,
    causal=True,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
    opt=False,
    PROC_INFO: dict =None,
):
    a2a_comm = A2AComm(process_groups, PROC_INFO)
    dout_h = a2a_comm.all_to_all(dout_s, -2, -3)  # [mbs, S, Nh / sp, D]
    
    def backward(dout, q, k, v, out, softmax_lse, causal):
        seqlen_q = q.shape[1]
        seqlen_kv = k.shape[1]
        _flash_attn_backward(
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            dq_buffer[:, :seqlen_q],
            dk_buffer[:, :seqlen_kv],
            dv_buffer[:, :seqlen_kv],
            dropout_p,
            softmax_scale,
            causal,
            window_size,
            alibi_slopes,
            deterministic,
            rng_state=None,
        )
        
    # Compute
    dq_s = a2a_comm.all_to_all(dq_h, -3, -2)  # [mbs, S / sp, Nh, D]
    dk_s = a2a_comm.all_to_all(dk_h, -3, -2)
    dv_s = a2a_comm.all_to_all(dv_h, -3, -2)
    return dq_s, dk_s, dv_s
    
class DeepSpeedUlyssesAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_softmax,
        groups,
        PROC_INFO,
        opt,
    ):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)

        assert alibi_slopes is None
        k = k.contiguous()
        v = v.contiguous()
        q_h, k_h, v_h, out_h, softmax_lse_h, out_s = deepspeed_ulysses_attn_forward(
            groups,
            q,
            k,
            v,
            softmax_scale=softmax_scale,
            dropout_p=dropout_p,
            causal=causal,
            window_size=window_size,
            alibi_slopes=alibi_slopes,
            deterministic=False,
            PROC_INFO=PROC_INFO,
            opt=opt,
        )
        # this should be out_padded
        ctx.save_for_backward(q_h, k_h, v_h, out_h, softmax_lse_h)
        ctx.dropout_p = dropout_p
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.alibi_slopes = alibi_slopes
        ctx.deterministic = deterministic
        ctx.groups = groups
        ctx.opt = opt
        return out_s if not return_softmax else (out_s, softmax_lse_h, None)

    @staticmethod
    def backward(ctx, dout, *args):
        q_h, k_h, v_h, out_h, softmax_lse_h = ctx.saved_tensors
        dq, dk, dv = deepspeed_ulysses_attn_backward(
            ctx.group,
            dout,
            q_h,
            k_h,
            v_h,
            out_h,
            softmax_lse_h,
            softmax_scale=ctx.softmax_scale,
            dropout_p=ctx.dropout_p,
            causal=ctx.causal,
            window_size=ctx.window_size,
            alibi_slopes=ctx.alibi_slopes,
            deterministic=ctx.deterministic,
            opt=ctx.opt,
        )
        return dq, dk, dv, None, None, None, None, None, None, None, None


def deepspeed_ulysses_attn_func(
    q,  # [mbs, S / sp, Nh, D]
    k,  # [mbs, S / sp, Nh, D]
    v,  # [mbs, S / sp, Nh, D]
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    groups=None,
    PROC_INFO=None,
    opt=False,
):
    return DeepSpeedUlyssesAttnFunc.apply(
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        groups,
        PROC_INFO,
        opt,
    )