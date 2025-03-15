# flexattn code example: https://gist.github.com/Chillee/2e270fc5413dbbce58c779f8c4eac66c
import torch
# from torch.nn.attention._flex_attention import _create_block_mask, _create_mask
from torch.nn.attention.flex_attention import create_block_mask, create_mask
from functools import partial
# from torch.nn.attention._flex_attention import _flex_attention
from torch.nn.attention.flex_attention import flex_attention
from triton.testing import do_bench
import torch.nn.functional as F
from functools import lru_cache
torch.set_default_device('cuda')

# Example usage
flex_attention = torch.compile(flex_attention, dynamic=False)

# Autotunes for better perf
# flex_attention = torch.compile(_flex_attention, dynamic=False, mode="max-autotune-no-cudagraphs")

torch.manual_seed(0)

data_type = torch.float16
@lru_cache
def create_block_mask_from_score_mod(score_mod, B, H, M, N, device='cuda'):
    SPARSE_BLOCK = 128
    block_mask = create_block_mask(score_mod, B, H, M, N, device=device)
    return block_mask

def eager_sdpa(query, key, value, score_mod, mask):
    return F.scaled_dot_product_attention(query, key, value, is_causal=True)

from triton.testing import do_bench

def test_mask(score_mod, mask_fn=None, B=16, H=16, S=8192, D=64, skip_correctness=False):
    if mask_fn is None:
        mask_fn = score_mod
    query = torch.randn(B, H, S, D, device="cuda", dtype=data_type, requires_grad=True)
    key = torch.randn(B, H, S, D, device="cuda", dtype=data_type, requires_grad=True)
    value = torch.randn(B, H, S, D, device="cuda", dtype=data_type, requires_grad=True)
    gradOut = torch.randn(B, H, S, D, device='cuda', dtype=data_type)

    # In this case assume that the mask only depends on q/kv, and so we can
    # broadcast the mask across batch and heads. If that's not the case, then
    # pass B and H instead of 1.
    block_mask = create_block_mask_from_score_mod(mask_fn, 1, 1, S, S, device=query.device)

    # Not needed for FlexAttention, only for F.scaled_dot_product_attention to check correctness.
    mask = create_mask(mask_fn, 1, 1, S, S, device=query.device)

    causal_fa2 = lambda: F.scaled_dot_product_attention(query, key, value, is_causal=True)
    xformers_mask = lambda: F.scaled_dot_product_attention(query, key, value, attn_mask=mask)
    flex_attention_call = lambda: flex_attention(query, key, value, score_mod=score_mod, block_mask=block_mask)

    print(score_mod.__name__)
    print("Forward: ")
    print("causal FA2: ", do_bench(causal_fa2))
    print("F.sdpa + mask: ", do_bench(xformers_mask))
    flex_ms = do_bench(flex_attention_call)
    print("flexattention: ", flex_ms)
    density = (100 - block_mask.sparsity())/100
    flops = (density * B * H * D * S * S)
    print("Flex FW FLOPS: ", 4 * flops * (1e3/flex_ms) / 1e12, "TF/s")

    causal_fa2_out = causal_fa2()
    xformers_out = xformers_mask()
    flex_out = flex_attention_call()
    print("Backward: ", )
    print("causal FA2: ", do_bench(lambda: causal_fa2_out.backward(gradOut, retain_graph=True)))
    flex_bw_ms = do_bench(lambda: flex_out.backward(gradOut, retain_graph=True))
    print("flexattention: ", flex_bw_ms)
    print("Flex BW FLOPS: ", 10 * flops * (1e3/flex_bw_ms) / 1e12, "TF/s")
    print(block_mask)
    print()
    if not skip_correctness:
        xformers_outs = []
        flex_outs = []

        query.grad = None
        key.grad = None
        value.grad = None

        out1 = xformers_mask()
        xformers_outs.append(out1)
        out1.backward(gradOut)
        xformers_outs += [query.grad, key.grad, value.grad]

        query.grad = None
        key.grad = None
        value.grad = None

        out2 = flex_attention_call()
        flex_outs.append(out2)
        out2.backward(gradOut)
        flex_outs += [query.grad, key.grad, value.grad]
        for flex, xformer in zip(flex_outs, xformers_outs):
            torch.testing.assert_close(flex, xformer, atol=1e-1, rtol=1e-2)

##################################
# Score mod examples start here!
##################################

################
# Full attention
################
def noop(score, b, h, q_idx, kv_idx):
    return score

################
# Standard causal mask
################
def causal_mask(score, b, h, q_idx, kv_idx):
    return torch.where(q_idx >= kv_idx, score, -float("inf"))

SLIDING_WINDOW = 1024

################
# Sliding window attention + causal 
################
def sliding_window_causal(score, b, h, q_idx, kv_idx):
    return torch.where((q_idx >= kv_idx) & (q_idx - kv_idx <= SLIDING_WINDOW), score, -float("inf"))

################
# prefix LM (bidirectional attention for first PREFIX_LENGTH tokens, then causal for the rest)
################
PREFIX_LENGTH = 2048
def prefix_lm_causal(score, b, h, q_idx, kv_idx):
    prefix_mask = kv_idx <= PREFIX_LENGTH
    causal_mask = q_idx >= kv_idx
    return torch.where(prefix_mask | causal_mask, score, -float("inf"))

################
# Document masking
################
# (Imagine that we have multiple documents of different lengths. We want to mask
# out the attention between documents, but allow attention between tokens within
# the same document. We can do this by using a document_id tensor that gives the
# document that each token belongs to. Then, we can mask out all attention
# scores where the document_id[q_idx] differs from document_id[kv_idx]

# Note: We *only* need to compile a new kernel when the `score_mod` changes
# (it'll automatically detect that using torch.compile infra). This example code
# is implemented with caching BlockMask, but in general, changing BlockMask
# *does not* require a recompile.

# That is, for document masking, we only need to compute a new BlockMask when
# the document lengths change, *not* a new kernel.
document_id = torch.zeros(32768, dtype=torch.int, device='cuda')
document_id[:4096] = 0
document_id[4096:8192] = 1
for i in range(8192, 32768, 8192):
    document_id[i:i+8192] = i // 8192 + 1

def document_masking_causal(score, b, h, q_idx, kv_idx):
    causal_mask = q_idx >= kv_idx
    document_mask = (document_id[q_idx] == document_id[kv_idx])
    return torch.where(causal_mask & document_mask, score, -float("inf"))

################
# Natten masking
################
# In this case, imagine that we have a 2D image of size (H x W) flattened into a
# sequence of tokens. We only want to attend to tokens within 8 "pixels", but
# from a 2D perspective.
#
# We can implement this score_mod by first translating the 1D position into the
# 2D coordinates. Then, we can simply check the distance of both coordinates to
# be within the window.
H = 128 
W = 128
WINDOW = 8

def get_x_y(idx):
    return idx // W, idx % W

def natten_mask(score, b, h, q_idx, kv_idx):
    q_x, q_y = get_x_y(q_idx)
    kv_x, kv_y = get_x_y(kv_idx)
    return torch.where(
        ((q_x - kv_x).abs() <= WINDOW) & ((q_y - kv_y).abs() <= WINDOW),
        score,
        -float("inf"),
    )

################
# Alibi Bias
################
# We are not restricted only to masking. For example, you can also implement
# alibi with this API.
alibi_bias = torch.randn(H, device='cuda')
def alibi_and_causal(score, b, h, q_idx, kv_idx):
    causal_mask = q_idx >= kv_idx
    bias = alibi_bias[h] * (q_idx - kv_idx)
    return torch.where(causal_mask, score + bias, -float("inf"))

################
# Tanh Soft-Capping
################
# We can also implement tanh soft-capping with this API.
# In this case, there are some nuances. In particular, the standard `tanh`
# operator in PyTorch (and CUDA/Triton) lowers to a numerically accurate but
# (relatively) quite slow implementation in SASS. See
# https://godbolt.org/z/W8afevWv1 for how the SASS looks like.
#
# So, in this case, we want to lower the `tanh` into the approximate tanh
# implementation. We can do so by register a custom operator in PyTorch and then
# an Inductor lowering.
@torch.library.custom_op("approx::tanh", mutates_args=())
def tanh_approx(inp: torch.Tensor) -> torch.Tensor:
    return torch.tanh(inp)

@tanh_approx.register_fake
def _(inp: torch.Tensor) -> torch.Tensor:
    return torch.tanh(inp)

# Some internal torch.compile details :P
from torch._inductor.virtualized import ops
from torch._inductor.lowering import make_pointwise, register_lowering

def tanh_approx_lowering(inp):
    fn = partial(ops.inline_asm_elementwise, asm="tanh.approx.f32 $0, $1;")
    return make_pointwise(fn)(inp)

register_lowering(torch.ops.approx.tanh)(tanh_approx_lowering)

class TanhApprox(torch.autograd.Function):
    @staticmethod
    def forward(x):
        return torch.ops.approx.tanh(x)

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, = inputs
        result = output
        ctx.save_for_backward(result)

    @staticmethod
    def backward(ctx, grad_output):
        result, = ctx.saved_tensors
        return grad_output * (1 - result * result)

tanh_approx = TanhApprox.apply
def tanh_soft_cap(score, b, h, q_idx, kv_idx):
    score = score / 2
    score = tanh_approx(score)
    score = score * 2
    return torch.where(q_idx >= kv_idx, score, -float("inf"))

test_mask(noop)
test_mask(causal_mask)
test_mask(sliding_window_causal)
test_mask(prefix_lm_causal)
test_mask(document_masking_causal, B=4, H=16, S=32768, D=64)
test_mask(natten_mask, B=4, H=16, S=H*W, D=64)

test_mask(alibi_and_causal, skip_correctness=True) # Biases more annoying to test correctness in our current setup
test_mask(tanh_soft_cap, mask_fn=causal_mask, skip_correctness=True)
