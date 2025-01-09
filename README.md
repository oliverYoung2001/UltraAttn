# UltraAttn

## Code Hierarchy
### Scheduler Searching Engine (Scheduling)

#### Step1 Row/Column (Q,KV) Allocation Engine

1. Strategy1: Sequential
2. Strategy2: ZigZag
3. Smart Strategy ???

**Output**: A map from gpu ranks to token ids.

#### Step2 Workload Partition and Allocation Engine

**Method**: Solution Binary Search + ILP; 
**Baseline for Step 1&2**: Ring; ZigZag
**Output**: Comp Distribution Table

#### Step3 Parallel Graph Transformation Engine

**Method**: Greedy Algorithm
**Baseline**: No Transformation
**Output**: CC Dependency Graph

#### Step4. Lowering Engine

**Method**: ILP
**Baseline**: Flexflow
**Output**: Cuda Stream Graph

### DistAttn Engine (Executing)
