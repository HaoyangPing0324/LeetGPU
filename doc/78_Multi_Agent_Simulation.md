# Multi-Agent Simulation

> Follow the [LeetGPU repository](https://github.com/HaoyangPing0324/LeetGPU) for more.

## Problem Description

![Difficulty: Hard](../assets/common/difficulty_hard.svg)

[LeetGPU Challenge](https://leetgpu.com/challenges/multi-agent-simulation)

Implement a program for a multi-agent flocking simulation (boids). The input consists of:

An array `agents` containing `N` agents, where `N` is the total number of agents

Each agent occupies 4 consecutive 32-bit floating point numbers in the array: $[x, y, v_x, v_y]$, where:

- $(x, y)$ represents the agent's position in 2D space
- $(v_x, v_y)$ represents the agent's velocity vector

The total array size is `4 * N` floats, with agent $i$'s data stored at indices `[4i, 4i+1, 4i+2, 4i+3]`

### Simulation Rules

1.  For each agent $i$, identify all neighbors $j$ (where $i \neq j$) within radius $r = 5.0$ using: $$ \sqrt{(x_i - x_j)^2 + (y_i - y_j)^2} \< r $$
2.  Compute average velocity of neighboring agents: $$ \vec{v}\_{avg} = \begin{cases} \frac{1}{\|N_i\|} \sum\_{j \in N_i} \vec{v}\_j & \text{if } \|N_i\| \> 0 \\ \vec{v}\_i & \text{if } \|N_i\| = 0 \end{cases} $$ where $N_i$ is the set of neighbors for agent $i$
3.  Update velocity: $$ \vec{v}\_{new} = \vec{v} + \alpha(\vec{v}\_{avg} - \vec{v}), \text{ where } \alpha = 0.05 $$
4.  Update position: $$ \vec{p}\_{new} = \vec{p} + \vec{v}\_{new} $$

### Implementation Requirements

- Use only native features (external libraries are not permitted)
- The `solve` function signature must remain unchanged
- The final result must be stored in the `agents_next` array

### Example 1:

    Input: N = 2
    agents = [
      0.0, 0.0, 1.0, 0.0,    // Agent 0: [x, y, vx, vy]
      3.0, 4.0, 0.0, -1.0    // Agent 1: [x, y, vx, vy]
    ]

    Output:
    agents_next = [
      1.0, 0.0, 1.0, 0.0,    // Agent 0: [x, y, vx, vy]
      3.0, 3.0, 0.0, -1.0    // Agent 1: [x, y, vx, vy]
    ]

### Constraints

- 1 ≤ `N` ≤ 100,000
- Each agent's position and velocity components are 32-bit floats
- Performance is measured with `N` = 10,000

## CUDA

> To be developed.

## Triton

> To be developed.


## PyTorch

> To be developed.

## References

- [LeetGPU Challenge](https://leetgpu.com/challenges/multi-agent-simulation)

> Follow the [LeetGPU repository](https://github.com/HaoyangPing0324/LeetGPU) for more.
