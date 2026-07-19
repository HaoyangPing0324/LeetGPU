# Token Embedding Layer

![Difficulty: Medium](../assets/common/difficulty_medium.svg)

[Original LeetGPU Challenge](https://leetgpu.com/challenges/token-embedding-layer)

Implement the input embedding layer used at the start of transformer models such as BERT. For each token in a batch, gather a row from the `token_embeddings` table using `token_ids`, gather a row from the `position_embeddings` table using `position_ids`, sum the two vectors, and apply Layer Normalization with learnable scale and shift parameters along the embedding dimension.

Formally, for batch index $b$ and time step $t$, let $$ s\_{b,t} = E_T\[\text{token\\ids}\_{b,t}\] + E_P\[\text{position\\ids}\_{t}\] \in \mathbb{R}^{D} $$ where $E_T \in \mathbb{R}^{V \times D}$ and $E_P \in \mathbb{R}^{P \times D}$ are the token and positional embedding tables. The output is then $$ \mu\_{b,t} = \frac{1}{D} \sum\_{d=1}^{D} s\_{b,t,d}, \qquad \sigma^2\_{b,t} = \frac{1}{D} \sum\_{d=1}^{D} (s\_{b,t,d} - \mu\_{b,t})^2, $$ $$ y\_{b,t,d} = \gamma_d \cdot \frac{s\_{b,t,d} - \mu\_{b,t}}{\sqrt{\sigma^2\_{b,t} + \epsilon}} + \beta_d. $$

### Implementation Requirements

- External libraries are not permitted
- The `solve` function signature must remain unchanged
- `token_ids` has shape `(B, T)` and `position_ids` has shape `(T)`, both with `int32` values
- `token_embeddings` has shape `(V, D)` and `position_embeddings` has shape `(P, D)`
- `gamma` and `beta` each have shape `(D)`
- The variance is computed without Bessel's correction (divide by `D`, not `D-1`)
- The final result must be stored in the `output` tensor with shape `(B, T, D)`

### Example 1:

    Input:  B = 1, T = 2, V = 3, P = 2, D = 4, eps = 1e-5
            token_ids        = [[2, 0]]
            position_ids     = [0, 1]
            token_embeddings = [[ 1.0,  2.0,  3.0,  4.0],
                                [ 0.0,  1.0,  0.0, -1.0],
                                [ 2.0,  0.0, -2.0,  0.0]]
            position_embeddings = [[ 0.0,  0.0,  0.0,  0.0],
                                   [ 1.0,  0.0, -1.0,  0.0]]
            gamma = [1.0, 1.0, 1.0, 1.0]
            beta  = [0.0, 0.0, 0.0, 0.0]
    Output: output = [[[ 1.4142,  0.0000, -1.4142,  0.0000],
                       [-0.5773, -0.5773, -0.5773,  1.7320]]]

### Constraints

- 1 ≤ `B` ≤ 64
- 1 ≤ `T` ≤ 1,024
- 1 ≤ `V` ≤ 50,000
- 1 ≤ `P` ≤ 4,096
- 1 ≤ `D` ≤ 1,024
- 0 ≤ `token_ids` values \< `V`
- 0 ≤ `position_ids` values \< `P`
- `eps` = 1e-5
- Performance is measured with `B` = 32, `T` = 512, `V` = 30,000, `P` = 2,048, `D` = 768
