# TurboQuant: Proof of Concept

A from-scratch implementation of **TurboQuant** (Zandieh et al., 2025) — an online vector quantization algorithm with near-optimal distortion rate, based on the paper [TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate](https://arxiv.org/abs/2504.19874).

## What is TurboQuant?

TurboQuant compresses high-dimensional vectors from 16-bit floats down to 1–5 bits per coordinate while preserving their geometric structure (inner products and distances). It's designed for online use cases like LLM KV cache compression and nearest neighbor search, where you can't afford expensive offline codebook training.

The core idea: randomly rotate the input vector so each coordinate follows a known Beta distribution, then apply an optimal scalar quantizer per coordinate. Because coordinates become nearly independent in high dimensions, this simple per-coordinate approach achieves distortion within ~2.7x of the information-theoretic lower bound.

## What's implemented

Both algorithms from the paper:

- **TurboQuant_mse** (Algorithm 1): Minimizes mean-squared reconstruction error. Random rotation → per-coordinate Lloyd-Max quantization → inverse rotation.
- **TurboQuant_prod** (Algorithm 2): Provides unbiased inner product estimation. Applies TurboQuant_mse at (b-1) bits, then a 1-bit QJL transform on the residual.

Supporting components:

- **Lloyd-Max solver**: Finds optimal scalar quantization codebooks for the Beta distribution by solving the continuous 1D k-means problem (Eq. 4 in the paper).
- **QJL (Quantized Johnson-Lindenstrauss)**: 1-bit inner product quantizer from [Zandieh et al., 2024](https://arxiv.org/abs/2406.03482). Maps vectors to sign bits via random Gaussian projection.

## Experiments

Six experiments that validate the paper's theoretical results on synthetic unit-sphere vectors (d=256):

| Experiment | What it verifies | Key finding |
|---|---|---|
| 1. Coordinate distribution | Lemma 1 | After rotation, coordinates follow Beta ≈ N(0, 1/d), correlation ≈ 0 |
| 2. MSE vs bit-width | Theorem 1 | Empirical MSE matches predicted values (0.36, 0.117, 0.03, 0.009 for b=1–4) |
| 3. Inner product distortion | Theorem 2 | TurboQuant_prod is unbiased; TurboQuant_mse has lower variance but is biased |
| 4. Error histograms | Figure 1 | Prod errors centered at zero; MSE errors shifted, shrinking with bit-width |
| 5. NN recall | Section 4.4 | 3-bit: perfect recall@10; 4-bit: 81% recall@1 |
| 6. Bias vs IP magnitude | Figure 2 | MSE bias proportional to true IP (shrinkage toward zero); prod bias is flat noise |

## Usage

```bash
pip install numpy scipy matplotlib
python turboquant_poc.py
```

Outputs six PNG plots in the working directory. Takes about a minute on a laptop CPU.

To experiment, edit the parameters in `main()`:

```python
d = 256        # vector dimension (try 64, 128, 512, 1024)
max_bits = 5   # bit-widths to test (1 through max_bits)
```

Higher `d` makes the Gaussian approximation tighter and coordinates more independent, so the algorithm works even better. Lower `d` (like 64) shows where the approximation starts to break down.

## How the code maps to the paper

| Code | Paper |
|---|---|
| `beta_pdf()` | Lemma 1: f_X(x) distribution |
| `lloyd_max()` | Eq. 4: continuous 1D k-means for optimal codebook |
| `TurboQuantMSE` | Algorithm 1 |
| `QJL` | Definition 1 |
| `TurboQuantProd` | Algorithm 2 |
| `experiment_mse_vs_bitwidth()` | Figure 3b, Theorem 1 bounds |
| `experiment_inner_product()` | Figure 3a, Theorem 2 bounds |
| `experiment_error_histograms()` | Figure 1 |
| `experiment_bias_vs_true_ip()` | Figure 2 |

## Limitations of this PoC

This is a reference implementation for understanding, not production use:

- Pure NumPy, no GPU acceleration or vectorized batch quantization
- Random rotation via dense matrix multiply (O(d²) per vector; production would use a fast structured rotation like a randomized Hadamard transform)
- Codebook lookup is brute-force argmin over centroids
- No outlier channel splitting (the paper's 2.5-bit and 3.5-bit configurations use separate bit allocations for outlier vs normal channels)
- No entropy coding of codebook indices (Section 3.1 mentions a ~5% bit-width reduction from this)

## References

- Zandieh, A., Daliri, M., Hadian, M., Mirrokni, V. (2025). *TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate*. [arXiv:2504.19874](https://arxiv.org/abs/2504.19874)
- Zandieh, A., Daliri, M., Han, I. (2024). *QJL: 1-bit Quantized JL Transform for KV Cache Quantization with Zero Overhead*. [arXiv:2406.03482](https://arxiv.org/abs/2406.03482)
- Shannon, C. E. (1959). *Coding Theorems for a Discrete Source with a Fidelity Criterion*. IRE Nat. Conv. Rec.
- Lloyd, S. (1982). *Least Squares Quantization in PCM*. IEEE Trans. Information Theory.