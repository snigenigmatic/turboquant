"""
TurboQuant : A fast Online KV Cache compression algorithm
Credit : https:#arxiv.org/pdf/2504.19874
"""

import numpy as np
from scipy.special import gamma
from scipy.integrate import quad
from scipy.optimize import minimize_scalar
from matplotlib import pyplot as plt
import time


def beta_pdf(x, d):
    """
    PDF of a single coordinate of a uniform random point on S^{d-1}.
    f_X(x) = Gamma(d/2) / (sqrt(pi) * Gamma((d-1)/2)) * (1 - x^2)^((d-3)/2)
    Supported on [-1, 1].
    """
    if abs(x) > 1.0:
        return 0.0
    else:
        return (
            gamma(d / 2)
            / (np.sqrt(np.pi) * gamma((d - 1) / 2))
            * (1 - x**2) ** ((d - 3) / 2)
        )


def beta_pdf_vec(x, d):
    """
    Vectorized version of beta_pdf.
    """
    coeff = gamma(d / 2) / (np.sqrt(np.pi) * gamma((d - 1) / 2))
    vals = np.where(np.abs(x) <= 1.0, coeff * (1 - x**2) ** ((d - 3) / 2), 0.0)
    return vals


def lloyd_max(d, num_levels, max_iter=200, tol=1e-12):
    """
    Find optimal scalar quantizer centroids for the Beta distribution on [-1, 1]
    using the Lloyd-Max iterative algorithm.

    This solves the continuous 1D k-means problem in Eq. (4) of the paper.

    Parameters
    ----------
    d : int - dimension (controls the Beta distribution shape)
    num_levels : int - number of quantization levels (2^b)
    max_iter : int - iteration cap
    tol : float - convergence tolerance on centroid movement

    Returns
    -------
    centroids : np.ndarray of shape (num_levels,) sorted ascending
    cost : float - the optimal MSE cost C(f_X, b)
    """
    sigma = 1.0 / np.sqrt(d)
    centroids = np.linspace(-3 * sigma, 3 * sigma, num_levels)

    def pdf(x):
        return beta_pdf(x, d)

    for iteration in range(max_iter):
        boundaries = np.concatenate(
            [[-1.0], (centroids[:-1] + centroids[1:]) / 2, [1.0]]
        )

        new_centroids = np.zeros(num_levels)
        total_cost = 0.0

        for i in range(num_levels):
            lo, hi = boundaries[i], boundaries[i + 1]
            if hi - lo < 1e-15:
                new_centroids[i] = centroids[i]
                continue

            numerator, _ = quad(lambda x: x * pdf(x), lo, hi, limit=100)
            denominator, _ = quad(pdf, lo, hi, limit=100)

            if denominator > 1e-15:
                new_centroids[i] = numerator / denominator
            else:
                new_centroids[i] = (lo + hi) / 2

            cost_i, _ = quad(
                lambda x: (x - new_centroids[i]) ** 2 * pdf(x), lo, hi, limit=100
            )
            total_cost += cost_i

        shift = np.max(np.abs(new_centroids - centroids))
        centroids = new_centroids

        if shift < tol:
            break

    return centroids, total_cost


def precompute_codebooks(d, max_bits=5):
    """
    Precompute optimal codebooks for bit-widths 1 through max_bits.
    This is done once and reused for all quantization calls.
    """
    codebooks = {}
    costs = {}
    for b in range(1, max_bits + 1):
        num_levels = 2**b
        centroids, cost = lloyd_max(d, num_levels)
        codebooks[b] = centroids
        costs[b] = cost
        print(
            f"  b={b}: {num_levels} levels, scalar cost C(f_X,{b}) = {cost:.6f}, "
            f"vector MSE = d*C = {d * cost:.6f}"
        )
    return codebooks, costs


# ------------ TurboQuant_mse (Algorithm 1) ------------
class TurboQuantMSE:
    """
    MSE-optimized vector quantizer.

    Pipeline:
      Quant:   x -> y = Pi @ x -> find nearest centroid per coordinate -> store indices
      DeQuant: indices -> look up centroids -> Pi^T @ centroids_vector

    """

    def __init__(self, d, b, codebook, rotation_matrix=None):
        self.d = d
        self.b = b
        self.codebook = codebook
        if rotation_matrix is None:
            self.Pi = self._random_rotation(d)
        else:
            self.Pi = rotation_matrix

    @staticmethod
    def _random_rotation(d):
        """Generate a random rotation matrix via QR decomposition of a Gaussian matrix."""
        G = np.random.randn(d, d)
        Q, R = np.linalg.qr(G)
        Q = Q @ np.diag(np.sign(np.diag(R)))
        return Q

    def quantize(self, x):
        """
        Quantize a vector x in R^d.
        Returns index array of shape (d,) with values in [0, 2^b - 1].
        """
        y = self.Pi @ x
        diffs = np.abs(y[:, None] - self.codebook[None, :])
        idx = np.argmin(diffs, axis=1)
        return idx

    def dequantize(self, idx):
        """
        Reconstruct vector from quantized indices.
        Returns x_tilde in R^d.
        """
        y_tilde = self.codebook[idx]
        x_tilde = self.Pi.T @ y_tilde
        return x_tilde

    def quantize_dequantize(self, x):
        """Full round-trip: quantize then dequantize."""
        idx = self.quantize(x)
        return self.dequantize(idx)


# ------------ QJL (Definition 1) : Quick 1-bit quantization  ------------
class QJL:
    """
    Quantization with Johnson-Lindenstrauss (QJL) for fast compression.
    Quant : x->sign(s@x) where s is a random sign vector.
    DeQuant : sqrt(pi/2) /d *s^T @ z
    """

    def __init__(self, d, projection_matrix=None):
        self.d = d
        if projection_matrix is not None:
            self.S = projection_matrix
        else:
            self.S = np.random.randn(d, d)

    def quantize(self, x):
        """Returns sign vector in {-1, +1}^d."""
        return np.sign(self.S @ x)

    def dequantize(self, z, gamma=1.0):
        """
        Dequantize sign vector z, scaled by gamma (= ||residual||).
        Returns reconstructed vector contribution.
        """
        return np.sqrt(np.pi / 2) / self.d * gamma * (self.S.T @ z)


# ------------ TurboQuant_prod (Algorithm 2) ------------
class TurboQuantProd:
    """
    Inner-product-optimized vector quantizer.

    Two-stage approach:
      1. Apply TurboQuant_mse with bit-width (b-1)
      2. Apply QJL on the residual (1 bit per coordinate)
      Total: b bits per coordinate, unbiased inner product estimator.
    """

    def __init__(self, d, b, codebooks):
        self.d = d
        self.b = b
        assert b >= 2, (
            "TurboQuant_prod needs b >= 2 (1 bit for MSE + 1 bit for QJL minimum)"
        )

        self.mse_quantizer = TurboQuantMSE(d, b - 1, codebooks[b - 1])
        self.qjl = QJL(d)

    def quantize(self, x):
        """
        Returns (idx, qjl_signs, residual_norm).
        """
        idx = self.mse_quantizer.quantize(x)
        x_mse = self.mse_quantizer.dequantize(idx)
        r = x - x_mse
        r_norm = np.linalg.norm(r)

        if r_norm > 1e-15:
            qjl_signs = self.qjl.quantize(r / r_norm)
        else:
            qjl_signs = np.ones(self.d)
            r_norm = 0.0

        return idx, qjl_signs, r_norm

    def dequantize(self, idx, qjl_signs, r_norm):
        """
        Reconstruct: x_tilde = x_mse + x_qjl
        """
        x_mse = self.mse_quantizer.dequantize(idx)
        x_qjl = self.qjl.dequantize(qjl_signs, gamma=r_norm)
        return x_mse + x_qjl

    def quantize_dequantize(self, x):
        idx, qjl_signs, r_norm = self.quantize(x)
        return self.dequantize(idx, qjl_signs, r_norm)


# ------------ Utils : random unit-vector generation ------------
def random_unit_vectors(n, d):
    """Sample n vectors uniformly from S^{d-1}."""
    X = np.random.randn(n, d)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / norms


# ------------ Experiments ------------
def experiment_coordinate_distribution(d=256, n_samples=50000):
    """
    Verify Lemma 1: after random rotation, each coordinate follows the Beta distribution.
    """
    print(f"\n{'=' * 60}")
    print(f"Experiment 1: Coordinate distribution verification (d={d})")
    print(f"{'=' * 60}")

    X = random_unit_vectors(n_samples, d)
    Pi = TurboQuantMSE._random_rotation(d)
    Y = (Pi @ X.T).T

    coord_0 = Y[:, 0]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(
        coord_0, bins=80, density=True, alpha=0.7, color="steelblue", label="Empirical"
    )
    x_range = np.linspace(-4 / np.sqrt(d), 4 / np.sqrt(d), 300)
    pdf_vals = beta_pdf_vec(x_range, d)
    axes[0].plot(x_range, pdf_vals, "r-", lw=2, label=f"Beta PDF (d={d})")
    gaussian_vals = np.sqrt(d / (2 * np.pi)) * np.exp(-d * x_range**2 / 2)
    axes[0].plot(x_range, gaussian_vals, "g--", lw=2, label=f"N(0, 1/{d})")
    axes[0].set_title("Single coordinate distribution after rotation")
    axes[0].set_xlabel("Coordinate value")
    axes[0].set_ylabel("Density")
    axes[0].legend()

    coord_0_centered = coord_0 - np.mean(coord_0)
    coord_1_centered = Y[:, 1] - np.mean(Y[:, 1])
    corr = np.corrcoef(coord_0_centered, coord_1_centered)[0, 1]
    axes[1].scatter(Y[:, 0], Y[:, 1], s=1, alpha=0.3)
    axes[1].set_title(f"Coord 0 vs Coord 1 (correlation = {corr:.4f})")
    axes[1].set_xlabel("y[0]")
    axes[1].set_ylabel("y[1]")
    axes[1].set_aspect("equal")

    plt.tight_layout()
    plt.savefig("exp1_coordinate_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(
        f"  Empirical variance of coord 0: {np.var(coord_0):.6f} (expected: {1 / d:.6f})"
    )
    print(f"  Correlation between coord 0 and coord 1: {corr:.6f} (expected: ~0)")
    print(f"  Saved: exp1_coordinate_distribution.png")


def experiment_mse_vs_bitwidth(d=256, n_vectors=2000, max_bits=5):
    """
    Verify Theorem 1: empirical MSE matches theoretical bounds.
    """
    print(f"\n{'=' * 60}")
    print(f"Experiment 2: MSE vs bit-width (d={d}, n={n_vectors})")
    print(f"{'=' * 60}")

    print("  Precomputing codebooks...")
    codebooks, scalar_costs = precompute_codebooks(d, max_bits)

    X = random_unit_vectors(n_vectors, d)

    empirical_mse = []
    bits_range = list(range(1, max_bits + 1))

    for b in bits_range:
        quant = TurboQuantMSE(d, b, codebooks[b])
        mse_sum = 0.0
        for i in range(n_vectors):
            x_tilde = quant.quantize_dequantize(X[i])
            mse_sum += np.sum((X[i] - x_tilde) ** 2)
        avg_mse = mse_sum / n_vectors
        empirical_mse.append(avg_mse)
        print(f"  b={b}: empirical MSE = {avg_mse:.6f}")

    upper_bound = [np.sqrt(3 * np.pi) / 2 * 4 ** (-b) for b in bits_range]
    lower_bound = [4 ** (-b) for b in bits_range]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(
        bits_range, empirical_mse, "bo-", lw=2, ms=8, label="TurboQuant_mse (empirical)"
    )
    ax.semilogy(
        bits_range,
        upper_bound,
        "r--",
        lw=2,
        label=r"Upper bound: $\frac{\sqrt{3\pi}}{2} \cdot 4^{-b}$",
    )
    ax.semilogy(bits_range, lower_bound, "g--", lw=2, label=r"Lower bound: $4^{-b}$")
    ax.set_xlabel("Bit-width (b)")
    ax.set_ylabel("MSE Distortion")
    ax.set_title(f"MSE Distortion vs Bit-width (d={d})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(bits_range)
    plt.tight_layout()
    plt.savefig("exp2_mse_vs_bitwidth.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: exp2_mse_vs_bitwidth.png")

    return codebooks, scalar_costs


def experiment_inner_product(
    d=256, n_train=2000, n_query=200, max_bits=5, codebooks=None
):
    """
    Verify Theorem 2: inner product distortion and unbiasedness.
    Compare TurboQuant_mse (biased) vs TurboQuant_prod (unbiased).
    """
    print(f"\n{'=' * 60}")
    print(f"Experiment 3: Inner product distortion (d={d})")
    print(f"{'=' * 60}")

    if codebooks is None:
        print("  Precomputing codebooks...")
        codebooks, _ = precompute_codebooks(d, max_bits)

    X = random_unit_vectors(n_train, d)
    Y = random_unit_vectors(n_query, d)

    bits_range = list(range(2, max_bits + 1))

    mse_ip_errors = []
    mse_ip_biases = []
    prod_ip_errors = []
    prod_ip_biases = []

    for b in bits_range:
        print(f"\n  Bit-width b={b}:")

        quant_mse = TurboQuantMSE(d, b, codebooks[b])
        quant_prod = TurboQuantProd(d, b, codebooks)

        mse_errors_b = []
        mse_signed_b = []
        prod_errors_b = []
        prod_signed_b = []

        for i in range(min(n_train, 500)):
            x = X[i]
            x_tilde_mse = quant_mse.quantize_dequantize(x)
            x_tilde_prod = quant_prod.quantize_dequantize(x)

            for j in range(min(n_query, 50)):
                y = Y[j]
                true_ip = np.dot(y, x)

                est_mse = np.dot(y, x_tilde_mse)
                err_mse = est_mse - true_ip
                mse_errors_b.append(err_mse**2)
                mse_signed_b.append(err_mse)

                est_prod = np.dot(y, x_tilde_prod)
                err_prod = est_prod - true_ip
                prod_errors_b.append(err_prod**2)
                prod_signed_b.append(err_prod)

        avg_mse_err = np.mean(mse_errors_b)
        avg_mse_bias = np.mean(mse_signed_b)
        avg_prod_err = np.mean(prod_errors_b)
        avg_prod_bias = np.mean(prod_signed_b)

        mse_ip_errors.append(avg_mse_err)
        mse_ip_biases.append(avg_mse_bias)
        prod_ip_errors.append(avg_prod_err)
        prod_ip_biases.append(avg_prod_bias)

        print(
            f"    TurboQuant_mse:  IP error = {avg_mse_err:.6f}, bias = {avg_mse_bias:.6f}"
        )
        print(
            f"    TurboQuant_prod: IP error = {avg_prod_err:.6f}, bias = {avg_prod_bias:.6f}"
        )

    upper_bound = [np.sqrt(3 * np.pi) / 2 / d * 4 ** (-b) for b in bits_range]
    lower_bound = [1 / d * 4 ** (-b) for b in bits_range]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].semilogy(
        bits_range, mse_ip_errors, "bs-", lw=2, ms=8, label="TurboQuant_mse"
    )
    axes[0].semilogy(
        bits_range, prod_ip_errors, "ro-", lw=2, ms=8, label="TurboQuant_prod"
    )
    axes[0].semilogy(
        bits_range,
        upper_bound,
        "r--",
        lw=1.5,
        label=r"Upper: $\frac{\sqrt{3\pi}}{2d} \cdot 4^{-b}$",
    )
    axes[0].semilogy(
        bits_range,
        lower_bound,
        "g--",
        lw=1.5,
        label=r"Lower: $\frac{1}{d} \cdot 4^{-b}$",
    )
    axes[0].set_xlabel("Bit-width (b)")
    axes[0].set_ylabel("Inner Product Error (variance)")
    axes[0].set_title("Inner Product Distortion vs Bit-width")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xticks(bits_range)

    axes[1].bar(
        np.array(bits_range) - 0.15,
        mse_ip_biases,
        0.3,
        label="TurboQuant_mse",
        color="steelblue",
    )
    axes[1].bar(
        np.array(bits_range) + 0.15,
        prod_ip_biases,
        0.3,
        label="TurboQuant_prod",
        color="salmon",
    )
    axes[1].axhline(y=0, color="black", lw=0.8)
    axes[1].set_xlabel("Bit-width (b)")
    axes[1].set_ylabel("Mean signed error (bias)")
    axes[1].set_title("Bias in Inner Product Estimation")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xticks(bits_range)

    plt.tight_layout()
    plt.savefig("exp3_inner_product.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved: exp3_inner_product.png")


def experiment_error_histograms(d=256, n_pairs=10000, codebooks=None, max_bits=5):
    """
    Reproduce Figure 1 from the paper: error distribution histograms.
    """
    print(f"\n{'=' * 60}")
    print(f"Experiment 4: Error distribution histograms (d={d})")
    print(f"{'=' * 60}")

    if codebooks is None:
        print("  Precomputing codebooks...")
        codebooks, _ = precompute_codebooks(d, max_bits)

    X = random_unit_vectors(n_pairs, d)
    Y = random_unit_vectors(n_pairs, d)

    fig, axes = plt.subplots(2, 4, figsize=(16, 7))

    for col, b in enumerate([1, 2, 3, 4]):
        print(f"  Computing b={b}...")

        if b >= 2:
            quant_prod = TurboQuantProd(d, b, codebooks)
        quant_mse = TurboQuantMSE(d, b, codebooks[b])

        prod_errors = []
        mse_errors = []

        for i in range(n_pairs):
            true_ip = np.dot(Y[i], X[i])

            if b >= 2:
                x_prod = quant_prod.quantize_dequantize(X[i])
                prod_errors.append(np.dot(Y[i], x_prod) - true_ip)

            x_mse = quant_mse.quantize_dequantize(X[i])
            mse_errors.append(np.dot(Y[i], x_mse) - true_ip)

        if b >= 2 and len(prod_errors) > 0:
            axes[0, col].hist(
                prod_errors, bins=60, density=True, color="steelblue", alpha=0.8
            )
            axes[0, col].set_title(f"TurboQuant_prod  b={b}")
            axes[0, col].axvline(x=0, color="red", lw=1, ls="--")
            mean_val = np.mean(prod_errors)
            axes[0, col].axvline(
                x=mean_val, color="orange", lw=1.5, ls="-", label=f"mean={mean_val:.4f}"
            )
            axes[0, col].legend(fontsize=8)
        else:
            axes[0, col].text(
                0.5,
                0.5,
                f"b={b}: needs b>=2\nfor prod variant",
                ha="center",
                va="center",
                transform=axes[0, col].transAxes,
            )
            axes[0, col].set_title(f"TurboQuant_prod  b={b}")

        axes[1, col].hist(mse_errors, bins=60, density=True, color="salmon", alpha=0.8)
        axes[1, col].set_title(f"TurboQuant_mse  b={b}")
        axes[1, col].axvline(x=0, color="red", lw=1, ls="--")
        mean_val = np.mean(mse_errors)
        axes[1, col].axvline(
            x=mean_val, color="orange", lw=1.5, ls="-", label=f"mean={mean_val:.4f}"
        )
        axes[1, col].legend(fontsize=8)

    for ax_row in axes:
        for ax in ax_row:
            ax.set_xlabel("IP error")

    plt.suptitle(f"Inner Product Error Distributions (d={d})", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig("exp4_error_histograms.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: exp4_error_histograms.png")


def experiment_nearest_neighbor(
    d=256, n_db=5000, n_query=100, codebooks=None, max_bits=5
):
    """
    Simple nearest neighbor recall experiment.
    Quantize database vectors, compute approximate inner products, check recall.
    """
    print(f"\n{'=' * 60}")
    print(f"Experiment 5: Nearest neighbor recall (d={d})")
    print(f"{'=' * 60}")

    if codebooks is None:
        print("  Precomputing codebooks...")
        codebooks, _ = precompute_codebooks(d, max_bits)

    DB = random_unit_vectors(n_db, d)
    Q = random_unit_vectors(n_query, d)

    true_ip = Q @ DB.T
    true_top1 = np.argmax(true_ip, axis=1)

    results = {}

    for b in [2, 3, 4]:
        print(f"\n  Bit-width b={b}:")

        quant_mse = TurboQuantMSE(d, b, codebooks[b])

        t0 = time.time()
        DB_quantized = np.zeros_like(DB)
        for i in range(n_db):
            DB_quantized[i] = quant_mse.quantize_dequantize(DB[i])
        quant_time = time.time() - t0

        approx_ip = Q @ DB_quantized.T

        recalls = {}
        for k in [1, 5, 10, 20, 50]:
            approx_topk = np.argsort(-approx_ip, axis=1)[:, :k]
            hits = sum(1 for i in range(n_query) if true_top1[i] in approx_topk[i])
            recalls[k] = hits / n_query

        results[b] = recalls
        print(f"    Quantization time: {quant_time:.3f}s")
        print(
            f"    Recall@1@1={recalls[1]:.3f}, @1@5={recalls[5]:.3f}, "
            f"@1@10={recalls[10]:.3f}, @1@20={recalls[20]:.3f}, @1@50={recalls[50]:.3f}"
        )

    fig, ax = plt.subplots(figsize=(8, 5))
    for b, recalls in results.items():
        ks = sorted(recalls.keys())
        ax.plot(
            ks, [recalls[k] for k in ks], "o-", lw=2, ms=7, label=f"TurboQuant b={b}"
        )
    ax.set_xlabel("Top-k")
    ax.set_ylabel("Recall@1@k")
    ax.set_title(f"Nearest Neighbor Recall (d={d}, n_db={n_db})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig("exp5_nn_recall.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved: exp5_nn_recall.png")


def experiment_bias_vs_true_ip(d=256, n_pairs=5000, codebooks=None, max_bits=5):
    """
    Reproduce Figure 2 from the paper: show that TurboQuant_mse bias depends on
    the magnitude of the true inner product, while TurboQuant_prod stays unbiased.
    """
    print(f"\n{'=' * 60}")
    print(f"Experiment 6: Bias vs true inner product magnitude (d={d})")
    print(f"{'=' * 60}")

    if codebooks is None:
        print("  Precomputing codebooks...")
        codebooks, _ = precompute_codebooks(d, max_bits)

    b = 2
    quant_mse = TurboQuantMSE(d, b, codebooks[b])
    quant_prod = TurboQuantProd(d, b, codebooks)

    X = random_unit_vectors(n_pairs, d)
    Y = random_unit_vectors(n_pairs, d)

    true_ips = []
    mse_errs = []
    prod_errs = []

    for i in range(n_pairs):
        true_ip = np.dot(Y[i], X[i])
        true_ips.append(true_ip)

        x_mse = quant_mse.quantize_dequantize(X[i])
        mse_errs.append(np.dot(Y[i], x_mse) - true_ip)

        x_prod = quant_prod.quantize_dequantize(X[i])
        prod_errs.append(np.dot(Y[i], x_prod) - true_ip)

    true_ips = np.array(true_ips)
    mse_errs = np.array(mse_errs)
    prod_errs = np.array(prod_errs)

    n_bins = 8
    bins = np.percentile(true_ips, np.linspace(0, 100, n_bins + 1))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, errs, name, color in [
        (axes[0], prod_errs, "TurboQuant_prod", "steelblue"),
        (axes[1], mse_errs, "TurboQuant_mse", "salmon"),
    ]:
        bin_means = []
        bin_biases = []
        bin_labels = []
        for k in range(n_bins):
            mask = (true_ips >= bins[k]) & (true_ips < bins[k + 1])
            if k == n_bins - 1:
                mask = (true_ips >= bins[k]) & (true_ips <= bins[k + 1])
            if np.sum(mask) > 10:
                bin_means.append(np.mean(true_ips[mask]))
                bin_biases.append(np.mean(errs[mask]))

        ax.bar(range(len(bin_means)), bin_biases, color=color, alpha=0.8)
        ax.set_xticks(range(len(bin_means)))
        ax.set_xticklabels([f"{m:.3f}" for m in bin_means], rotation=45, fontsize=8)
        ax.axhline(y=0, color="black", lw=0.8)
        ax.set_xlabel("Average true inner product in bin")
        ax.set_ylabel("Mean signed error (bias)")
        ax.set_title(f"{name} (b={b})")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("exp6_bias_vs_ip.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: exp6_bias_vs_ip.png")


# ---------------------------------------------------------------------------
# 8. Main
# ---------------------------------------------------------------------------


def main():
    np.random.seed(42)

    d = 256
    max_bits = 5

    print("TurboQuant Proof of Concept")
    print("=" * 60)
    print(f"Dimension d = {d}")
    print(f"Max bit-width = {max_bits}")

    experiment_coordinate_distribution(d=d)

    codebooks, costs = experiment_mse_vs_bitwidth(
        d=d, n_vectors=1000, max_bits=max_bits
    )

    experiment_inner_product(
        d=d, n_train=500, n_query=100, max_bits=max_bits, codebooks=codebooks
    )

    experiment_error_histograms(
        d=d, n_pairs=5000, codebooks=codebooks, max_bits=max_bits
    )

    experiment_nearest_neighbor(
        d=d, n_db=3000, n_query=100, codebooks=codebooks, max_bits=max_bits
    )

    experiment_bias_vs_true_ip(
        d=d, n_pairs=3000, codebooks=codebooks, max_bits=max_bits
    )

    print("\n" + "=" * 60)
    print("All experiments complete. Output images:")
    print("  exp1_coordinate_distribution.png  - Lemma 1 verification")
    print("  exp2_mse_vs_bitwidth.png          - Theorem 1 verification")
    print("  exp3_inner_product.png            - Theorem 2 verification")
    print("  exp4_error_histograms.png         - Figure 1 reproduction")
    print("  exp5_nn_recall.png                - NN search recall")
    print("  exp6_bias_vs_ip.png               - Figure 2 reproduction")


if __name__ == "__main__":
    main()
