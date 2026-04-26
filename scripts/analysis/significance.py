"""
Significance analysis utilities for paired binary outcome comparisons.

Usage (CLI):
    python scripts/analysis/significance.py <run_dir_a> <run_dir_b>

Each run directory must contain either:
  - trajectories.json     (list of trajectory dicts)
  - agent_trajectories.jsonl  (append-only JSONL, one dict per line)

The two directories must have been run on the same task list (paired outcomes).
Tasks are matched by task_id. Only tasks present in both runs are compared.

Functions:
    mcnemar_exact      -- McNemar's exact test (binomial-based) on paired binary outcomes
    paired_bootstrap   -- Paired bootstrap on per-task success rates
    holm_bonferroni    -- Holm-Bonferroni family-wise error-rate correction
"""

import json
import math
import os
import sys
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# Core statistical functions
# ---------------------------------------------------------------------------

def _binomial_cdf(k: int, n: int, p: float) -> float:
    """Compute P(X <= k) for X ~ Binomial(n, p) using the regularised
    incomplete beta function via a continued-fraction approximation.
    This is a pure-stdlib fallback used when scipy is unavailable."""
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    # Use the regularised incomplete beta identity:
    # P(X <= k) = I_{1-p}(n-k, k+1)
    # where I_x(a, b) = 1 - I_{1-x}(b, a)
    # We evaluate I_x(a, b) via a simple series expansion (for moderate n).
    # For large n this may lose precision; install scipy for production use.
    total = 0.0
    log_p = math.log(p) if p > 0 else -1e300
    log_1mp = math.log(1 - p) if p < 1 else -1e300
    # Compute log(n choose i) iteratively
    log_coeff = 0.0
    for i in range(k + 1):
        if i == 0:
            log_coeff = 0.0  # log(C(n,0)) = 0
        else:
            log_coeff += math.log(n - i + 1) - math.log(i)
        term = math.exp(log_coeff + i * log_p + (n - i) * log_1mp)
        total += term
    return min(total, 1.0)


def _binomtest_pvalue(b: int, c: int) -> float:
    """Return the two-sided p-value for McNemar's exact test.

    Under H0 the discordant counts (b, c) satisfy b ~ Binomial(b+c, 0.5).
    The two-sided p-value is 2 * P(X <= min(b,c)) clipped to [0, 1].
    Falls back to a pure-stdlib implementation when scipy is unavailable.
    """
    n = b + c
    if n == 0:
        return 1.0
    try:
        from scipy.stats import binomtest  # type: ignore
        result = binomtest(min(b, c), n, 0.5, alternative="two-sided")
        return float(result.pvalue)
    except ImportError:
        pass
    # Pure-stdlib fallback
    p_val = 2.0 * _binomial_cdf(min(b, c), n, 0.5)
    return min(p_val, 1.0)


def mcnemar_exact(
    success_a: List[bool],
    success_b: List[bool],
) -> Tuple[float, Dict[str, int]]:
    """McNemar's exact test on paired binary outcomes.

    Counts the four cells of the 2x2 contingency table:
      - n00: both fail
      - n01: A fails, B succeeds
      - n10: A succeeds, B fails
      - n11: both succeed

    The test statistic uses only the discordant cells (n01, n10).
    H0: the marginal success rates are equal (i.e., n01 == n10 in expectation).

    Args:
        success_a: Binary outcomes for method A (True = success).
        success_b: Binary outcomes for method B (True = success).

    Returns:
        Tuple of (p_value, agreement_matrix) where agreement_matrix has keys
        "n00", "n01", "n10", "n11".

    Raises:
        ValueError: if the two lists have different lengths.
    """
    if len(success_a) != len(success_b):
        raise ValueError(
            f"success_a and success_b must have the same length "
            f"({len(success_a)} vs {len(success_b)})"
        )
    n00 = n01 = n10 = n11 = 0
    for a, b in zip(success_a, success_b):
        if not a and not b:
            n00 += 1
        elif not a and b:
            n01 += 1
        elif a and not b:
            n10 += 1
        else:
            n11 += 1

    p_value = _binomtest_pvalue(n01, n10)
    agreement = {"n00": n00, "n01": n01, "n10": n10, "n11": n11}
    return p_value, agreement


def paired_bootstrap(
    success_a: List[bool],
    success_b: List[bool],
    n_resamples: int = 10000,
    seed: int = 0,
) -> Dict[str, float]:
    """Paired bootstrap on per-task success rates.

    Resamples task indices with replacement and computes the success-rate
    delta (mean_b - mean_a) on each resample. Returns the mean delta,
    a 95% confidence interval, and a two-sided p-value.

    The p-value is estimated as the fraction of bootstrap deltas whose
    absolute value is >= |observed_delta|; equivalently, we shift the
    bootstrap distribution to be centred at 0 under H0 and count
    how often |bootstrap_delta - observed_delta| >= |observed_delta|.

    Args:
        success_a: Binary outcomes for method A.
        success_b: Binary outcomes for method B.
        n_resamples: Number of bootstrap resamples (default: 10000).
        seed: RNG seed for reproducibility (default: 0).

    Returns:
        Dict with keys: "observed_delta", "mean_bootstrap_delta",
        "ci_low", "ci_high", "p_value", "n_tasks", "n_resamples".
    """
    if len(success_a) != len(success_b):
        raise ValueError(
            f"success_a and success_b must have the same length "
            f"({len(success_a)} vs {len(success_b)})"
        )
    import random as _random
    rng = _random.Random(seed)

    n = len(success_a)
    if n == 0:
        return {
            "observed_delta": 0.0, "mean_bootstrap_delta": 0.0,
            "ci_low": 0.0, "ci_high": 0.0, "p_value": 1.0,
            "n_tasks": 0, "n_resamples": n_resamples,
        }

    # Observed delta: rate_b - rate_a
    observed_delta = (sum(success_b) - sum(success_a)) / n

    # Resample
    deltas: List[float] = []
    for _ in range(n_resamples):
        indices = [rng.randint(0, n - 1) for _ in range(n)]
        resample_a = sum(success_a[i] for i in indices)
        resample_b = sum(success_b[i] for i in indices)
        deltas.append((resample_b - resample_a) / n)

    deltas.sort()
    mean_delta = sum(deltas) / n_resamples
    ci_low = deltas[int(0.025 * n_resamples)]
    ci_high = deltas[int(0.975 * n_resamples)]

    # Two-sided p-value: shift distribution to H0 (centred at 0) and count
    # how often the shifted bootstrap delta is as extreme as observed.
    extreme = sum(
        1 for d in deltas if abs(d - observed_delta) >= abs(observed_delta)
    )
    p_value = extreme / n_resamples

    return {
        "observed_delta": observed_delta,
        "mean_bootstrap_delta": mean_delta,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_value": p_value,
        "n_tasks": n,
        "n_resamples": n_resamples,
    }


def holm_bonferroni(
    p_values: List[float],
    alpha: float = 0.05,
) -> List[bool]:
    """Holm-Bonferroni step-down correction for family-wise error rate.

    Rejects H_i if p_{(i)} <= alpha / (m - i + 1) for all j <= i where
    no earlier hypothesis was retained. Returns a list of booleans
    (True = reject H0, False = fail to reject) in the original order.

    Args:
        p_values: List of unadjusted p-values.
        alpha: Family-wise significance level (default: 0.05).

    Returns:
        List of booleans of the same length as p_values.
    """
    m = len(p_values)
    if m == 0:
        return []

    # Pair (p_value, original_index) and sort ascending
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    rejected = [False] * m

    for rank, (orig_idx, p) in enumerate(indexed):
        threshold = alpha / (m - rank)
        if p <= threshold:
            rejected[orig_idx] = True
        else:
            # Once we fail to reject, all remaining hypotheses also fail
            break

    return rejected


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _load_trajectories(run_dir: str) -> List[Dict]:
    """Load trajectories from a run directory.

    Tries agent_trajectories.jsonl first (memory variants), then
    trajectories.json (react / reflexion). Returns a list of dicts.
    """
    jsonl_path = os.path.join(run_dir, "agent_trajectories.jsonl")
    json_path = os.path.join(run_dir, "trajectories.json")

    if os.path.exists(jsonl_path):
        records = []
        with open(jsonl_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return records

    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        return []

    raise FileNotFoundError(
        f"No trajectory file found in {run_dir}. "
        f"Expected agent_trajectories.jsonl or trajectories.json."
    )


def _build_success_map(trajectories: List[Dict]) -> Dict[str, bool]:
    """Return {task_id: success} keeping the last trial for each task."""
    result: Dict[str, bool] = {}
    for t in trajectories:
        task_id = t.get("task_id", "")
        if not task_id:
            continue
        result[task_id] = bool(t.get("success", False))
    return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    if len(sys.argv) < 3:
        print("Usage: python significance.py <run_dir_a> <run_dir_b> [--alpha FLOAT] [--seed INT]")
        sys.exit(1)

    run_dir_a = sys.argv[1]
    run_dir_b = sys.argv[2]

    alpha = 0.05
    seed = 0
    i = 3
    while i < len(sys.argv):
        if sys.argv[i] == "--alpha" and i + 1 < len(sys.argv):
            alpha = float(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == "--seed" and i + 1 < len(sys.argv):
            seed = int(sys.argv[i + 1])
            i += 2
        else:
            i += 1

    traj_a = _load_trajectories(run_dir_a)
    traj_b = _load_trajectories(run_dir_b)

    map_a = _build_success_map(traj_a)
    map_b = _build_success_map(traj_b)

    # Intersect on common task_ids
    common_ids = sorted(set(map_a) & set(map_b))
    if not common_ids:
        print("No common task_ids found between the two run directories.")
        sys.exit(1)

    success_a = [map_a[tid] for tid in common_ids]
    success_b = [map_b[tid] for tid in common_ids]

    n = len(common_ids)
    rate_a = sum(success_a) / n
    rate_b = sum(success_b) / n

    print(f"\n{'='*60}")
    print(f"Comparing {n} paired tasks")
    print(f"  Run A: {run_dir_a}")
    print(f"  Run B: {run_dir_b}")
    print(f"  Success rate A: {rate_a:.4f}  ({sum(success_a)}/{n})")
    print(f"  Success rate B: {rate_b:.4f}  ({sum(success_b)}/{n})")
    print(f"  Delta (B - A):  {rate_b - rate_a:+.4f}")
    print(f"{'='*60}")

    # McNemar
    p_mcnemar, agreement = mcnemar_exact(success_a, success_b)
    print(f"\nMcNemar's Exact Test (two-sided)")
    print(f"  n00={agreement['n00']}  n01={agreement['n01']}  n10={agreement['n10']}  n11={agreement['n11']}")
    print(f"  Discordant pairs: b={agreement['n01']}, c={agreement['n10']}")
    print(f"  p-value: {p_mcnemar:.6f}  {'<-- significant' if p_mcnemar < alpha else ''}")

    # Paired bootstrap
    boot = paired_bootstrap(success_a, success_b, seed=seed)
    print(f"\nPaired Bootstrap (n_resamples={boot['n_resamples']}, seed={seed})")
    print(f"  Observed delta: {boot['observed_delta']:+.4f}")
    print(f"  Bootstrap mean: {boot['mean_bootstrap_delta']:+.4f}")
    print(f"  95% CI:         [{boot['ci_low']:+.4f}, {boot['ci_high']:+.4f}]")
    print(f"  p-value:        {boot['p_value']:.6f}  {'<-- significant' if boot['p_value'] < alpha else ''}")

    # Holm-Bonferroni on both p-values
    p_list = [p_mcnemar, boot["p_value"]]
    labels = ["McNemar", "Bootstrap"]
    rejected = holm_bonferroni(p_list, alpha=alpha)
    print(f"\nHolm-Bonferroni correction (alpha={alpha})")
    for lbl, p, rej in zip(labels, p_list, rejected):
        status = "REJECT H0" if rej else "fail to reject"
        print(f"  {lbl:12s}: p={p:.6f}  {status}")

    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
