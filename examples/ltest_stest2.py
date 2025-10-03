#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Jon Paul Lundquist
"""
Created on Mon Sep 29 22:14:22 2025

A very rough draft...
ChatGPT-5 Generated Shift accuracy evaluation for L-test's shift estimator.

Goal
----
When the L-test p-value is large (fail to reject), the two samples are consistent
with "same shape up to a shift", and the L-test's argmin shift (l_shift) is
interpretable as a robust location offset. This script quantifies that accuracy.

What this file does
-------------------
1) Generates SAME-SHAPE+SHIFT data (optionally with uniform contamination/jitter).
2) Calls your ltest(x, y, ...) and compares:
      - l_shift (and shift_boot) vs true shift Δ
      - mean/median differences vs Δ (baselines)
3) Reports Bias / MAE / RMSE and (if available) CI coverage.

It can also run a SHAPE-MISMATCH scenario (e.g., normal vs skew mixture) for
contrast, showing why l_shift is *not* a location estimator when shapes differ.

Requirements
-----------
- numpy
- your ltest importable as: from ltest import ltest
  ltest must return: (l_p, l_stat, l_shift, shift_err, shift_boot, cvm_p, t)
  
@author: Jon Paul Lundquist
"""

import math
import numpy as np

# ------------------ Import your L-test (adjust path if needed) ------------------
from ltest import ltest   # <- ensure this import works


# -------------------- Optional: Wilson/Clopper CI for coverage ------------------
try:
    from scipy.stats import beta as _beta_dist
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False

def prop_ci(successes, n, alpha=0.05, method="wilson"):
    """95% CI for a binomial proportion (coverage)."""
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    if method == "clopper" and _HAVE_SCIPY:
        a = alpha / 2.0
        lo = 0.0 if successes == 0 else _beta_dist.ppf(a, successes, n - successes + 1)
        hi = 1.0 if successes == n else _beta_dist.ppf(1 - a, successes + 1, n - successes)
        return (lo, hi)
    # Wilson score interval (SciPy-free)
    z = 1.959963984540054
    denom = 1.0 + (z*z)/n
    center = (p + (z*z)/(2*n)) / denom
    half = (z/denom) * math.sqrt(p*(1-p)/n + (z*z)/(4*n*n))
    return (max(0.0, center - half), min(1.0, center + half))


# --------------------------- Family samplers (parametric) -----------------------
def sample_family(rng, n, family, mu, sigma):
    """Draw n samples from a location-scale family (no post-editing)."""
    if family == "normal":
        return rng.normal(mu, sigma, size=n)
    elif family == "laplace":
        b = sigma / math.sqrt(2.0)
        return rng.laplace(mu, b, size=n)
    elif family == "t3":
        df = 3.0
        s  = sigma * math.sqrt((df - 2.0) / df)
        return mu + s * rng.standard_t(df, size=n)
    elif family == "mix2":
        d = 1.25
        centers = mu + rng.choice((-1.0, 1.0), size=n) * (d * sigma)
        return rng.normal(centers, sigma, size=n)
    elif family == "skew2":
        # two-sided normal with different scales (right-skew if k>1)
        k = 1.8
        p_right = k / (1.0 + k)
        c = math.sqrt(2.0 / (1.0 + k*k))
        sigma_L = sigma * c
        sigma_R = sigma * c * k
        u = rng.random(n)
        left = u >= p_right
        x = np.empty(n)
        x[~left] = rng.normal(mu, sigma_R, size=(~left).sum())
        x[left]  = rng.normal(mu, sigma_L, size=left.sum())
        return x
    else:
        raise ValueError(f"Unknown family '{family}'")


# ---------- SAME-SHAPE + SHIFT generator (with optional contamination/jitter) ---
def draw_pair_same_shape_shift(
    rng, n, *,
    family="normal",
    delta_dist="uniform", delta_width=2.0, delta_sd=1.0,   # true shift Δ law
    mu_sd=2.0,
    sigma_dist="lognormal", logsig_sd=0.35,
    same_scale=True,                      # enforce σ_y = σ_x for identical shapes
    contam_eps=0.0, contam_width=0.0,     # per-point replace with U[-w,w] with prob eps
    add_uniform_jitter=0.0                # always add U[-j, j] noise; j=0 disables
):
    """
    Returns (x, y, delta_true) where parents share the same shape except for
    a location shift Δ. Contamination/jitter use the SAME law on both samples
    (independent draws) to preserve shape equivalence.
    """
    mu_common = rng.normal(0.0, mu_sd)
    delta_true = (rng.uniform(-delta_width, +delta_width)
                  if delta_dist == "uniform" else
                  rng.normal(0.0, delta_sd))

    if sigma_dist == "lognormal":
        sigma_x = float(np.exp(rng.normal(0.0, logsig_sd)))
        sigma_y = sigma_x if same_scale else float(np.exp(rng.normal(0.0, logsig_sd)))
    elif sigma_dist == "fixed":
        sigma_x = sigma_y = 1.0 if same_scale else (1.0, 1.2)[0]  # simple alternative
    else:
        raise ValueError("sigma_dist must be 'lognormal' or 'fixed'")

    x = sample_family(rng, n, family, mu=mu_common,            sigma=sigma_x)
    y = sample_family(rng, n, family, mu=mu_common + delta_true, sigma=sigma_y)

    # optional contamination (same law, independent draws)
    if contam_eps > 0.0 and contam_width > 0.0:
        mX = rng.random(n) < contam_eps
        mY = rng.random(n) < contam_eps
        x[mX] = rng.uniform(-contam_width, contam_width, size=mX.sum())
        y[mY] = rng.uniform(-contam_width, contam_width, size=mY.sum())

    # optional small uniform jitter (same law)
    if add_uniform_jitter > 0.0:
        j = add_uniform_jitter
        x = x + rng.uniform(-j, j, size=n)
        y = y + rng.uniform(-j, j, size=n)

    return x, y, float(delta_true)


# --------------------------- SHAPE-MISMATCH generator ---------------------------
def draw_pair_with_true_shift_mismatch(
    rng, n, *,
    families=("normal","skew2"),   # (fam_x, fam_y)
    mu_sd=2.0, logsig_sd=0.35,
    delta_dist="uniform", delta_width=2.0, delta_sd=1.0
):
    """For contrast: different shapes + shift."""
    mu_common = rng.normal(0.0, mu_sd)
    delta_true = (rng.uniform(-delta_width, +delta_width)
                  if delta_dist == "uniform" else
                  rng.normal(0.0, delta_sd))
    sigma_x = float(np.exp(rng.normal(0.0, logsig_sd)))
    sigma_y = float(np.exp(rng.normal(0.0, logsig_sd)))
    fx, fy = families
    x = sample_family(rng, n, fx, mu=mu_common,            sigma=sigma_x)
    y = sample_family(rng, n, fy, mu=mu_common + delta_true, sigma=sigma_y)
    return x, y, float(delta_true)


# ------------------------- shift_boot parser (flexible) -------------------------
def parse_shift_boot(ret):
    """
    Accepts:
      - float/int -> (est, None, None)
      - (est, lo, hi) or (lo, hi) -> returns (est, lo, hi) with est inferred if needed
      - dict {'est':..., 'lo':..., 'hi':...}
      - long 1D array-like of replicates -> (median, None, None)
    """
    if isinstance(ret, (float, int)):
        return float(ret), None, None

    if isinstance(ret, dict):
        est = ret.get("est"); lo = ret.get("lo"); hi = ret.get("hi")
        est = float(est) if est is not None else est
        lo  = float(lo)  if lo  is not None else lo
        hi  = float(hi)  if hi  is not None else hi
        if est is None and (lo is not None and hi is not None):
            est = 0.5*(lo + hi)
        if est is None:
            raise TypeError("shift_boot dict requires 'est' or ('lo','hi').")
        if lo is not None and hi is not None and lo > hi:
            lo, hi = hi, lo
        return est, lo, hi

    if isinstance(ret, (tuple, list)):
        L = len(ret)
        if L == 2:
            lo, hi = float(ret[0]), float(ret[1])
            if lo > hi: lo, hi = hi, lo
            return 0.5*(lo+hi), lo, hi
        if L >= 3:
            est, lo, hi = float(ret[0]), float(ret[1]), float(ret[2])
            if lo > hi: lo, hi = hi, lo
            return est, lo, hi
        if L == 1:
            return float(ret[0]), None, None
        # could be long array-like
        try:
            arr = np.asarray(ret, dtype=float)
            if arr.ndim == 1 and arr.size >= 10:
                return float(np.median(arr)), None, None
        except Exception:
            pass

    raise TypeError("Unrecognized 'shift_boot' format.")


# ----------------------- Accuracy runner (single configuration) -----------------
def evaluate_shift_accuracy_with_ltest(
    n=200, trials=1000, seed=0,
    *,
    # choose generator
    pure_shift=True,
    family="normal",                       # used when pure_shift=True
    families_mismatch=("normal","skew2"),  # used when pure_shift=False
    # shift/scale randomness
    mu_sd=2.0, logsig_sd=0.35,
    delta_dist="uniform", delta_width=2.0, delta_sd=1.0,
    same_scale=True,
    contam_eps=0.0, contam_width=0.0, add_uniform_jitter=0.0,
    # L-test control
    ltest_B=500, ltest_tol_p=0.05,
    # coverage CI
    ci_method="wilson"
):
    rng = np.random.default_rng(seed)
    sums = {
        "l_shift"    : dict(sq=0.0, abs=0.0, bias=0.0, cov_hits=0, cov_total=0, ciw_sum=0.0),
        "shift_boot" : dict(sq=0.0, abs=0.0, bias=0.0, cov_hits=0, cov_total=0, ciw_sum=0.0),
        "mean_diff"  : dict(sq=0.0, abs=0.0, bias=0.0),
        "median_diff": dict(sq=0.0, abs=0.0, bias=0.0),
    }

    for _ in range(trials):
        if pure_shift:
            x, y, delta_true = draw_pair_same_shape_shift(
                rng, n,
                family=family,
                delta_dist=delta_dist, delta_width=delta_width, delta_sd=delta_sd,
                mu_sd=mu_sd,
                sigma_dist="lognormal", logsig_sd=logsig_sd,
                same_scale=same_scale,
                contam_eps=contam_eps, contam_width=contam_width,
                add_uniform_jitter=add_uniform_jitter
            )
        else:
            x, y, delta_true = draw_pair_with_true_shift_mismatch(
                rng, n,
                families=families_mismatch,
                mu_sd=mu_sd, logsig_sd=logsig_sd,
                delta_dist=delta_dist, delta_width=delta_width, delta_sd=delta_sd
            )

        # --- Call your L-test (correct unpacking) ---
        
        l_p, l_p_err, l_shift_est, shift_boot_ret, shift_err, l_stat = ltest(
            x, y, B=ltest_B, tol_p=ltest_tol_p
        )

        # l_shift errors and Wald CI coverage if shift_err is numeric
        err = float(l_shift_est) - delta_true
        S = sums["l_shift"]
        S["sq"]   += err*err
        S["abs"]  += abs(err)
        S["bias"] += err
        if isinstance(shift_err, (float, int)):
            se = float(shift_err)
            lo = l_shift_est - 1.959963984540054 * se
            hi = l_shift_est + 1.959963984540054 * se
            S["cov_total"] += 1
            if lo <= delta_true <= hi:
                S["cov_hits"] += 1
            S["ciw_sum"] += (hi - lo)

        # shift_boot errors + coverage if CI is provided
        try:
            est_b, lo_b, hi_b = parse_shift_boot(shift_boot_ret)
        except Exception:
            est_b, lo_b, hi_b = float(l_shift_est), None, None
        errb = est_b - delta_true
        SB = sums["shift_boot"]
        SB["sq"]   += errb*errb
        SB["abs"]  += abs(errb)
        SB["bias"] += errb
        if lo_b is not None and hi_b is not None:
            SB["cov_total"] += 1
            if lo_b <= delta_true <= hi_b:
                SB["cov_hits"] += 1
            SB["ciw_sum"] += (hi_b - lo_b)

        # baselines
        md = float(np.mean(y) - np.mean(x))
        sums["mean_diff"]["sq"]   += (md - delta_true)**2
        sums["mean_diff"]["abs"]  += abs(md - delta_true)
        sums["mean_diff"]["bias"] += (md - delta_true)

        med = float(np.median(y) - np.median(x))
        sums["median_diff"]["sq"]   += (med - delta_true)**2
        sums["median_diff"]["abs"]  += abs(med - delta_true)
        sums["median_diff"]["bias"] += (med - delta_true)

    def finalize(s):
        out = {
            "RMSE": math.sqrt(s["sq"]/trials),
            "MAE" : s["abs"]/trials,
            "Bias": s["bias"]/trials,
        }
        if "cov_total" in s and s["cov_total"] > 0:
            cov = s["cov_hits"]/s["cov_total"]
            out.update({
                "CI_coverage": cov,
                "CI_coverage_CI95": prop_ci(s["cov_hits"], s["cov_total"], method=ci_method),
                "avg_CI_width": s["ciw_sum"]/s["cov_total"],
            })
        else:
            if "cov_total" in s:
                out.update({"CI_coverage": None, "CI_coverage_CI95": None, "avg_CI_width": None})
        return out

    results = {k: finalize(v) for k, v in sums.items()}
    meta = {
        "pure_shift": pure_shift,
        "family": family,
        "families_mismatch": families_mismatch,
        "n": n, "trials": trials,
        "mu_sd": mu_sd, "logsig_sd": logsig_sd,
        "delta_dist": delta_dist, "delta_width": delta_width, "delta_sd": delta_sd,
        "same_scale": same_scale,
        "contam_eps": contam_eps, "contam_width": contam_width,
        "add_uniform_jitter": add_uniform_jitter,
        "ltest_B": ltest_B, "ltest_tol_p": ltest_tol_p
    }
    return {"meta": meta, "results": results}


# ----------------------------------- Demo --------------------------------------
if __name__ == "__main__":
    from pprint import pprint

    # 1) SAME-SHAPE + SHIFT (recommended for shift accuracy)
    res1 = evaluate_shift_accuracy_with_ltest(
        n=200, trials=800, seed=123,
        pure_shift=True, family="normal",
        delta_dist="uniform", delta_width=2.0,
        mu_sd=2.0, logsig_sd=0.35,
        same_scale=True,             # identical shapes
        contam_eps=0.00,             # try 0.05 with contam_width=3.0 to stress robustness
        contam_width=0.0,
        add_uniform_jitter=0.0,
        ltest_B=500, ltest_tol_p=0.05
    )
    print("\n=== Accuracy summary (same shape + shift) ===")
    pprint(res1["meta"]); pprint(res1["results"])

    # # 2) SHAPE-MISMATCH (contrast): normal vs skew2 (not a location-estimator use case)
    # res2 = evaluate_shift_accuracy_with_ltest(
    #     n=200, trials=800, seed=456,
    #     pure_shift=False, families_mismatch=("normal","skew2"),
    #     delta_dist="uniform", delta_width=2.0,
    #     mu_sd=2.0, logsig_sd=0.35,
    #     ltest_B=500, ltest_tol_p=0.05
    # )
    # print("\n=== Accuracy summary (shape mismatch: normal vs skew2) ===")
    # pprint(res2["meta"]); pprint(res2["results"])