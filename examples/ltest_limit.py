#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Jon Paul Lundquist
"""
Created on Mon Sep 29 21:55:49 2025

Consider a very rough draft...
Written by ChatGPT in an attempt to start testing if there are limiting 
distributions for the L-test statistic. Not looked at too deeply...

@author: Jon Paul Lundquist
"""

import numpy as np
from math import sqrt
#from scipy.stats import kstest
from ltest import ltest

# ---- Configure ----
SIZES = [(50,50), (100,100), (200,200), (400,400), (800,800)]
M = 500   # Monte Carlo replicates per (size, base)
SEED = 123
B_INNER = 0   # we only need l_stat; set B=0 or small to skip inner resampling if your ltest allows
WORKERS = 1

# ---- Base laws under H0 (same shape up to shift). We simulate X~F, Y~F shifted by s=0. ----
def draw_pair(n, m, kind, rng):
    if kind == "normal":
        x = rng.standard_normal(n); y = rng.standard_normal(m)
    elif kind == "logistic":
        # logistic via inverse CDF
        u = rng.random(n); v = rng.random(m)
        x = np.log(u) - np.log1p(-u)
        y = np.log(v) - np.log1p(-v)
    elif kind == "laplace":
        # laplace via inverse CDF
        u = rng.random(n); v = rng.random(m)
        x = np.where(u < 0.5,  np.log(2*u),  -np.log(2*(1-u)))
        y = np.where(v < 0.5,  np.log(2*v),  -np.log(2*(1-v)))
    elif kind == "t5":
        x = rng.standard_t(df=5, size=n) * sqrt((5-2)/5)  # var≈1
        y = rng.standard_t(df=5, size=m) * sqrt((5-2)/5)
    else:
        raise ValueError("unknown kind")
    # H0 allows any common shift; we keep shift=0 to avoid injecting finite-sample asymmetry
    return x, y

BASES = ["normal", "logistic", "laplace", "t5"]

# ---- Utilities ----
def l_stat_only(x, y):
    # ltest returns: l_p0, l_p, l_stat, l_shift, shift_err, shift_boot
    _, _, _, _, _, l_stat = ltest(x, y, B=B_INNER, workers=WORKERS)
    return float(l_stat)

def ks_distance(sample_a, sample_b):
    # Two-sample KS distance (on the l_stat draws)
    # Convert to ECDF distance by using scipy.stats.kstest on a merged trick (or write your own).
    # Simpler: brute-force ECDF difference
    a = np.sort(sample_a); b = np.sort(sample_b)
    # Evaluate ECDFs at pooled points
    grid = np.sort(np.concatenate([a, b]))
    Fa = np.searchsorted(a, grid, side="right") / a.size
    Fb = np.searchsorted(b, grid, side="right") / b.size
    return float(np.max(np.abs(Fa - Fb)))

def qq_error(sample, ref, q=np.linspace(0.01, 0.99, 99)):
    s = np.quantile(sample, q)
    r = np.quantile(ref, q)
    # measure departure from y=x
    return float(np.max(np.abs(s - r)))

# ---- Experiment ----
def run_limiting_check():
    rng = np.random.default_rng(SEED)
    results = {}  # (n,m,base) -> l_stat array

    for (n, m) in SIZES:
        for base in BASES:
            vals = np.empty(M, dtype=float)
            for i in range(M):
                x, y = draw_pair(n, m, base, rng)
                vals[i] = l_stat_only(x, y)
            results[(n, m, base)] = vals

    # Compare across bases at each size (distribution-free check)
    print("=== KS distances across bases at each size (smaller is more similar) ===")
    for (n, m) in SIZES:
        ref = results[(n, m, BASES[0])]
        for base in BASES[1:]:
            d = ks_distance(ref, results[(n, m, base)])
            print(f"n={n}, m={m}: KS({BASES[0]} vs {base}) = {d:.4f}")

    # Compare sizes for one base (stability as n increases)
    base = "normal"
    print("\n=== QQ max-abs error vs reference size for base='normal' ===")
    ref = results[SIZES[-1] + (base,)]  # largest size as reference
    for (n, m) in SIZES[:-1]:
        e = qq_error(results[(n, m, base)], ref)
        print(f"n={n}, m={m}: QQ max|Δ| = {e:.5f}")

    # Moment diagnostics (do mean/var stabilize with size?)
    print("\n=== Moment diagnostics under H0 ===")
    for (n, m) in SIZES:
        row = []
        for base in BASES:
            v = results[(n, m, base)]
            row.append((np.mean(v), np.var(v, ddof=1)))
        means = ", ".join(f"{mu:.5f}" for (mu, _) in row)
        vars_ = ", ".join(f"{sg2:.5f}" for (_, sg2) in row)
        print(f"n={n}, m={m}: mean[{means}]  var[{vars_}]")

if __name__ == "__main__":
    run_limiting_check()