#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Jon Paul Lundquist
"""
Created on Mon Sep 29 12:46:06 2025

Consider this a rough draft...
ChatGPT-5 generated and not changed. Shows some confusion between Type I and Type II tests

Result of example:
    
shift=0.000  power=0.048  beta=0.953  95% CI[0.031,0.073]
shift=0.050  power=0.045  beta=0.955  95% CI[0.029,0.070]
shift=0.100  power=0.043  beta=0.958  95% CI[0.027,0.067]
shift=0.200  power=0.043  beta=0.958  95% CI[0.027,0.067]
t(3)+shift: power=0.960 beta=0.040 CI=(np.float64(0.9360177522821589), np.float64(0.9752309369302883))

So, both the included Type I test and Type 2 test are great results.
  
@author: Jon Paul Lundquist
"""

import os, numpy as np
from math import sqrt
from ltest import ltest

# ---- 1) Generic binomial CI for power (Wilson) ----
def wilson_ci(p_hat, n, z=1.959963984540054):  # ~95%
    if n == 0:
        return (0.0, 1.0)
    denom = 1 + (z*z)/n
    center = (p_hat + (z*z)/(2*n)) / denom
    half = z * np.sqrt((p_hat*(1-p_hat) + (z*z)/(4*n)) / n) / denom
    return max(0.0, center - half), min(1.0, center + half)

# ---- 2) Power / Type II estimator, test_fn is ANY function returning a p-value ----
def estimate_typeII(test_fn, gen_alt_fn, nx, ny, alpha=0.05, M=500, seed=123):
    """
    test_fn(x, y) -> p_value   # e.g., wraps your ltest and returns its p-value
    gen_alt_fn(nx, ny, rng) -> (x, y)  # draws a dataset under the ALTERNATIVE
    """
    rng = np.random.default_rng(seed)
    rejects = 0
    for _ in range(M):
        x, y = gen_alt_fn(nx, ny, rng)
        p = test_fn(x, y)      # black-box test; includes your full L-test logic
        rejects += (p <= alpha)
    power = rejects / M
    beta  = 1.0 - power
    lo, hi = wilson_ci(power, M)
    return power, beta, (lo, hi)

# ---- 3) Example: generic alternatives WITH a shift (and optional shape change) ----
def make_alt_generator(shift=0.2, shape="same", **kwargs):
    """
    Returns a gen_alt_fn that draws X,Y with a specified shift in Y relative to X.
    shape:
      "same"   -> same family, just shifted (classic location alternative)
      "t"      -> Y has t_nu tails, still shifted by 'shift'
      "bimodal"-> Y is mixture of two Normals, shifted by 'shift'
      "var"    -> Y has different variance, plus shift
    All alternatives *include* a shift in Y by 'shift'.
    """
    nu = kwargs.get("nu", 3)
    d  = kwargs.get("d", 1.5)
    sigma = kwargs.get("sigma", 1.5)

    def gen_alt(nx, ny, rng):
        # Base X from N(0,1)
        x = rng.standard_normal(nx)

        if shape == "same":
            y0 = rng.standard_normal(ny)

        elif shape == "t":
            y0 = rng.standard_t(df=nu, size=ny)
            if nu > 2:
                y0 *= np.sqrt((nu-2)/nu)  # scale to var≈1

        elif shape == "bimodal":
            comp = rng.integers(0, 2, size=ny)
            y0 = rng.standard_normal(ny) + np.where(comp==0, -d, d)

        elif shape == "var":
            y0 = rng.standard_normal(ny) * sigma

        else:
            raise ValueError("unknown shape")

        # Apply the shift difference in the alternative (Y shifted by 'shift')
        y = y0 + shift

        return x, y

    return gen_alt

# ---- 4) Example: adapter to your L-test (returns its L-test p-value) ----
def make_ltest_adapter(B=2000, workers=None):
    def test_fn(x, y):
        # ltest returns: l_p0, l_p, l_stat, l_shift, shift_err, shift_boot, cvm_p, t
        l_p, *_ = ltest(x, y, B=B, workers=workers)
        return l_p
    return test_fn

# ---- 5) Running it (examples) ----
if __name__ == "__main__":
    # choose your inner-boot B large enough that MC noise on p isn't dominating
    test_fn = make_ltest_adapter(B=3000, workers=os.cpu_count())

    #JPL: This actually is just a Type I test over shifts... ChatGPT-5 has difficulty understanding this
    # power should be about equal to alpha for any shift with the L-test -- and it is.
    
    # Power curve vs shift, same-shape alternatives (pure location differences)
    for delta in [0.0, 0.05, 0.1, 0.2]:
        gen_alt = make_alt_generator(shift=delta, shape="same")
        power, beta, (lo, hi) = estimate_typeII(test_fn, gen_alt, nx=200, ny=200, alpha=0.05, M=400)
        print(f"shift={delta:.3f}  power={power:.3f}  beta={beta:.3f}  95% CI[{lo:.3f},{hi:.3f}]")

    #JPL: Below seems to be an actual Type II test... power should be close to 1.
    #L-test result is t(3)+shift: power=0.960 beta=0.040 CI=(np.float64(0.9360177522821589), np.float64(0.9752309369302883))
    
    # Power for a shape+shift alternative (heavier tails + shift)
    gen_alt_t = make_alt_generator(shift=0.1, shape="t", nu=3)
    power, beta, ci = estimate_typeII(test_fn, gen_alt_t, nx=200, ny=200, alpha=0.05, M=400)
    print("t(3)+shift: power=%.3f beta=%.3f CI=%s" % (power, beta, ci))