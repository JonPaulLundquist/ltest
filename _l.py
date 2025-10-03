#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Jon Paul Lundquist
"""
Created on Mon Sep 29 20:42:54 2025

Unused pieces of cramervonmises_2samp that may be of interest. Such as calculting
CvM p-value evaluated at the L-test minimum (U_min) without Monte Carlo
resampling (i.e., CvM reporting under the minimized statistic).

Also, includes bad ways of calculating L-test p-value

@author: jplundquist
"""

import numpy as np
from math import comb
from scipy.special import kv, gammaln
from ltest import l_stats
from concurrent.futures import ProcessPoolExecutor, as_completed
from os import cpu_count

#Copied private function directly from scipy.stats
def _pval_cvm_2samp_exact(s, m, n):
    """
    Compute the exact p-value of the Cramer-von Mises two-sample test
    for a given value s of the test statistic.
    m and n are the sizes of the samples.

    [1] Y. Xiao, A. Gordon, and A. Yakovlev, "A C++ Program for
        the Cramér-Von Mises Two-Sample Test", J. Stat. Soft.,
        vol. 17, no. 8, pp. 1-15, Dec. 2006.
    [2] T. W. Anderson "On the Distribution of the Two-Sample Cramer-von Mises
        Criterion," The Annals of Mathematical Statistics, Ann. Math. Statist.
        33(3), 1148-1159, (September, 1962)
    """

    # [1, p. 3]
    lcm = np.lcm(m, n)
    # [1, p. 4], below eq. 3
    a = lcm // m
    b = lcm // n
    # Combine Eq. 9 in [2] with Eq. 2 in [1] and solve for $\zeta$
    # Hint: `s` is $U$ in [2], and $T_2$ in [1] is $T$ in [2]
    mn = m * n
    zeta = lcm ** 2 * (m + n) * (6 * s - mn * (4 * mn - 1)) // (6 * mn ** 2)

    # bound maximum value that may appear in `gs` (remember both rows!)
    zeta_bound = lcm**2 * (m + n)  # bound elements in row 1
    combinations = comb(m + n, m)  # sum of row 2
    max_gs = max(zeta_bound, combinations)
    dtype = np.min_scalar_type(max_gs)

    # the frequency table of $g_{u, v}^+$ defined in [1, p. 6]
    gs = ([np.array([[0], [1]], dtype=dtype)]
          + [np.empty((2, 0), dtype=dtype) for _ in range(m)])
    for u in range(n + 1):
        next_gs = []
        tmp = np.empty((2, 0), dtype=dtype)
        for v, g in enumerate(gs):
            # Calculate g recursively with eq. 11 in [1]. Even though it
            # doesn't look like it, this also does 12/13 (all of Algorithm 1).
            vi, i0, i1 = np.intersect1d(tmp[0], g[0], return_indices=True)
            tmp = np.concatenate([
                np.stack([vi, tmp[1, i0] + g[1, i1]]),
                np.delete(tmp, i0, 1),
                np.delete(g, i1, 1)
            ], 1)
            res = (a * v - b * u) ** 2
            tmp[0] += res.astype(dtype)
            next_gs.append(tmp)
        gs = next_gs
    value, freq = gs[m]
    return np.float64(np.sum(freq[value >= zeta]) / combinations)

#Copied private function directly from scipy.stats
def _cdf_cvm_inf(x):
    """
    Calculate the cdf of the Cramér-von Mises statistic (infinite sample size).

    See equation 1.2 in Csörgő, S. and Faraway, J. (1996).

    Implementation based on MAPLE code of Julian Faraway and R code of the
    function pCvM in the package goftest (v1.1.1), permission granted
    by Adrian Baddeley. Main difference in the implementation: the code
    here keeps adding terms of the series until the terms are small enough.

    The function is not expected to be accurate for large values of x, say
    x > 4, when the cdf is very close to 1.
    """
    x = np.asarray(x)

    def term(x, k):
        # this expression can be found in [2], second line of (1.3)
        u = np.exp(gammaln(k + 0.5) - gammaln(k+1)) / (np.pi**1.5 * np.sqrt(x))
        y = 4*k + 1
        q = y**2 / (16*x)
        b = kv(0.25, q)
        return u * np.sqrt(y) * np.exp(-q) * b

    tot = np.zeros_like(x, dtype='float')
    cond = np.ones_like(x, dtype='bool')
    k = 0
    while np.any(cond):
        z = term(x[cond], k)
        tot[cond] = tot[cond] + z
        cond[cond] = np.abs(z) >= 1e-7
        k += 1

    return tot

#taken from scipy.stats cramervonmises_2samp
def cvm_pval(u, k, N, t, nx, ny, cvm_method):
    
    if cvm_method == 'exact':
        p = _pval_cvm_2samp_exact(u, nx, ny)

    else:
        # compute expected value and variance of T (eq. 11 and 14 in [2])
        et = (1 + 1/N)/6
        vt = (N+1) * (4*k*N - 3*(nx**2 + ny**2) - 2*k)
        vt = vt / (45 * N**2 * 4 * k)

        # computed the normalized statistic (eq. 15 in [2])
        tn = 1/6 + (t - et) / np.sqrt(45 * vt)

        # approximate distribution of tn with limiting distribution
        # of the one-sample test statistic
        # if tn < 0.003, the _cdf_cvm_inf(tn) < 1.28*1e-18, return 1.0 directly
        if tn < 0.003:
            p = 1.0

        else:
            p = max(0, 1. - _cdf_cvm_inf(tn))
            
    return p

#Bad options for calculating L-test p-value. For comparisons sake
def worker_b(args):
    rng = np.random.default_rng()
    pool, xa, ya, nx, ny, ix, iy, k, N, u_min = args
    xb = np.sort(rng.choice(xa, size=nx, replace=True))
    yb = np.sort(rng.choice(ya, size=ny, replace=True))
    s_b, _ = l_stats(xb, yb, nx, ny, ix, iy, k, N)
       
    idx = rng.permutation(N)
    xb2 = np.sort(pool[idx[:nx]]); yb2 = np.sort(pool[idx[nx:]])
    _, u_b = l_stats(xb2, yb2, nx, ny, ix, iy, k, N)
       
    return s_b, u_b

def worker_c(args):
    rng = np.random.default_rng()
    pool, xa, ya, nx, ny, ix, iy, k, N, u_min = args
    xb = np.sort(rng.choice(xa, size=nx, replace=True))
    yb = np.sort(rng.choice(ya, size=ny, replace=True))
    s_b, _ = l_stats(xb, yb, nx, ny, ix, iy, k, N)
       
    draw = rng.choice(pool, size=N, replace=True)
    xb2 = np.sort(draw[:nx]); yb2 = np.sort(draw[nx:])
    _, u_b = l_stats(xb2, yb2, nx, ny, ix, iy, k, N)        
       
    return s_b, u_b

#Version B and C are alternate versions suggested by ChatGPT-5 which are not good at all
def l_pval_b(xa, ya, nx, ny, ix, iy, k, N, u_min, l_shift, B, tol_rel=0.05, max_workers=None):

    if max_workers==None:
        max_workers = cpu_count() or 1
    
    #Just use data shift and bootstrap/permute pooled distributions for samples
    pool = np.concatenate([xa - l_shift, ya]) 

    count = 0
    s_vals = []
    b = 0
    l_p = 1/(B+1)
        
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        #boostrap pooled distributions
        futures = [ex.submit(worker_b, (pool, xa, ya, nx, ny, ix, iy, k, N, u_min)) for _ in range(B)]
        #permute pooled distributions
        #futures = [ex.submit(worker_c, (pool, xa, ya, nx, ny, ix, iy, k, N, u_min)) for _ in range(B)]
        for fut in as_completed(futures):
            hit, s_b = fut.result()
            count += hit
            b = b + 1
            s_vals.append(s_b)

            l_p = (count + 1.0) / (b + 1.0)
            l_p_err = np.sqrt(b*l_p*(1-l_p))/(b+1)
            err_per = l_p_err/l_p if count > 2 else float("inf")
 
            if err_per < tol_rel:
                # cancel any not-yet-started tasks
                for f in futures:
                    f.cancel()
                # optional: ex.shutdown(cancel_futures=True)  # Python 3.9+
                break
                
    shift_boot = float(np.mean(s_vals)) if s_vals else 0.0
    shift_err  = float(np.std(s_vals, ddof=1)) if len(s_vals) > 1 else 0.0
       
    return l_p, l_p_err, shift_boot, shift_err

def l_pval_c(xa, ya, nx, ny, ix, iy, k, N, u_min, l_shift, B, tol_rel=0.05, max_workers=None):

    if max_workers==None:
        max_workers = cpu_count() or 1

    #Just use data shift and bootstrap/permute pooled distributions for samples
    pool = np.concatenate([xa - l_shift, ya])

    count = 0
    s_vals = []
    b = 0
    l_p = 1/(B+1)
        
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(worker_c, (pool, xa, ya, nx, ny, ix, iy, k, N, u_min)) for _ in range(B)]
        for fut in as_completed(futures):
            hit, s_b = fut.result()
            count += hit
            b = b + 1
            s_vals.append(s_b)

            l_p = (count + 1.0) / (b + 1.0)
            l_p_err = np.sqrt(b*l_p*(1-l_p))/(b+1)
            err_per = l_p_err/l_p if count > 2 else float("inf")
 
            if err_per < tol_rel:
                # cancel any not-yet-started tasks
                for f in futures:
                    f.cancel()
                # optional: ex.shutdown(cancel_futures=True)  # Python 3.9+
                break
                
    shift_boot = float(np.mean(s_vals)) if s_vals else 0.0
    shift_err  = float(np.std(s_vals, ddof=1)) if len(s_vals) > 1 else 0.0
    
    return l_p, l_p_err, shift_boot, shift_err
