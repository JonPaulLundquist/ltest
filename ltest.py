#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Jon Paul Lundquist
"""
Created on Wed Sep 24 16:34:33 2025

    Two-sample L-test (shift-invariant Cramér–von Mises variant).
    
    Performs the two-sample L-test. The L-test is a shift-invariant modification 
    of the Cramér–von Mises (CvM) two-sample test [1] that minimizes the integral 
    squared difference between two empirical CDFs (L_2 squared distance), called U, 
    by optimizing a scalar location shift, s, between samples. For independent 
    samples X = {X_i}_{i=1..n} and Y = {Y_j}_{j=1..m}, the null hypothesis is:
    
    H0: ∃ s ∈ ℝ such that F_X(t) = F_Y(t + s) for all t
    
    i.e., samples X and Y are draws from the same (unspecified) continuous 
    distribution up to a location difference.
    
    Motivation
    ----------
    Originally developed for ultra–high-energy cosmic-ray (UHECR) composition
    work comparing X_max distributions to model predictions, where model means
    have larger uncertainty than higher moments (such that there is significant 
    CI overlap) [2]. The L-test is useful whenever relative location may be biased 
    (instrument/location/seasonal effects, unknown inter-experiment offsets), 
    and/or shape differences (variance, skew, tails, multimodality) are of primary 
    interest. See Appendix B (“L-test”) of the author’s Ph.D. thesis for background 
    and derivations [3].

    Parameters
    ----------
    x : array_like, shape (n,)
        Sample 1 (must have n ≥ 2).
    y : array_like, shape (m,)
        Sample 2 (must have m ≥ 2).
    B : int, optional (default=5000)
        Maximum number of bootstrap/permutation replicates used to estimate the
        L-test p-value. Early stopping may terminate before B if the relative
        Monte Carlo uncertainty target is met.
    tol_p : float, optional (default=0.05)
        Target relative standard error on the Monte Carlo p-value; when the
        running binomial SE / p ≤ tol_p and at least a few hits are observed,
        replication stops early.
    tol_s : float, optional (default=0.05)
        Target relative standard error on the Monte Carlo shift standard deviation 
        (uncertainty); when the running Welford and Pébay [4] SE_SD / p ≤ tol_s.
    workers : int or None, optional (default=None)
        Number of parallel workers used for the inner resampling loop. If None,
        uses the number of logical CPUs reported by the OS.
    brute : True or False, optional (default=False)
        A more brute force search for shift s that minimizes U. Can be very very
        slow as grows proportional to n*m.
    stat : 'cvm' or 'ad', optional (default='cvm')
        Uses Anderson-Darling reweighting of L_2 distance that assigns more 
        importance to distribution tails [5]. Not well tested.
   
    Returns
    -------
    l_p : float
        L-test p-value estimated by resampling (bootstrap + permutation). Uses
        add-one smoothing: (count + 1) / (B_eff + 1), where B_eff is the number
        of completed replicates.
    l_p_err : Uncertainty on p-values calculated by binomial error 
        (np.sqrt(b*l_p*(1-l_p))/(b+1)) where b is the number of replicates 
        tested before stopping condition satisfied.
    l_shift : float
        Estimated location shift (in the units of the data) for sample 1 (x-l_shift), 
        that minimizes the CvM U functional between the empirical CDFs.
    shift_boot : float
        Mean of the bootstrap shifts across replicates. Can be a more accurate
        estimator.
    shift_err : float
        Standard error of l_shift estimated from the bootstrap replicates.
    l_stat : float
        L-test T statistic derived from U_min accounting for sample sizes.
    
    Prints
    ------
    Number of completed bootstrap replicates. Shows whether the stopping condition
    was B or the tolerances (tol_p, tol_s).
        
    Notes
    -----
    * The L statistic equals the minimized version of Eq. (9) in Anderson [6] when
      including a free location parameter. Minimization is performed numerically
      (scalar search).
    * The L-shift is generally interpretable as a location offset only when the 
      L-test fails to reject (i.e., shapes appear compatible) or the two parent 
      distributions are symmetrical. Under shape mismatch, it is a nuisance 
      alignment chosen to minimize the ECDF distance and should not be interpreted 
      as a population location difference.
    * Can be a robust/efficient distribution location estimator of sample 1 (x) by
      using a sharply peaked distribution such as y = rng.triangular(-t, 0, t, size=500*y.size) 
      where t = 0.25*np.median(np.abs(x - np.median(x))) (MAD(x)/4) for example. 
      y = np.zeros(x.size) (delta function) is exactly l_shift = np.median(x).
    * Monte Carlo p-value uncertainty: by default we report the smoothed estimator
      (count+1)/(B_eff+1) with an early-stop rule keyed to the running binomial SE.
      If very few exceedances occur, expect wider relative error unless B is large.
   
    This is new code based upon the statistical distribution test proposed in 
    the author's 2017 Ph.D. thesis [3] and was used for the conference paper 
    'Study of UHECR Composition Using Telescope Array’s Middle Drum Detector and 
    Surface Array in Hybrid Mode', 34th ICRC, 2016. Additionally, paper 
    The Astrophysical Journal, 858:76 (27pp), 2018 May 10 used a similar principle.
   
    See Also
    --------
    scipy.stats.cramervonmises_2samp, scipy.stats.anderson_ksamp
    
    Dependencies
    ------------
    - Python ≥ 3.8
    - NumPy ≥ 1.23
    - SciPy ≥ 1.9  (for `scipy.optimize.minimize_scalar` / `minimize`)

    Standard library: `os`, `concurrent.futures`, `math`.

    Installation
    ------------
    pip install numpy scipy
    # or: conda install numpy scipy
    pip install ltest-shift
    
    Multiprocessing note
    --------------------
    When calling this from a script on Windows/macOS (spawn start method),
    wrap the entry point:
        if __name__ == "__main__":
            # your code that calls ltest(...)
        
   References
   ----------
   .. [1] https://en.wikipedia.org/wiki/Cramer-von_Mises_criterion
   .. [2] Abbasi, R. U., Thomson, G. B., <xmax> uncertainty from extrapolation 
          of cosmic ray air shower parameters arXiv:1605.05241
   .. [3] Lundquist, J.P. Energy Anisotropies of Proton-Like Ultra-High 
          Energy Cosmic Rays, Ph.D. Thesis, University of Utah, 2017
   .. [4] Pebay, P.P. (2008) Formulas for Robust, One-Pass Parallel 
          Computation of Covariances and Arbitrary-Order Statistical Moments, 
          Sandia National Laboratories Technical Report
   .. [5] Scholz, F. W and Stephens, M. A. (1987), K-Sample Anderson-Darling 
          Tests, Journal of the American Statistical Association, Vol. 82, pp. 918-924
   .. [6] Anderson, T.W., On the distribution of the two-sampleCramer-von-Mises 
          criterionm, The Annals of Mathematical Statistics, pp. 1148-1159
   
@author: Jon Paul Lundquist
"""

import numpy as np
import math
#from scipy.stats import rankdata, cramervonmises_2samp
from scipy.optimize import minimize_scalar, minimize
from concurrent.futures import ProcessPoolExecutor, as_completed
from os import cpu_count
#from _l import *

#Minimizers often give the value when the algorithm stopped not the lowest value seen.
class BestKeeper:
    def __init__(self, f):
        self.f = f
        self.best_s = None
        self.best_u = np.inf
    def __call__(self, s):
        u = self.f(s)
        if u < self.best_u:
            self.best_u, self.best_s = u, float(np.asarray(s).ravel()[0])
        return u

#Returns same ranks as scipy rankdata(x,method='average') but is often slightly faster
#Particularly for smaller samples
def rank_data(a, n):
    """
    Compute 1‐based ranks with “average” tie handling,
    purely in NumPy (Numba‐friendly).
    """

    # get sort order
    sorter = np.argsort(a)
    ranks = np.empty(n, dtype=np.float64)

    i = 0
    while i < n:
        # find end of block of ties
        j = i + 1
        while j < n and a[sorter[j]] == a[sorter[i]]:
            j += 1
        # positions i..j-1 inclusive all tied
        # their 1-based ranks would be i+1, i+2, …, j
        # average = ((i+1)+(j))/2 = 0.5*(i+j+1)
        avg_rank = 0.5 * (i + j + 1)
        #for k in range(i, j):
        #    ranks[sorter[k]] = avg_rank
        ranks[sorter[i:j]] = avg_rank
        i = j

    return ranks

# ---- Welford's Online updates up to 4th central moment ---- Pébay (2008) [4]
def update_s_stats(s_b, b, mean, M2, M3, M4):
    """
    One-pass update of running mean and central moments (M2,M3,M4)
    given a new sample value s_b.

    Inputs:
      b    : current count (>=0)
      mean : current running mean
      M2   : sum of squared deviations
      M3   : sum of cubed  deviations
      M4   : sum of quartic deviations
    Returns:
      n_new, mean_new, M2_new, M3_new, M4_new
    """
    n1 = b + 1
    delta = s_b - mean
    delta_b = delta / n1
    delta_b2 = delta_b * delta_b
    term1 = delta * delta_b * b

    M4_new = M4 + (term1 * delta_b2 * (b*b - 3*b + 3)
                   + 6.0 * delta_b2 * M2
                   - 4.0 * delta_b * M3)
    M3_new = M3 + (term1 * delta_b * (b - 2.0)
                   - 3.0 * delta_b * M2)
    M2_new = M2 + term1
    mean_new = mean + delta_b
    return n1, mean_new, M2_new, M3_new, M4_new


def cur_s_std(b, M2, M4):
    """
    From running moments, return:
      sd     : sample standard deviation of the stream
      se_sd  : delta-method SE of the SD
      rse_sd : relative SE = se_sd / sd

    Notes:
      sd uses unbiased sample variance s^2 = M2/(b-1).
      Var(s^2) = [ m4 - ((b-3)/(b-1)) * s^4 ] / b  with m4 = M4/b
      se_sd ≈ sqrt(Var(s^2)) / (2*sd)
    """
    if b < 2:
        return math.inf, math.inf, math.inf

    s2 = M2 / (b - 1)
    sd = math.sqrt(s2) if s2 > 0.0 else 0.0 #Current standard deviation

    if b < 4 or s2 <= 0.0:
        return sd, math.inf, math.inf

    m4 = M4 / b
    var_s2 = (m4 - ((b - 3.0) / (b - 1.0)) * (s2 * s2)) / b
    var_s2 = max(var_s2, 0.0)  # numeric guard
    se_sd = math.sqrt(var_s2) / (2.0 * sd) if sd > 0.0 else 0.0 #Current error on the standard deviation
    rse_sd = (se_sd / sd) if sd > 0.0 else math.inf #The important one: relative SE = se_sd / sd
    return sd, se_sd, rse_sd

def ad2(x, y, nx, ny):
    """
    Raw two-sample Anderson–Darling statistic A^2_{2N} (Scholz–Stephens, unadjusted).
    Matches the textbook definition; *not* the standardized/adjusted value SciPy prints.
    """
    N = nx + ny
    k = nx*ny
    
    z = np.concatenate([x, y])

    # One stable argsort; in the sorted pooled array, entries coming from x are
    # exactly those whose original index < nx.
    order = np.argsort(z, kind="mergesort")
    is_x_sorted = (order < nx)

    # cumulative # of x among first i pooled obs, i = 1..N-1
    A = np.cumsum(is_x_sorted, dtype=np.int64)[:-1]
    i = np.arange(1, N, dtype=float)

    F = A / float(nx)
    G = (i - A) / float(ny)
    H = i / float(N)

    D = F - G
    w = 1.0 / (H * (1.0 - H))           # i in [1..N-1] => safe
    A2_raw = (k / (N * N)) * np.sum(D * D * w)
    return float(A2_raw)

def cvm_u(xa, ya, nx, ny, ix, iy):
    #This is what needs to be minimized for the L-test
    
    #Ranks of x and y in the pooled sample
    z = np.concatenate([xa, ya])
    #print(z)
    # in case of ties, use midrank (see [1])
    r = rank_data(z, nx+ny)     #r = rankdata(z, method='average')

    rx = r[:nx] #ranks of x
    ry = r[nx:] #ranks of y
    
    # compute U (eq. 10 in [2])
    u = nx * np.sum((rx - ix)**2) + ny * np.sum((ry - iy)**2)

    return u

#Normalize U statistic to the CvM criterion
def cvm_stat(u, k, N):
    return float(u / (k*N) - (4*k - 1)/(6*N))

def mode(x):
    vals,counts = np.unique(x, return_counts=True)
    index = np.argmax(counts)
    
    return vals[index]

def l_u(xs, ya, nx, ny, ix, iy, stat):
    """
    Minimize CvM U value Equation 10 in [6] after initial guess for shift is used
    Minimize f(s) = cvm_u(xs - s, ya, nx, ny, ix, iy) over s.
    Returns (u_min, s_opt).
    """
    #def f(s):
    #    return cvm_u(xs - s, ya, nx, ny, ix, iy)
    
    def f(s):
        if stat == 'cvm':
            return cvm_u(xs-s, ya, nx, ny, ix, iy)
        elif stat == 'ad':
            return ad2(xs-s, ya, nx, ny)
        else:
            raise ValueError('Invalid Test Statistic')
    
    wrapped = BestKeeper(f)
    
    # pick a reasonable search half-width
    sd = np.sqrt(np.var(xs, ddof=1) + np.var(ya, ddof=1))
    if not np.isfinite(sd) or sd == 0:
        sd = 1.0
    
    width = sd/2
    
    #Let's overkill...they're all pretty fast. None of them are the best 100% of the time. 
    #You can delete probably all but 1 or 2 or 5 or 6 if you need more speed with ~0.1% different U.
    results = []
    results.append(minimize_scalar(wrapped, method='golden', options={'maxiter': 10_000}))
    results.append(minimize_scalar(wrapped, bounds=(-width, width), method='bounded', 
                                   options={'maxiter': 10_000}))
    results.append(minimize_scalar(wrapped, bounds=(float(np.min(xs)), float(np.median(xs))), 
                                   method='bounded', options={'maxiter': 10_000}))
    results.append(minimize_scalar(wrapped, bounds=(float(np.median(xs)), float(np.max(xs))), 
                                   method='bounded', options={'maxiter': 10_000}))
    results.append(minimize_scalar(wrapped, bracket=(float(np.min(xs)), float(np.max(xs))), 
                                   method='brent', options={'maxiter': 10_000}))
    results.append(minimize(wrapped, float(np.median(xs)), method='Nelder-Mead', 
                            options={'maxiter': 10_000}))
    results.append(minimize_scalar(wrapped, bounds=(float(np.min(xs)), float(np.max(xs))), 
                                   method='bounded', options={'maxiter': 10_000}))

    # If none succeeded (unlikely), expand once or twice
    if not any(getattr(r, "success", True) for r in results):
        for _ in range(2):
            width *= 2.0
            r = minimize_scalar(wrapped, bounds=(-width, width), method='bounded')
            results.append(r)
            if getattr(r, "success", True):
                break
    
    # Return the best value actually observed across all runs/evaluations
    #if wrapped.best_s is not None:
    return wrapped.best_u, wrapped.best_s
    
    #else:
        # Fallback (shouldn’t happen unless f never evaluated)
    #    return float(res.fun), float(res.x)

def l_u_brute(xs, ya, nx, ny, ix, iy, stat):
    """
    A more brute force minimization of U(s) by scanning all rank-constant intervals
    within +/- pooled variance.
    Returns (u_min, s_opt).
    """
    def f(s):
        if stat == 'cvm':
            return cvm_u(xs-s, ya, nx, ny, ix, iy)
        elif stat == 'ad':
            return ad2(xs-s, ya, nx, ny)
        else:
            raise ValueError('Invalid Test Statistic')
    
    # All possible crossing shifts where an x_i would equal a y_j after shifting
    # s = x_i - y_j. Between consecutive unique values, ranks are constant.
    diffs = (xs[:, None] - ya[None, :]).ravel()
    diffs = np.unique(diffs)
    diffs.sort()
    #np.min(xs-diffs[0]) ~= max(ya) and np.max(xs-diffs[-1]) == min(ya)
    
    # Representative points for each interval: midpoints between consecutive diffs,
    if diffs.size == 0:
        s_grid = np.array([0.0], dtype=float)
        
    elif diffs.size == 1:
        # one breakpoint -> two intervals; just pick something just below/above
        s_grid = np.array([diffs[0] - 1.0, diffs[0] + 1.0])
        
    else:
        mids = 0.5 * (diffs[:-1] + diffs[1:])
        sd = np.sqrt(np.var(xs, ddof=1) + np.var(ya, ddof=1))
        width = 0.5 * sd if np.isfinite(sd) and sd > 0 else 1.0
        mask = np.abs(mids) <= width
        s_grid = mids if not np.any(mask) else mids[mask]

    best_u = np.inf
    best_s = 0.0
    for s in s_grid:
        #print(s)
        #u = cvm_u(xs-s, ya, nx, ny, ix, iy)
        #u = ad2(xs-s, ya, nx, ny)
        u = f(s)
        
        if u < best_u:
            best_u, best_s = u, float(s)
            
            # early exit if perfect alignment ever happens (rare): U cannot go below 0
            if best_u == 0.0:
                break
    return best_u, best_s

def l_stats(xa, ya, nx, ny, ix, iy, k, N, brute, stat):
    #Calculate L-test
    #initial shift guesses based on sample statistics. 
    #One of them should be close to the shift depending on distribution shape
    
    stats = (np.median, np.mean, mode)
    shifts = [f(xa) - f(ya) for f in stats]
    if stat == 'cvm':
        u_vals = [cvm_u(xa - s, ya, nx, ny, ix, iy) for s in shifts]
    elif stat == 'ad':
        u_vals = [ad2(xa - s, ya, nx, ny) for s in shifts]
    else:
        raise ValueError('Invalid Test Statistic')
    
    s1 = shifts[int(np.argmin(u_vals))]
    
    xs = xa - s1
    
    if brute:
        u_min, s2 = l_u_brute(xs, ya, nx, ny, ix, iy, stat)
    else:
        u_min, s2 = l_u(xs, ya, nx, ny, ix, iy, stat)
    
    l_shift = float(s1 + s2) # full shift including initial guess
    
    return l_shift, u_min

def worker_a(args):
    xa, ya, nx, ny, ix, iy, k, N, u_min, brute, stat = args
    rng = np.random.default_rng()  # independent per worker
    xb = np.sort(rng.choice(xa, size=nx, replace=True))
    yb = np.sort(rng.choice(ya, size=ny, replace=True))
    s_b, _ = l_stats(xb, yb, nx, ny, ix, iy, k, N, brute, stat)  # re-estimates shift

    pool = np.concatenate([xb - s_b, yb])
    idx = rng.permutation(pool.size)
    xb = np.sort(pool[idx[:nx]])
    yb = np.sort(pool[idx[nx:]])
    _, u_b = l_stats(xb, yb, nx, ny, ix, iy, k, N, brute, stat)  # includes internal shift

    hit = int(u_b >= u_min)  #greater deviation on boostrap/permutated sample
    return hit, float(s_b)
    
#Versions A, B, and C are different methods of estimating p-value using bootstrap and permutation. 
#Option A (which ChatGPT-5 didn't like) gives by far best TypeI error result.
#Option A also has the benefit of calculating shift error using the same arrays
#Options B and C are in _l.py
def l_pval(xa, ya, nx, ny, ix, iy, k, N, u_min, B, brute=False, tol_p=0.05, 
           tol_s=0.05, workers=None, stat='cvm'):
    """
    Early-stops when relative p-value binomial error... 
    (np.sqrt(b*l_p*(1-l_p))/(b+1))/l_p < tol_p.
    Or relative Pebay online error [4] for the shift < tol_s (large sample limit 
                                                              1/sqrt(2(b-1)) < tol_s)
    Returns (l_p, shift_boot, shift_err). L-test p-value, mean/std shift of bootstrap
    """

    if workers==None:
        workers = cpu_count() or 1

    count = 0
    b = 0
    l_p = 1/(B+1)
    l_p_err = float("inf")
    
    # Running stats for s_b (Welford and Pébay)
    mean_s = 0.0
    M2 = M3 = M4 = 0.0
    
    if workers <= 1:
    # serial fallback (no processes)
        for _ in range(B):
            hit, s_b = worker_a((xa, ya, nx, ny, ix, iy, k, N, u_min, brute, stat))
            count += hit
            # update bootstrap shift running stats
            b, mean_s, M2, M3, M4 = update_s_stats(s_b, b, mean_s, M2, M3, M4)
            
            l_p = (count + 1.0) / (b + 1.0) # add-one smoothing
            l_p_err = np.sqrt(b*l_p*(1-l_p))/(b+1)
            p_rel = l_p_err/l_p if count > 5 else float("inf")
            
            #Current bootstrap shift standard deviation, error on the SD, and relative fraction                
            sd_s, se_sd_s, rse_sd_s = cur_s_std(b, M2, M4)

            # require at least a few hits and misses for a meaningful p
            enough_hits = (min(count, b - count) > 5)
            
            if enough_hits and (p_rel <= tol_p) and (rse_sd_s <= tol_s):
                break
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(worker_a, (xa, ya, nx, ny, ix, iy, k, N, u_min, 
                                            brute, stat)) 
                       for _ in range(B)]
            for fut in as_completed(futures):
                hit, s_b = fut.result()
                count += hit
                b, mean_s, M2, M3, M4 = update_s_stats(s_b, b, mean_s, M2, M3, M4)
    
                l_p = (count + 1.0) / (b + 1.0) # add-one smoothing
                l_p_err = np.sqrt(b*l_p*(1-l_p))/(b+1)
                p_rel = l_p_err/l_p if count > 10 else float("inf")
     
                #Current bootstrap shift standard deviation, error on the SD, and relative fraction                
                sd_s, se_sd_s, rse_sd_s = cur_s_std(b, M2, M4)

                # require at least a few hits and misses for a meaningful p
                enough_hits = (min(count, b - count) > 5)
                
                if enough_hits and (p_rel <= tol_p) and (rse_sd_s <= tol_s):
                    # cancel any not-yet-started tasks
                    for f in futures:
                        f.cancel()
                    # optional: ex.shutdown(cancel_futures=True)
                    break

    print('Bootstrap Samples: ' + str(b))
    
    shift_boot = float(mean_s)
    shift_err  = float(sd_s) if b > 1 else 0.0

    return l_p, float(l_p_err), shift_boot, shift_err

#x and y are samples
#B is *max* number of bootstrap trial for l_p p-value calculation
#tol_p is allowed percent uncertainty on p-value l_p
#workers is number of workers where default is os.cpu_count()
def ltest(x, y, B=5000, tol_p=0.05, tol_s=0.05, workers=None, brute=False, stat='cvm'):

    x = np.asarray(x)
    y = np.asarray(y)
    
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]

    nx = x.size
    ny = y.size
    
    if nx <= 1 or ny <= 1:
        raise ValueError('Arrays must have at least two values')
    
    xa = np.sort(x)
    ya = np.sort(y)
    
    k, N = nx*ny, nx + ny
    ix = np.arange(1, nx+1)
    iy = np.arange(1, ny+1)
    
    # #Calculate initial statistic and p-value according to regular CvM-test [1]
    
    # if min(nx,ny) > 40: #scipy cramervonmises_2samp uses max(nx,ny) > 20.
    #     cvm_method = 'asymptotic'

    # else:
    #     cvm_method = 'exact'
    
    # #u = cvm_u(xa, ya, nx, ny, ix, iy)
    # # compute T (eq. 9 in [2])
    # #t = cvm_stat(u, k, N) #CvM test statistic
    
    # #cvm_p = cvm_pval(u, k, N, t, nx, ny, cvm_method)
    
    # res = cramervonmises_2samp(xa, ya, method=cvm_method)
    
    # t = res.statistic
    # cvm_p = res.pvalue
    
    #Calculate shift that minimizes U (and therefore T [6]) and the minimum U    
    l_shift, u_min = l_stats(xa, ya, nx, ny, ix, iy, k, N, brute, stat)
    l_stat = cvm_stat(u_min, k, N) #T_min could be another name
    
    #CvM p-value which assumes no shifting was done. May be of interest.
    #l_p0 = cvm_pval(u_min, k, N, l_stat, nx, ny, cvm_method) #Function is in _l.py
    
    #Properly estimated L-test p-value, average bootstrap/permutation shift, 
    # and uncertainty on l_shift
    l_p, l_p_err, shift_boot, shift_err = \
        l_pval(xa, ya, nx, ny, ix, iy, k, N, u_min, B, brute=brute, tol_p=tol_p, 
               tol_s=tol_s, workers=workers, stat=stat)

    return l_p, l_p_err, l_shift, shift_boot, shift_err, l_stat
