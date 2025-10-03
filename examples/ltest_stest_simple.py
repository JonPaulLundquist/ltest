#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Jon Paul Lundquist
"""
Created on Wed Oct  1 11:34:10 2025

A much simpler test of shift accuracy.

Example runs:
    
    mu = 1
    shift = 5
    sigma_x = 3
    sigma_y = 5
    nx = 20
    ny = 20
    x = rng.normal(mu, sigma_x, size=nx)
    x = np.concatenate((x, rng.uniform(-sigma_x,2*sigma_x,size=15)))
    y = rng.normal(mu+shift, sigma_y, size=ny)
    
    l_shift
    -4.652018894109061 #Mean of samples
    1.4115607738098932 #std of samples
    shift_boot
    -4.67359943999969
    1.3319652358999856
    mean shift
    -4.648681023092828
    1.2453674698627826
    median shift
    -4.714260165409821
    1.5203243318999276
    
    Shift_boot more accurate than mean difference with less variance than median 
    with for two Gaussian with different sigma 43% noise.
    
    mu = 1
    shift = 5
    sigma_x = 3
    sigma_y = 3
    nx = 20
    ny = 20
    
    x = rng.normal(mu, sigma_x, size=nx)
    x = np.concatenate((x, rng.uniform(-sigma_x,2*sigma_x,size=15)))
    y = rng.normal(mu+shift, sigma_y, size=ny)
    
    l_shift
    -4.7678854736272855
    0.9490739819309353
    shift_boot
    -4.758738206884455
    0.923941925283994
    mean shift
    -4.765993700846819
    0.8615101422208603
    median shift
    -4.7810089480787985
    1.1253728893035198
    
    Same sigma 43% noise. Median_shift most accurate but highest variance. 
    l_shift has a good mix of accuracy and variance.
    
@author: jplundquist
"""

import math
import numpy as np
from ltest import ltest
from scipy.stats import bootstrap

rng = np.random.default_rng()

mu = 1
shift = 5
sigma_x = 3
sigma_y = 5
nx = 20
ny = 20

iterations = 500
l_shift = np.zeros(iterations)
shift_boot = np.zeros(iterations)
shift_err = np.zeros(iterations)
mean_shift = np.zeros(iterations)
mean_err = np.zeros(iterations)
median_shift = np.zeros(iterations)
median_err = np.zeros(iterations)

for i in range(iterations):
    x = rng.normal(mu, sigma_x, size=nx)
    x = np.concatenate((x, rng.uniform(-sigma_x,2*sigma_x,size=15)))
    y = rng.normal(mu+shift, sigma_y, size=ny)
    
    l_p, l_p_err, l_shift[i], shift_boot[i], shift_err[i], l_stat = \
    ltest(x,y, B=15000, tol_p = 0.1, tol_s = 0.005, brute=True) #Gotta be accurate here on shift...
    
    mean_shift[i] = np.mean(x)-np.mean(y)
    #mean_err = np.sqrt(np.var(x, ddof=1)**2/nx + np.var(y, ddof=1)**2/ny)
    res = bootstrap((x, y), lambda a, b: np.mean(a) - np.mean(b),
                    vectorized=False, n_resamples=10_000)
    mean_err[i]   = getattr(res, "standard_error",  # SciPy ≥ 1.9
                         (res.confidence_interval.high - res.confidence_interval.low)/(2*1.9599639845))
    
    median_shift[i] = np.median(x)-np.median(y)
    #mean_err = np.sqrt(np.var(x, ddof=1)**2/nx + np.var(y, ddof=1)**2/ny)
    res = bootstrap((x, y), lambda a, b: np.median(a) - np.median(b),
                    vectorized=False, n_resamples=10_000)
    median_err[i]   = getattr(res, "standard_error",  # SciPy ≥ 1.9
                         (res.confidence_interval.high - res.confidence_interval.low)/(2*1.9599639845))

print('l_shift')
print(np.mean(l_shift))
print(np.std(l_shift, ddof=1))
print('shift_boot')
print(np.mean(shift_boot))
print(np.std(shift_boot, ddof=1))
print('mean shift')
print(np.mean(mean_shift))
print(np.std(mean_shift, ddof=1))
print('median shift')
print(np.mean(median_shift))
print(np.std(median_shift, ddof=1))