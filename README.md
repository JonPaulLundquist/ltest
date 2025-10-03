# ltest — Shift-invariant two-sample CvM (the L-test)

The **L-test** compares two samples while allowing a free location shift. It minimizes the squared \(L_2\) distance between ECDFs (Cramér–von Mises style) over a scalar shift \(s\). It returns a Monte-Carlo p-value, its uncertainty, a shift estimate, its uncertainty, and the minimized statistic.

## Install
```bash
pip install numpy scipy
# local install from source
pip install -e .

## Usage
import numpy as np
from ltest import ltest

rng = np.random.default_rng(0)
x = rng.normal(size=200)
y = rng.normal(loc=0.3, scale=1.2, size=220)

l_p, l_p_err, l_shift, shift_boot, shift_err, l_stat = ltest(
    x, y, B=1000, tol_p=0.05, tol_s=0.05, workers=None, brute=False
)
print(l_p, l_shift, shift_err)

## Notes
Parallel bootstrap with early stopping by relative error on p and on the shift SD.
Optional “brute” search of s via rank-change breakpoints (slower).
See directory examples/ for statistical verifications such as Type I/II power and shift-accuracy experiments.
Codes is in tests/. run test_basic.py for pytest.
On Windows/macOS, protect entry point when using multiprocessing:
if __name__ == "__main__":
    # call ltest(...)

## Dependencies
Python ≥ 3.8
NumPy ≥ 1.23
SciPy ≥ 1.9

## License
This project is licensed under the **CC BY-NC 4.0** license.  
See the full text in [LICENSE](./LICENSE).
