"""E3 calibration: power of the lag-1 factorization test at LLM series lengths.

Output: experiments/results/e3/power.json
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'sims'))
import coupled_agents as ca  # noqa: E402
from fathon.tensor import factorization as fz  # noqa: E402


def main():
    res = {}
    for T in (200, 1000):
        for eps in (0.0, 0.1, 0.2, 0.4):
            rej = [fz.shift_null(ca.simulate([[0, 1]], T=T, eps=eps, seed=s), [0], [1], 'phi2', k=1,
                                 nSurr=99, lag=1, minShift=10, seed=s)[3] <= 0.05 for s in range(40)]
            res['T={} eps={}'.format(T, eps)] = float(np.mean(rej))
    os.makedirs(os.path.join(HERE, 'results', 'e3'), exist_ok=True)
    with open(os.path.join(HERE, 'results', 'e3', 'power.json'), 'w') as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
