'''
DIIS
'''

import adcc
from adcc.functions import *
import numpy as np


class DIIS:
    def __init__(self, intermediates, size=8):
        t1, t2 = intermediates.t1, intermediates.t2
        self.size = size
        self.t1s, self.e1s = [], []
        self.t2s, self.e2s = [], []
        self._t1_prev = t1
        self._t2_prev = t2
        self.b = np.zeros((0,0))

    def push(self, t1, t2):
        self.t1s.append(t1)
        self.t2s.append(t2)

        self.e1s.append(t1-self._t1_prev)
        self.e2s.append(t2-self._t2_prev)
        n = len(self.e1s)
        b = self.b

        bdiag = list(np.diag(b)) if b is not None else []
        bdiag.append(
            + einsum('ia,ia->', self.e1s[-1], self.e1s[-1])
            + einsum('iajb,iajb->', self.e2s[-1], self.e2s[-1])
        )

        arg = np.argmax(bdiag)

        if n > self.size:
            del self.t1s[arg], self.t2s[arg]
            del self.e1s[arg], self.e2s[arg]
            b = np.delete(b, arg, 0)
            b = np.delete(b, arg, 1)
            n -= 1

        b = np.pad(b, ((0,1),(0,1)), 'constant')

        for i in range(n):
            b[i,-1] = b[-1,i] = (
                + einsum('ia,ia->', self.e1s[i], self.e1s[-1])
                + einsum('iajb,iajb->', self.e2s[i], self.e2s[-1])
            )

        self.b = b

    def __call__(self, intermediates):
        t1, t2 = intermediates.t1, intermediates.t2
        self.push(t1, t2)
        n = len(self.e1s)

        b = -np.ones((n+1, n+1))
        b[n, n] = 0.0
        b[:n, :n] = self.b / np.absolute(self.b).max()

        z = np.zeros((n+1))
        z[n] = -1.0

        try:
            cs = np.linalg.solve(b, z)

            t1_new = sum([float(c)*t for c,t in zip(cs, self.t1s)])
            t2_new = sum([float(c)*t for c,t in zip(cs, self.t2s)])
            self._t1_prev, self._t2_prev = t1_new, t2_new

            intermediates.cached_tensors['t1'] = t1_new
            intermediates.cached_tensors['t2'] = t2_new

        except np.linalg.LinAlgError:
            print('Singularity detected in DIIS')
