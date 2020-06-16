#!/usr/bin/env python3
## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
## ---------------------------------------------------------------------
##
## Copyright (C) 2020 by the adcc authors
##
## This file is part of adcc.
##
## adcc is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published
## by the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## adcc is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with adcc. If not, see <http://www.gnu.org/licenses/>.
##
## ---------------------------------------------------------------------
import adcc

from pyscf import gto, scf

from adcc import direct_sum, einsum, evaluate
from adcc.timings import Timer
from adcc.workflow import construct_adcmatrix, obtain_guesses_by_inspection


def assert_allclose(a, b):
    from numpy.testing import assert_array_almost_equal_nulp

    assert_array_almost_equal_nulp(a.to_ndarray(), b.to_ndarray(), nulp=10)


class MpEquations:
    def __init__(self, reference_state):
        self.reference_state = reference_state

    def get_t2(self):
        # Not verified
        hf = self.reference_state
        df = hf.df("o1v1")
        return 4.0 * hf.oovv / direct_sum("ia+jb->ijab", df, df).symmetrise(0, 1)


class Adc2Intermediates:
    def __init__(self, ground_state):
        self.ground_state = ground_state
        self.reference_state = ground_state.reference_state

    def get_i1(self):
        # Not verified
        hf = self.reference_state
        t2 = self.ground_state.t2("o1o1v1v1")
        return 0.5 * einsum("ijac,ijbc->ab", t2, hf.oovv).symmetrise()

    def get_i2(self):
        # Not verified
        hf = self.reference_state
        t2 = self.ground_state.t2("o1o1v1v1")
        return 0.5 * einsum("ikab,jkab->ij", t2, hf.oovv).symmetrise()


class Adc2Matrix:
    def __init__(self, ground_state, intermediates):
        self.intermediates = intermediates
        self.ground_state = ground_state
        self.reference_state = ground_state.reference_state

    def matrices(self):
        hf = self.reference_state
        self.intermediates.adc2_i1
        self.intermediates.adc2_i2
        self.ground_state.t2("o1o1v1v1")
        hf.oovv
        hf.ovov
        hf.ovvv
        hf.ooov

    def apply_ss(self, u1):
        i1 = self.intermediates.adc2_i1
        i2 = self.intermediates.adc2_i2
        t2 = self.ground_state.t2("o1o1v1v1")
        hf = self.reference_state

        return (
            + einsum("ib,ab->ia", u1, hf.fvv + i1)
            - einsum("ij,ja->ia", hf.foo - i2, u1)
            - einsum("jaib,jb->ia", hf.ovov, u1)
            - 0.5 * einsum("ijab,jkbc,kc->ia", t2, hf.oovv, u1)
            - 0.5 * einsum("ijab,jkbc,kc->ia", hf.oovv, t2, u1)
        )

    def apply_sd(self, u2):
        hf = self.reference_state
        return (
            + einsum("jkib,jkab->ia", hf.ooov, u2)
            + einsum("ijbc,jabc->ia", u2, hf.ovvv)
        )

    def apply_ds(self, u1):
        hf = self.reference_state
        return (
            + einsum("ic,jcab->ijab", u1, hf.ovvv).antisymmetrise(0, 1)
            - einsum("ijka,kb->ijab", hf.ooov, u1).antisymmetrise(2, 3)
        )

    def apply_dd(self, u2):
        hf = self.reference_state
        return (
            + 2 * einsum("ijac,bc->ijab", u2, hf.fvv).antisymmetrise(2, 3)
            - 2 * einsum("ik,kjab->ijab", hf.foo, u2).antisymmetrise(0, 1)
        )

    def get_d(self, u1, u2):
        return self.apply_ds(u1) + self.apply_dd(u2)

    def get_s(self, u1, u2):
        return self.apply_ss(u1) + self.apply_sd(u2)

    def __matmul__(self, u):
        return adcc.AmplitudeVector(self.get_s(u["s"], u["d"]),
                                    self.get_d(u["s"], u["d"]))


def setup(basis="sto-3g", system="water", n_threads=1):
    if system == "water":
        atom = """
            O 0 0 0
            H 0 0 1.795239827225189
            H 1.693194615993441 0 -0.599043184453037
        """
    elif system == "furane":
        atom = """
            C    -8.6177788253     2.0112153891  0.00051235659219
            C    -6.3040115561     3.1274648798  -0.0006541676556
            C    -4.5137879088     1.1286427651  0.00072548644099
            C    -5.8772815311    -1.0486071846  -0.00038821119252
            O      -8.38534184   -0.53776466715  -8.8403977675e-05
            H    -10.507432007     2.7282820239  0.00078101962987
            H    -5.9396014956     5.1171089212  -0.0012237234603
            H    -2.4961654416     1.2724406465  0.0011406258647
            H    -5.3719819944    -3.0055553217  -0.00080498224167
        """
    mol = gto.M(
        atom=atom,
        basis=basis,
        unit="Bohr"
    )
    scfres = scf.RHF(mol)
    scfres.conv_tol = 1e-8
    scfres.kernel()

    refmatrix = construct_adcmatrix(scfres, method="adc2")
    guesses = obtain_guesses_by_inspection(refmatrix, 5, "singlet")

    adcc.set_n_threads(n_threads)
    return refmatrix, guesses[1]


def individual_test(matrix, refmatrix, guess):
    cpy = guess.copy()
    for s in ["s", "d"]:
        for t in ["s", "d"]:
            print("testing", s, t)
            refmatrix.compute_apply(s + t, guess[t], cpy[s])
            fun = "apply_" + s + t
            assert_allclose(getattr(matrix, fun)(guess[t]), cpy[s])


def timings(matrix, refmatrix, guess, repeat=300):
    cpy = guess.copy()
    timer = Timer()

    for s in ["s", "d"]:
        for t in ["s", "d"]:
            with timer.record(f"{s}{t} libadcc"):
                for i in range(repeat):
                    refmatrix.compute_apply(s + t, guess[t], cpy[s])

            with timer.record(f"{s}{t} newimpl"):
                fun = "apply_" + s + t
                for i in range(repeat):
                    evaluate(getattr(matrix, fun)(guess[t]))

            tlibadcc = timer.total(f"{s}{t} libadcc") / repeat
            tnewimpl = timer.total(f"{s}{t} newimpl") / repeat
            diff = tnewimpl - tlibadcc

            c = " "
            if tlibadcc < tnewimpl:
                c = "<"
            print(f"{s}{t}   {tlibadcc * 1000:8.2f} ms {c} "
                  f"{tnewimpl * 1000:8.2f} ms "
                  f"= {diff * 1000:8.2f} ms")


def full_timings(matrix, refmatrix, guess, repeat=300):
    timer = Timer()

    with timer.record("full libadcc"):
        for i in range(repeat):
            refmatrix @ guess

    with timer.record("full newimpl"):
        for i in range(repeat):
            evaluate(matrix @ guess)

    tlibadcc = timer.total("full libadcc") / repeat
    tnewimpl = timer.total("full newimpl") / repeat
    diff = tnewimpl - tlibadcc

    c = " "
    if tlibadcc < tnewimpl:
        c = "<"
    print(f"full {tlibadcc * 1000:8.2f} ms {c} {tnewimpl * 1000:8.2f} ms "
          f"= {diff * 1000:8.2f} ms")


def run_timings(system="water", basis="sto-3g", n_threads=1,
                testing=False, repeat=300):
    refmatrix, guess = setup(basis, system, n_threads=n_threads)
    nocc = refmatrix.mospaces.n_orbs("o1")
    nvirt = refmatrix.mospaces.n_orbs("v1")
    matrix = Adc2Matrix(refmatrix.ground_state, refmatrix.intermediates)
    matrix.matrices()
    print(f"nocc = {nocc}   nvirt = {nvirt}")

    if testing:
        individual_test(matrix, refmatrix, guess)

    print("       libadcc     newimpl      old - new")
    timings(matrix, refmatrix, guess, repeat=repeat)
    full_timings(matrix, refmatrix, guess, repeat=repeat)


def run_threads(system="water", basis="sto-3g", repeat=100, slim=False):
    refmatrix, guess = setup(basis, system)
    nocc = refmatrix.mospaces.n_orbs("o1")
    nvirt = refmatrix.mospaces.n_orbs("v1")
    matrix = Adc2Matrix(refmatrix.ground_state, refmatrix.intermediates)
    matrix.matrices()
    print(f"nocc = {nocc}   nvirt = {nvirt}")

    print("   libadcc     newimpl")
    for n_threads in [1, 2, 4]:
        adcc.set_n_threads(n_threads)
        if not slim:
            timings(matrix, refmatrix, guess, repeat=repeat)
        full_timings(matrix, refmatrix, guess, repeat=repeat)
        print()
