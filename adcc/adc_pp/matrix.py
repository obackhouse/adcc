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
from math import sqrt
from collections import namedtuple

from adcc import block as b
from adcc.functions import direct_sum, einsum, zeros_like
from adcc.Intermediates import Intermediates
from adcc.AmplitudeVector import AmplitudeVector

__all__ = ["block"]

# TODO Extend AmplitudeVector to cases where only the doubles block is present.

#
# Dispatch routine
#

"""
`apply` is a function mapping an AmplitudeVector to the contribution of this
block to the result of applying the ADC matrix. `diagonal` is an `AmplitudeVector`
containing the expression to the diagonal of the ADC matrix from this block.
"""
AdcBlock = namedtuple("AdcBlock", ["apply", "diagonal"])


def block(ground_state, spaces, order, variant=None,
          intermediates=Intermediates()):
    """
    Gets ground state, potentially intermediates, spaces (ph, pphh and so on)
    and the perturbation theory order for the block,
    variant is "cvs" or sth like that.

    It is assumed largely, that CVS is equivalent to mp.has_core_occupied_space,
    while one would probably want in the long run that one can have an "o2" space,
    but not do CVS
    """
    if isinstance(variant, str):
        variant = [variant]
    elif variant is None:
        variant = []
    reference_state = ground_state.reference_state

    if ground_state.has_core_occupied_space and "cvs" not in variant:
        raise ValueError("Cannot run a general (non-core-valence approximated) "
                         "ADC method on top of a ground state with a "
                         "core-valence separation.")
    if not ground_state.has_core_occupied_space and "cvs" in variant:
        raise ValueError("Cannot run a core-valence approximated ADC method on "
                         "top of a ground state without a "
                         "core-valence separation.")

    fn = "_".join(["block"] + variant + spaces + [str(order)])

    if fn not in globals():
        raise ValueError("Could not dispatch: "
                         f"spaces={spaces} order={order} variant=variant")
    return globals()[fn](reference_state, ground_state, intermediates)


#
# 0th order general
#
def diagonal_ph_ph_0(hf):
    fCC = hf.fcc if hf.has_core_occupied_space else hf.foo
    return AmplitudeVector(ph=direct_sum("a-i->ia", hf.fvv.diagonal(),
                                         fCC.diagonal()))


def block_ph_ph_0(hf, mp, intermediates):
    def apply(ampl):
        fCC = hf.fcc if hf.has_core_occupied_space else hf.foo
        return AmplitudeVector(ph=(
            + einsum("ib,ab->ia", ampl.ph, hf.fvv)
            - einsum("IJ,Ja->Ia", fCC, ampl.ph)
        ))
    return AdcBlock(apply, diagonal_ph_ph_0(hf))


def block_ph_pphh_0(hf, mp, intermediates):
    return AdcBlock(lambda ampl: 0, 0)


def block_pphh_ph_0(hf, mp, intermediates):
    return AdcBlock(lambda ampl: 0, 0)


def diagonal_pphh_pphh_0(hf):
    # Note: adcman similarly does not symmetrise the occupied indices
    #       (for both CVS and general ADC)
    fCC = hf.fcc if hf.has_core_occupied_space else hf.foo
    res = direct_sum("-i-J+a+b->iJab",
                     hf.foo.diagonal(), fCC.diagonal(),
                     hf.fvv.diagonal(), hf.fvv.diagonal())
    return AmplitudeVector(pphh=res.symmetrise(2, 3))


def block_pphh_pphh_0(hf, mp, intermediates):
    def apply(ampl):
        return AmplitudeVector(pphh=(
            + 2 * einsum("ijac,bc->ijab", ampl.pphh, hf.fvv).antisymmetrise(2, 3)
            - 2 * einsum("ik,kjab->ijab", hf.foo, ampl.pphh).antisymmetrise(0, 1)
        ))
    return AdcBlock(apply, diagonal_pphh_pphh_0(hf))


#
# 0th order CVS
#
diagonal_cvs_ph_ph_0 = diagonal_ph_ph_0
diagonal_cvs_pphh_pphh_0 = diagonal_pphh_pphh_0
block_cvs_ph_ph_0 = block_ph_ph_0
block_cvs_ph_pphh_0 = block_ph_pphh_0
block_cvs_pphh_ph_0 = block_pphh_ph_0


def block_cvs_pphh_pphh_0(hf, mp, intermediates):
    def apply(ampl):
        return AmplitudeVector(pphh=(
            + 2 * einsum("iJac,bc->iJab", ampl.pphh, hf.fvv).antisymmetrise(2, 3)
            - einsum("ik,kJab->iJab", hf.foo, ampl.pphh)
            - einsum("JK,iKab->iJab", hf.fcc, ampl.pphh)
        ))
    return AdcBlock(apply, diagonal_cvs_pphh_pphh_0(hf))


#
# 1st order general
#
def diagonal_ph_ph_1(hf):
    CvCv = hf.cvcv if hf.has_core_occupied_space else hf.ovov
    return diagonal_ph_ph_0(hf) - AmplitudeVector(ph=einsum("IaIa->Ia", CvCv))


def block_ph_ph_1(hf, mp, intermediates):
    fCC = hf.fcc if hf.has_core_occupied_space else hf.foo
    CvCv = hf.cvcv if hf.has_core_occupied_space else hf.ovov

    def apply(ampl):
        return AmplitudeVector(ph=(
            + einsum("ib,ab->ia", ampl.ph, hf.fvv)
            - einsum("ij,ja->ia", fCC, ampl.ph)
            - einsum("jaib,jb->ia", CvCv, ampl.ph)
        ))
    return AdcBlock(apply, diagonal_ph_ph_1(hf))


def block_ph_pphh_1(hf, mp, intermediates):
    def apply(ampl):
        return AmplitudeVector(ph=(
            + einsum("jkib,jkab->ia", hf.ooov, ampl.pphh)
            + einsum("ijbc,jabc->ia", ampl.pphh, hf.ovvv)
        ))
    return AdcBlock(apply, 0)


def block_pphh_ph_1(hf, mp, intermediates):
    def apply(ampl):
        return AmplitudeVector(pphh=(
            + einsum("ic,jcab->ijab", ampl.ph, hf.ovvv).antisymmetrise(0, 1)
            - einsum("ijka,kb->ijab", hf.ooov, ampl.ph).antisymmetrise(2, 3)
        ))
    return AdcBlock(apply, 0)


#
# 1st order CVS
#
diagonal_cvs_ph_ph_1 = diagonal_ph_ph_1
block_cvs_ph_ph_1 = block_ph_ph_1


def block_cvs_ph_pphh_1(hf, mp, intermediates):
    def apply(ampl):
        return AmplitudeVector(ph=(
            + sqrt(2) * einsum("jKIb,jKab->Ia", hf.occv, ampl.pphh)
            - 1 / sqrt(2) * einsum("jIbc,jabc->Ia", ampl.pphh, hf.ovvv)
        ))
    return AdcBlock(apply, 0)


def block_cvs_pphh_ph_1(hf, mp, intermediates):
    def apply(ampl):
        return AmplitudeVector(pphh=(
            + sqrt(2) * einsum("jIKb,Ka->jIab",
                               hf.occv, ampl.ph).antisymmetrise(2, 3)
            - 1 / sqrt(2) * einsum("Ic,jcab->jIab", ampl.ph, hf.ovvv)
        ))
    return AdcBlock(apply, 0)


#
# 2nd order general
#
def diagonal_ph_ph_2(hf, mp, intermediates):
    i1 = intermediates.adc2_i1
    i2 = intermediates.adc2_i2
    return diagonal_ph_ph_1(hf) + AmplitudeVector(ph=(
        + direct_sum("a+i->ia", i1.diagonal(), i2.diagonal())
        - einsum("ikac,ikac->ia", mp.t2(b.oovv), hf.oovv)
    ))


def block_ph_ph_2(hf, mp, intermediates):
    # TODO Ideas: Add fvv to i1 and foo to i2 once and for all
    #      How about precomputing the contractions
    #      ijab,jkbc,kc->ia and ijab,jkbc,kc->ia ?
    t2 = mp.t2(b.oovv)
    i1_expr = 0.5 * einsum("ijac,ijbc->ab", t2, hf.oovv).symmetrise()
    i2_expr = 0.5 * einsum("ikab,jkab->ij", t2, hf.oovv).symmetrise()
    i1 = intermediates.push("adc2_i1", i1_expr)
    i2 = intermediates.push("adc2_i2", i2_expr)

    def apply(ampl):
        return AmplitudeVector(ph=(
            + einsum("ib,ab->ia", ampl.ph, hf.fvv + i1)
            - einsum("ij,ja->ia", hf.foo - i2, ampl.ph)
            - einsum("jaib,jb->ia", hf.ovov, ampl.ph)
            - 0.5 * einsum("ijab,jkbc,kc->ia", t2, hf.oovv, ampl.ph)
            - 0.5 * einsum("ijab,jkbc,kc->ia", hf.oovv, t2, ampl.ph)
        ))
    return AdcBlock(apply, diagonal_ph_ph_2(hf, mp, intermediates))


#
# 2nd order CVS
#
def diagonal_cvs_ph_ph_2(hf, mp, intermediates):
    i1 = intermediates.adc2_i1
    zeros_c = zeros_like(hf.orbital_energies(b.c))
    return diagonal_cvs_ph_ph_1(hf, mp) + AmplitudeVector(ph=(
        direct_sum("a+i->ia", i1.diagonal(), zeros_c)
    ))


def block_cvs_ph_ph_2(hf, mp, intermediates):
    t2 = mp.t2(b.oovv)
    i1_expr = 0.5 * einsum("ijac,ijbc->ab", t2, hf.oovv).symmetrise()
    i1 = intermediates.push("adc2_i1", i1_expr)

    def apply(ampl):
        return AmplitudeVector(ph=(
            + einsum("ib,ab->ia", ampl.ph, hf.fvv + i1)
            - einsum("ij,ja->ia", hf.fcc, ampl.ph)
            - einsum("JaIb,Jb->Ia", hf.cvcv, ampl.ph)
        ))
    return AdcBlock(apply, diagonal_cvs_ph_ph_2(hf, mp, intermediates))
