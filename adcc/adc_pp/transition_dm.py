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

from adcc import block as b
from adcc.AdcMethod import AdcMethod
from adcc.functions import einsum
from adcc.AmplitudeVector import AmplitudeVector
from adcc.OneParticleOperator import OneParticleOperator

import libadcc


def tdm_adc0(mp, amplitude, intermediates):
    if "s" not in amplitude.blocks:
        raise ValueError("transition_dm at ADC(0) level and beyond expects "
                         "an excitation amplitude with a singles part.")
    # C is either c(ore) or o(ccupied)
    C = b.c if mp.has_core_occupied_space else b.o

    u1 = amplitude["s"]
    if u1.subspaces != [C, b.v]:
        raise ValueError(f"Mismatch in subspaces singles part "
                         f"(== {u1.subspaces}), where {C}{b.v} was expected")

    # Transition density matrix for (CVS-)ADC(0)
    dm = OneParticleOperator(mp, is_symmetric=False)
    dm[b.v + C] = u1.transpose()
    return dm


def tdm_adc1(mp, amplitude, intermediates):
    dm = tdm_adc0(mp, amplitude, intermediates)  # Get ADC(0) result
    # adc1_dp0_ov
    dm[b.ov] = -einsum("ijab,jb->ia", mp.t2(b.oovv), amplitude["s"])
    return dm


def tdm_cvs_adc2(mp, amplitude, intermediates):
    # Get CVS-ADC(1) result (same as CVS-ADC(0))
    dm = tdm_adc0(mp, amplitude, intermediates)
    if "d" not in amplitude.blocks:
        raise ValueError("transition_dm at ADC(2) level and beyond "
                         "expects an excitation amplitude with a singles and a "
                         "doubles part.")
    u1 = amplitude["s"]
    u2 = amplitude["d"]
    if u2.subspaces != [b.o, b.c, b.v, b.v]:
        raise ValueError("Mismatch in subspaces doubles part "
                         f"(== {u2.subspaces}), where "
                         f"{b.o}{b.c}{b.v}{b.v} was expected.")

    t2 = mp.t2(b.oovv)
    p0_ov = intermediates.cv_p_ov
    p0_vv = intermediates.cv_p_vv

    # Compute CVS-ADC(2) tdm
    dm[b.oc] = (  # cvs_adc2_dp0_oc
        - einsum("ja,Ia->jI", p0_ov, u1)
        + (1 / sqrt(2)) * einsum("kIab,jkab->jI", u2, t2)
    )

    # cvs_adc2_dp0_vc
    dm[b.vc] = u1.transpose() - einsum("ab,Ib->aI", p0_vv, u1)
    return dm


def tdm_adc2(mp, amplitude, intermediates):
    dm = tdm_adc1(mp, amplitude, intermediates)  # Get ADC(1) result
    if "d" not in amplitude.blocks:
        raise ValueError("transition_dm at ADC(2) level and beyond "
                         "expects an excitation amplitude with a singles and a "
                         "doubles part.")
    u1 = amplitude["s"]
    u2 = amplitude["d"]
    if u2.subspaces != [b.o, b.o, b.v, b.v]:
        raise ValueError("Mismatch in subspaces doubles part "
                         f"(== {u2.subspaces}), where "
                         f"{b.o}{b.o}{b.v}{b.v} was expected.")

    t2 = mp.t2(b.oovv)
    td2 = mp.td2(b.oovv)
    p0_ov = mp.mp2_diffdm[b.ov]
    p0_oo = mp.mp2_diffdm[b.oo]
    p0_vv = mp.mp2_diffdm[b.vv]

    # Compute ADC(2) tdm
    dm[b.oo] = (  # adc2_dp0_oo
        - einsum("ia,ja->ij", p0_ov, u1)
        - einsum("ikab,jkab->ij", u2, t2)
    )
    dm[b.vv] = (  # adc2_dp0_vv
        + einsum("ia,ib->ab", u1, p0_ov)
        + einsum("ijac,ijbc->ab", u2, t2)
    )
    dm[b.ov] -= einsum("ijab,jb->ia", td2, u1)  # adc2_dp0_ov
    dm[b.vo] += 0.5 * (  # adc2_dp0_vo
        + einsum("ijab,jkbc,kc->ai", t2, t2, u1)
        - einsum("ab,ib->ai", p0_vv, u1)
        + einsum("ja,ij->ai", u1, p0_oo)
    )
    return dm


DISPATCH = {
    "adc0": tdm_adc0,
    "adc1": tdm_adc1,
    "adc2": tdm_adc2,
    "adc2x": tdm_adc2,
    "cvs-adc0": tdm_adc0,
    "cvs-adc1": tdm_adc0,  # No extra contribs for CVS-ADC(1)
    "cvs-adc2": tdm_cvs_adc2,
    "cvs-adc2x": tdm_cvs_adc2,
}


def transition_dm(method, ground_state, amplitude, intermediates=None):
    """
    Compute the one-particle transition density matrix from ground to excited
    state in the MO basis.

    Parameters
    ----------
    method : str, AdcMethod
        The method to use for the computation (e.g. "adc2")
    ground_state : LazyMp
        The ground state upon which the excitation was based
    amplitude : AmplitudeVector
        The amplitude vector
    intermediates : AdcIntermediates
        Intermediates from the ADC calculation to reuse
    """
    if not isinstance(method, AdcMethod):
        method = AdcMethod(method)
    if not isinstance(ground_state, libadcc.LazyMp):
        raise TypeError("ground_state should be a LazyMp object.")
    if not isinstance(amplitude, AmplitudeVector):
        raise TypeError("amplitude should be an AmplitudeVector object.")
    if intermediates is None:
        intermediates = libadcc.AdcIntermediates(ground_state)

    if method.name not in DISPATCH:
        raise NotImplementedError("transition_dm is not implemented "
                                  f"for {method.name}.")
    else:
        ret = DISPATCH[method.name](ground_state, amplitude, intermediates)
        return ret.evaluate()
