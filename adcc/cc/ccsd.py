'''
CCSD
'''

import time
from functools import partial
from collections import namedtuple

import adcc
from adcc.functions import *
from adcc.Intermediates import Intermediates, register_as_intermediate
from opt_einsum.contract import _tensordot

from diis import DIIS

#TODO non-canonical

EVAL_TAU = True
EVAL_ONE_PARTICLE = True
EVAL_TWO_PARTICLE = True
EVAL_AMPLITUDE = True

tensordot = partial(_tensordot, backend="libadcc")


@register_as_intermediate
def tau(hf, mp, intermediates):
    ''' Eq. 10 '''

    τ = intermediates.t2 + (
        + einsum('ia,jb->ijab', intermediates.t1, intermediates.t1)
        - einsum('ia,jb->ijba', intermediates.t1, intermediates.t1)
    )

    if EVAL_TAU:
        τ = τ.evaluate()

    return τ


@register_as_intermediate
def tau_tilde(hf, mp, intermediates):
    ''' Eq. 9 '''

    τ = intermediates.t2 + 0.5 * (
        + einsum('ia,jb->ijab', intermediates.t1, intermediates.t1)
        - einsum('ia,jb->ijba', intermediates.t1, intermediates.t1)
    )

    if EVAL_TAU:
        τ = τ.evaluate()

    return τ


@register_as_intermediate
def fvv(hf, mp, intermediates):
    ''' Eq. 3 '''

    fvv = (
        + 1.0 * einsum('mf,mafe->ae', intermediates.t1, hf.ovvv)
        - 0.5 * einsum('mnaf,mnef->ae', intermediates.tau_tilde, hf.oovv)
    )

    if EVAL_ONE_PARTICLE:
        fvv = fvv.evaluate()

    return fvv


@register_as_intermediate
def foo(hf, mp, intermediates):
    ''' Eq. 4 '''

    foo = (
        + 1.0 * einsum('ne,mnie->mi', intermediates.t1, hf.ooov)
        + 0.5 * einsum('inef,mnef->mi', intermediates.tau_tilde, hf.oovv)
    )
    
    if EVAL_ONE_PARTICLE:
        foo = foo.evaluate()

    return foo


@register_as_intermediate
def fov(hf, mp, intermediates):
    ''' Eq. 5 '''

    fov = einsum('nf,mnef->me', intermediates.t1, hf.oovv)
    
    if EVAL_ONE_PARTICLE:
        fov = fov.evaluate()

    return fov


@register_as_intermediate
def woooo(hf, mp, intermediates):
    ''' Eq. 6 '''

    pij = einsum('je,mnie->mnij', intermediates.t1, hf.ooov)

    if EVAL_TWO_PARTICLE:
        pij.evaluate()

    woooo = (
        + hf.oooo
        + pij
        - pij.transpose((0, 1, 3, 2))
        + 0.25 * einsum('ijef,mnef->mnij', intermediates.tau, hf.oovv)
    )
    
    if EVAL_TWO_PARTICLE:
        woooo = woooo.evaluate()

    return woooo


@register_as_intermediate
def wvvvv(hf, mp, intermediates):
    ''' Eq. 7 '''

    #pab = einsum('mb,amef->abef', intermediates.t1, hf.vovv)
    pab = tensordot(
            intermediates.t1,
            hf.vovv,
            axes=((0,), (1,)),
    ).transpose((1, 0, 2, 3))

    if EVAL_TWO_PARTICLE:
        pab.evaluate()

    wvvvv = (
        + hf.vvvv
        - pab
        + pab.transpose((1, 0, 2, 3))
        #+ 0.25 * einsum('mnab,mnef->abef', taus.τ, hf.oovv)
        + 0.25 * tensordot(
                intermediates.tau,
                hf.vvoo,
                axes=((0, 1), (2, 3)),
        )
    )
    
    if EVAL_TWO_PARTICLE:
        wvvvv = wvvvv.evaluate()

    return wvvvv


@register_as_intermediate
def wovvo(hf, mp, intermediates):
    ''' Eq. 8 '''

    tmp = 0.5 * intermediates.t2 + einsum('jf,nb->jnfb', intermediates.t1, intermediates.t1)
    if EVAL_TWO_PARTICLE:
        tmp.evaluate()

    wovvo = (
        + hf.ovvo
        + einsum('jf,mbef->mbej', intermediates.t1, hf.ovvv)
        - einsum('nb,mnej->mbej', intermediates.t1, hf.oovo)
        #- einsum('jnfb,mnef->mbej', tmp, hf.oovv)
        - tensordot(
                tmp,
                hf.oovv,
                axes=((1, 2), (1, 3)),
        ).transpose((2, 1, 3, 0))
    )
    
    if EVAL_TWO_PARTICLE:
        wovvo = wovvo.evaluate()

    return wovvo


def update_amplitudes(hf, mp, intermediates):
    ''' Update T1 and T2 amplitudes

        Warning: also clears intermediates since they are defined
        with respect to the previous set of amplitudes.
    '''

    ei = hf.fock('o1o1').diagonal()
    ea = hf.fock('v1v1').diagonal()
    eia = direct_sum('i-a->ia', ei, ea)
    eijab = direct_sum('ia+jb->ijab', eia, eia)

    t1_new = (
        + 1.0 * einsum('ie,ae->ia', intermediates.t1, intermediates.fvv)
        - 1.0 * einsum('ma,mi->ia', intermediates.t1, intermediates.foo)
        + 1.0 * einsum('imae,me->ia', intermediates.t2, intermediates.fov)
        - 1.0 * einsum('nf,naif->ia', intermediates.t1, hf.ovov)
        - 0.5 * einsum('imef,maef->ia', intermediates.t2, hf.ovvv)
        - 0.5 * einsum('mnae,nmei->ia', intermediates.t2, hf.oovo)
    ) / eia

    if EVAL_AMPLITUDE:
        t1_new = t1_new.evaluate()

    tmp = intermediates.fvv - 0.5 * einsum('mb,me->be', intermediates.t1, intermediates.fov)
    if EVAL_AMPLITUDE:
        tmp.evaluate()
    pab = (
        + einsum('ijae,be->ijab', intermediates.t2, tmp)
        - einsum('ma,mbij->ijab', intermediates.t1, hf.ovoo)
    )

    tmp = intermediates.foo + 0.5 * einsum('je,me->mj', intermediates.t1, intermediates.fov)
    if EVAL_AMPLITUDE:
        tmp.evaluate()
    pij = (
        + einsum('imab,mj->ijab', intermediates.t2, tmp)
        - einsum('ie,abej->ijab', intermediates.t1, hf.vvvo)
    )

    pijab = (
        #+ einsum('imae,mbej->ijab', intermediates.t2, wovvo)
        + tensordot(
                intermediates.t2,
                intermediates.wovvo,
                axes=((1, 3), (0, 2)),
        ).transpose((0, 3, 1, 2))
        - einsum('ie,ma,mbej->ijab', intermediates.t1, intermediates.t1, hf.ovvo)
    )

    if EVAL_AMPLITUDE:
        pab = pab.evaluate()
        pij = pij.evaluate()
        pijab = pijab.evaluate()

    t2_new = (
        + hf.oovv
        + pab
        - pab.transpose((0, 1, 3, 2))
        - pij
        + pij.transpose((1, 0, 2, 3))
        #+ 0.5 * einsum('mnab,mnij->ijab', taus.τ, woooo)
        #+ 0.5 * einsum('ijef,abef->ijab', taus.τ, wvvvv)
        + 0.5 * tensordot(
                intermediates.woooo,
                intermediates.tau,
                axes=((0, 1), (0, 1)),
        )
        + 0.5 * tensordot(
                intermediates.tau,
                intermediates.wvvvv,
                axes=((2, 3), (2, 3)),
        )
        + pijab
        - pijab.transpose((0, 1, 3, 2))
        - pijab.transpose((1, 0, 2, 3))
        + pijab.transpose((1, 0, 3, 2))
    ) / eijab

    if EVAL_AMPLITUDE:
        t2_new = t2_new.evaluate()

    intermediates.clear()
    intermediates.cached_tensors['t1'] = t1_new
    intermediates.cached_tensors['t2'] = t2_new


def compute_ccsd_energy(hf, mp, intermediates):
    ''' CCSD correlated energy '''

    e_ccsd = (
        + 0.25 * einsum('ijab,ijab->', hf.oovv, intermediates.t2)
        + 0.5  * einsum('ijab,ia,jb->', hf.oovv, intermediates.t1, intermediates.t1)
    )

    return e_ccsd


def run_ccsd(
        mf,
        reference_state=None,
        ground_state=None,
        intermediates=None,
        maxiter=50,
        tol=1e-7,
        diis_space=8,
):
    '''
    Run CCSD with adcc backend.
    '''

    if reference_state is None:
        reference_state = adcc.ReferenceState(mf)
    if ground_state is None:
        ground_state = adcc.LazyMp(reference_state)
    if intermediates is None:
        intermediates = Intermediates(ground_state)
        intermediates.cached_tensors['t1'] = zeros_like(reference_state.fock('o1v1'))
        intermediates.cached_tensors['t2'] = ground_state.t2('o1o1v1v1')

    diis = DIIS(intermediates)
    t0 = time.time()
    e_corr = 0.0

    for niter in range(1, maxiter+1):
        update_amplitudes(reference_state, ground_state, intermediates)
        diis(intermediates)

        e_corr, e_prev = compute_ccsd_energy(
                reference_state, ground_state, intermediates), e_corr

        print('%3d %12.8f %12.8f' % (niter, e_corr, mf.e_tot+e_corr))

        if abs(e_corr - e_prev) < tol:
            break

    print('Time elapsed: %.2f s' % (time.time() - t0))

    return e_corr, intermediates





if __name__ == '__main__':
    from pyscf import gto, scf, cc

    #mol = gto.M(atom='O 0 0 0; H 0 0 1; H 0 1 0', basis='cc-pvdz', verbose=0)
    mol = gto.M(atom='Li 0 0 0; H 0 0 1.64', basis='sto3g', verbose=0)
    rhf = scf.RHF(mol).run()

    run_ccsd(rhf)

    t0 = time.time()
    print(cc.CCSD(rhf).run().e_corr)
    t1 = time.time()
    print(t1-t0)


