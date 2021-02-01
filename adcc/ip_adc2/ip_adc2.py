from pyscf import gto, scf, adc, ao2mo, lib
from collections import namedtuple
import numpy as np
import warnings
import sys

from adcc import Tensor, AdcMatrix, AmplitudeVector
from adcc.functions import direct_sum, einsum
from adcc.Intermediates import Intermediates, register_as_intermediate
from adcc.timings import Timer
from adcc.adc_pp import matrix as ppmatrix
from adcc import block as block_getter
import libadcc


#NOTE: equations all have a -1 factor to appease select_eigenpairs in
# the Davidson solver. I don't like this, because whilst it means one 
# solves for IPs of the correct sign, you are solving for quasiparticle 
# states of the incorrect sign

#TODO: core-valence separation
#TODO: support for AdcMatrix.construct_symmetrisation_for_blocks

AdcBlock = namedtuple('AdcBlock', ['apply', 'diagonal'])

# Mean-field Fock matrix
def diagonal_h_h_0(hf, mp, intermediates):
    d = hf.foo.diagonal()
    return AmplitudeVector(ph=-d)

def block_h_h_0(hf, mp, intermediates):
    e_i = hf.foo.diagonal()
    def apply(ampl):
        v = e_i * ampl.ph
        return AmplitudeVector(ph=-v)
    return AdcBlock(apply, diagonal_h_h_0(hf, mp, intermediates))

# MP2 self-energy denominator
def diagonal_phh_phh_0(hf, mp, intermediates):
    return AmplitudeVector(pphh=-intermediates.adc2_i1)

def block_phh_phh_0(hf, mp, intermediates):
    def apply(ampl):
        v = intermediates.adc2_i1 * ampl.pphh
        return AmplitudeVector(pphh=-v)
    return AdcBlock(apply, diagonal_phh_phh_0(hf, mp, intermediates))

# MP2 self-energy coupling
def block_h_phh_1(hf, mp, intermediates):
    def apply(ampl):
        v = einsum('ija,ijak->k', ampl.pphh, hf.oovo) * np.sqrt(0.5)
        return AmplitudeVector(ph=-v)
    return AdcBlock(apply, 0)

def block_phh_h_1(hf, mp, intermediates):
    def apply(ampl):
        v = einsum('ijak,k->ija', hf.oovo, ampl.ph) * np.sqrt(0.5)
        v = v.antisymmetrise(0,1)
        return AmplitudeVector(pphh=-v)
    return AdcBlock(apply, 0)

@register_as_intermediate
def adc2_i1(hf, mp, intermediates):
    e_i = hf.foo.diagonal()
    e_a = hf.fvv.diagonal()
    return direct_sum('i,j,a->ija', e_i, e_i, -e_a).symmetrise(0,1)

# Approximate dynamics of particle MP2 self-energy folded into 1h
def diagonal_h_h_2(hf, mp, intermediates):
    d  = diagonal_h_h_0(hf, mp, intermediates)
    d += intermediates.adc2_i2.diagonal() * 2.0
    return d

def block_h_h_2(hf, mp, intermediates):
    e_i = hf.foo.diagonal()
    t2 = mp.t2(block_getter.oovv)
    def apply(ampl):
        v  = e_i * ampl.ph 
        v -= einsum('ij,j->i', intermediates.adc2_i2, ampl.ph)
        v -= einsum('j,ji->i', ampl.ph, intermediates.adc2_i2)
        return AmplitudeVector(ph=-v)
    return AdcBlock(apply, diagonal_h_h_2(hf, mp, intermediates))

@register_as_intermediate
def adc2_i2(hf, mp, intermediates):
    t2 = mp.t2(block_getter.oovv)
    return einsum('ikab,jkab->ij', t2, hf.oovv).symmetrise() * 0.25

# Null contributions:
def block_h_h_1(hf, mp, intermediates):
    return Adcblock(lambda ampl: 0, 0)

def block_phh_h_0(hf, mp, intermediates):
    return AdcBlock(lambda ampl: 0, 0)

def block_h_phh_0(hf, mp, intermediates):
    return AdcBlock(lambda ampl: 0, 0)

null_block = lambda *args: AdcBlock(lambda ampl: 0, 0)
block_h_h_1 = block_phh_h_0 = block_h_phh_0 = null_block

# Hack to avoid having to change the entire AdcMatrix code
ppmatrix.__dict__.update({
    'block_ph_ph_0': block_h_h_0,
    'block_ph_ph_1': block_h_h_1,
    'block_ph_ph_2': block_h_h_2,
    'block_ph_pphh_0': block_h_phh_0,
    'block_pphh_ph_0': block_phh_h_0,
    'block_ph_pphh_1': block_h_phh_1,
    'block_pphh_ph_1': block_phh_h_1,
    'block_pphh_pphh_0': block_phh_phh_0,
    'diagonal_pphh_pphh_0': diagonal_phh_phh_0,
    'adc2_i1': register_as_intermediate(adc2_i1),
    'adc2_i2': register_as_intermediate(adc2_i2),
})

def get_guesses_from_singles(matrix, nroots):
    h_diag = matrix.diagonal('s')

    if nroots >= h_diag.size:
        raise ValueError('nroots cannot be greater than number of '
                         'occupied states')

    arg = np.argsort(np.absolute(h_diag.to_ndarray()))[:nroots]
    guesses = []

    for i in arg:
        v = np.zeros(h_diag.shape)
        v[i] = 1.0
        h = Tensor(matrix.mospaces, space='o1')
        phh = Tensor(matrix.mospaces, space='o1o1v1', permutations=['ija', '-jia'])
        phh = phh.antisymmetrise(0,1)
        h.set_from_ndarray(v)
        ampl = AmplitudeVector(ph=h, pphh=phh)
        guesses.append(ampl)

    return guesses

def get_guesses_from_diag(matrix, nroots):
    h_diag = matrix.diagonal('s')
    phh_diag = matrix.diagonal('d')
    diag = np.concatenate([x.to_ndarray().ravel() for x in [h_diag, phh_diag]])

    arg = np.argsort(np.absolute(diag))[:nroots]
    guesses = []

    for x in arg:
        vi = np.zeros(h_diag.shape)
        vija = np.zeros(phh_diag.shape)

        if x < h_diag.size:
            vi[x] = 1.0
        else:
            i, j, a = np.unravel_index(x, phh_diag.shape)
            vija[i,j,a] = np.sqrt(0.5)
            vija[j,i,a] = -np.sqrt(0.5)

        h = Tensor(matrix.mospaces, 'o1')
        h.set_from_ndarray(vi)

        phh = Tensor(matrix.mospaces, 'o1o1v1', permutations=['ija','-jia'])
        phh.set_from_ndarray(vija)
        phh.antisymmetrise(0,1)

        ampl = AmplitudeVector(ph=h, pphh=phh)
        guesses.append(ampl)

    return guesses

def get_dense_matrix(matrix):
    ns = matrix.diagonal('s').size
    nd = matrix.diagonal('d').size

    m = np.zeros((ns+nd, ns+nd))

    for i in range(ns):
        v = np.zeros((ns))
        v[i] = 1.0
        h = Tensor(matrix.mospaces, 'o1')
        phh = Tensor(matrix.mospaces, 'o1o1v1')
        h.set_from_ndarray(v)
        ampl_in = AmplitudeVector(ph=h, pphh=phh)
        ampl_out = matrix.compute_matvec(ampl_in)
        m[i] = np.concatenate([x.to_ndarray().ravel() for x in ampl_out])

    for ija in range(nd):
        v = np.zeros((nd))
        v[ija] = 1.0
        h = Tensor(matrix.mospaces, 'o1')
        phh = Tensor(matrix.mospaces, 'o1o1v1')
        phh.set_from_ndarray(v)
        ampl_in = AmplitudeVector(ph=h, pphh=phh)
        ampl_out = matrix.compute_matvec(ampl_in)
        m[ns+ija] = np.concatenate([x.to_ndarray().ravel() for x in ampl_out])

    return m


if __name__ == '__main__':
    from adcc.solver import davidson
    from adcc import ReferenceState

    mol = gto.M(atom='O 0 0 0; H 0 0 1; H 0 1 0', basis='cc-pvdz', verbose=0)
    mol = gto.M(atom='O 0 0 0; O 0 0 1', basis='cc-pvdz', verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-12)
    nroots = 5
    print(adc.ADC(mf).kernel(nroots=nroots)[0])

    hf = ReferenceState(mf)
    matrix = AdcMatrix('adc2', hf)
    matrix.construct_symmetrisation_for_blocks = lambda: dict(d=lambda v: v)

    #guesses = get_guesses_from_singles(matrix, nroots)
    guesses = get_guesses_from_diag(matrix, nroots)

    def callback(state, identifier):
        return davidson.default_print(state, identifier, sys.stdout)

    kwargs = dict(n_ep=nroots, 
                  callback=callback, 
                  which='SM', 
                  debug_checks=True,
                  preconditioner=davidson.JacobiPreconditioner, 
                  preconditioning_method='Davidson'
    )
    davidson.eigsh(matrix, guesses, **kwargs)

