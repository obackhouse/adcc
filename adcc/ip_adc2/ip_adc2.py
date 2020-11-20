from pyscf import gto, scf, adc, ao2mo, lib
from collections import namedtuple
import numpy as np
import warnings
import sys

from adcc import ReferenceState, LazyMp, Tensor, AdcMatrix, \
                 AdcMatrixlike, AdcMethod, AmplitudeVector
from adcc.functions import direct_sum, einsum
from adcc.solver import davidson
from adcc.Intermediates import Intermediates
from adcc.AdcMatrixPython import AdcMatrixPython
from adcc import block as block_getter
import libadcc


#NOTE: equations all have a -1 factor to appease select_eigenpairs in
# the Davidson solver. I don't like this, because whilst it means one 
# solves for IPs of the correct sign, you are solving for quasiparticle 
# states of the incorrect sign

#TODO: hole antisymm in ija diagrams, spin symmetry
#TODO: core-valence separation

AdcBlock = namedtuple('AdcBlock', ['apply', 'diagonal'])

# Mean-field Fock matrix
def diagonal_h_h_0(hf, mp, intermediates):
    d = hf.foo.diagonal()
    return AmplitudeVector(ph=-d)

def block_h_h_0(hf, mp, intermediates):
    e_i = hf.foo.diagonal()
    def apply(ampl):
        v = e_i * ampl.h
        return AmplitudeVector(ph=-v)
    return AdcBlock(apply, diagonal_h_h_0(hf, mp, intermediates))

# MP2 self-energy denominator
def diagonal_hhp_hhp_0(hf, mp, intermediates):
    d = intermediates.i3
    return AmplitudeVector(pphh=-d)

def block_hhp_hhp_0(hf, mp, intermediates):
    e_i = hf.foo.diagonal()
    e_a = hf.fvv.diagonal()
    e_ija = direct_sum('i,j,a->ija', e_i, e_i, -e_a)
    e_ija = intermediates.push('i3', e_ija)
    def apply(ampl):
        v = e_ija * ampl.pphh
        return AmplitudeVector(pphh=-v)
    return AdcBlock(apply, diagonal_hhp_hhp_0(hf, mp, intermediates))

# MP2 self-energy coupling
def block_h_hhp_1(hf, mp, intermediates):
    def apply(ampl):
        v = einsum('ija,ijak->k', ampl.pphh, hf.oovo) * np.sqrt(0.5)
        return AmplitudeVector(ph=-v)
    return AdcBlock(apply, 0)

def block_hhp_h_1(hf, mp, intermediates):
    def apply(ampl):
        v = einsum('ijak,k->ija', hf.oovo, ampl.ph) * np.sqrt(0.5)
        return AmplitudeVector(pphh=-v)
    return AdcBlock(apply, 0)

# Approximate dynamics of particle MP2 self-energy folded into 1h
def diagonal_h_h_2(hf, mp, intermediates):
    d  = diagonal_h_h_0(hf, mp, intermediates)
    d += intermediates.i1.diagonal() 
    d += intermediates.i2.diagonal()
    return d

def block_h_h_2(hf, mp, intermediates):
    e_i = hf.foo.diagonal()
    t2 = mp.t2(block_getter.oovv)
    i1_expr = einsum('ikab,jkab->ij', t2, hf.oovv) * 0.25
    i2_expr = einsum('ikab,jkab->ij', hf.oovv, t2) * 0.25
    i1 = intermediates.push('i1', i1_expr)
    i2 = intermediates.push('i2', i2_expr)
    def apply(ampl):
        v  = e_i * ampl.ph 
        v -= einsum('ij,j->i', i1, ampl.ph)
        v -= einsum('ij,j->i', i2, ampl.ph)
        return AmplitudeVector(ph=-v)
    return AdcBlock(apply, diagonal_h_h_2(hf, mp, intermediates))

#NOTE: AttributeError is raised if you name this anything else
class AdcMatrixPython(AdcMatrixPython):
    def __init__(self, hf_or_mp):
        if isinstance(hf_or_mp, (libadcc.ReferenceState, libadcc.HartreeFockSolution_i)):
            hf_or_mp = LazyMp(hf_or_mp)

        self.method = AdcMethod('adc2')
        self.ground_state = hf_or_mp
        self.reference_state = hf_or_mp.reference_state
        self.mospaces = hf_or_mp.reference_state.mospaces
        self.is_core_valence_separated = self.method.is_core_valence_separated
        assert not self.is_core_valence_separated 

        orders = dict(h_h=2, h_hhp=1, hhp_h=1, hhp_hhp=0)
        variant = 'cvs' if self.is_core_valence_separated else None

        self.intermediates = Intermediates()

        args = (self.reference_state, self.ground_state, self.intermediates)
        self.__blocks = {
                'h_h': block_h_h_2(*args),
                'h_hhp': block_h_hhp_1(*args),
                'hhp_h': block_hhp_h_1(*args),
                'hhp_hhp': block_hhp_hhp_0(*args),
        }
        self.__block_spaces = {
                's': ['o1'],
                'd': ['o1', 'o1', 'v1'],
        }
        self.blocks = list(self.__block_spaces.keys())

class AdcMatrix(AdcMatrix):
    def __init__(self, hf_or_mp):
        if isinstance(hf_or_mp, (libadcc.ReferenceState, libadcc.HartreeFockSolution_i)):
            hf_or_mp = LazyMp(hf_or_mp)

        self.method = AdcMethod('adc2')
        self.python = True

        AdcMatrixlike.__init__(self, AdcMatrixPython(hf_or_mp))

    def construct_symmetrisation_for_blocks(self):
        return dict(d=lambda v: v)

def get_guesses_from_singles(matrix, nroots):
    h_diag = matrix.diagonal('s')

    if nroots >= size:
        raise ValueError('nroots cannot be greater than number of '
                         'occupied states')

    arg = np.argsort(np.absolute(h_diag.to_ndarray()))[:nroots]
    guesses = []

    for i in arg:
        v = np.zeros(h_diag.shape)
        v[i] = 1.0
        h = Tensor(matrix.mospaces, space='o1')
        hhp = Tensor(matrix.mospaces, space='o1o1v1')
        h.set_from_ndarray(v)
        ampl = AmplitudeVector(ph=h, pphh=hhp)
        guesses.append(ampl)

    return guesses

def get_guesses_from_diag(matrix, nroots):
    h_diag = matrix.diagonal('s')
    hhp_diag = matrix.diagonal('d')
    diag = np.concatenate([x.to_ndarray().ravel() for x in [h_diag, hhp_diag]])

    arg = np.argsort(np.absolute(diag))[:nroots]
    guesses = []

    for x in arg:
        v = np.zeros(diag.shape)
        v[x] = 1.0
        h = Tensor(matrix.mospaces, 'o1')
        hhp = Tensor(matrix.mospaces, 'o1o1v1')
        h.set_from_ndarray(v[:h_diag.size])
        hhp.set_from_ndarray(v[h_diag.size:])
        ampl = AmplitudeVector(ph=h, pphh=hhp)
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
        hhp = Tensor(matrix.mospaces, 'o1o1v1')
        h.set_from_ndarray(v)
        ampl_in = AmplitudeVector(ph=h, pphh=hhp)
        ampl_out = matrix.compute_matvec(ampl_in)
        m[i] = np.concatenate([x.to_ndarray().ravel() for x in ampl_out])

    for ija in range(nd):
        v = np.zeros((nd))
        v[ija] = 1.0
        h = Tensor(matrix.mospaces, 'o1')
        hhp = Tensor(matrix.mospaces, 'o1o1v1')
        hhp.set_from_ndarray(v)
        ampl_in = AmplitudeVector(ph=h, pphh=hhp)
        ampl_out = matrix.compute_matvec(ampl_in)
        m[ns+ija] = np.concatenate([x.to_ndarray().ravel() for x in ampl_out])

    return m


if __name__ == '__main__':
    mol = gto.M(atom='O 0 0 0; H 0 0 1; H 0 1 0', basis='cc-pvdz', verbose=0)
    mol = gto.M(atom='O 0 0 0; O 0 0 1', basis='cc-pvdz', verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-12)
    nroots = 5
    print(adc.ADC(mf).kernel(nroots=nroots)[0])

    hf = ReferenceState(mf)
    matrix = AdcMatrix(hf)

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

