#!/usr/bin/env python3
## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
## ---------------------------------------------------------------------
##
## Copyright (C) 2019 by the adcc authors
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
import unittest
import itertools

from adcc.testdata.cache import cache
from adcc.backends.testing import cached_backend_hf

import pytest

from .misc import expand_test_templates
from .test_ReferenceState_refdata import compare_refstate_with_reference

# The methods to test
testcases = [case for case in cache.hfimport.keys()]
backends = [b for b in adcc.backends.available() if b != "molsturm"]


@expand_test_templates(list(itertools.product(testcases, backends)))
class TestBackendsImportReferenceData(unittest.TestCase):
    def base_test(self, system, backend, case):
        data = cache.hfdata[system]
        reference = cache.hfimport[system][case]

        if not adcc.backends.have_backend(backend):
            pytest.skip("Backend {} not available.".format(backend))

        basis = system.split("_")[-1]
        molecule = system.split("_")[0]

        if basis == "def2tzvp" and \
           backend == "veloxchem" and molecule == "h2o":
            pytest.skip("VeloxChem does not support f-functions.")

        multiplicity = 1
        compare_eri = "abs"
        if molecule in ["cn", "ch2nh2"]:
            multiplicity = 2
        if molecule == "cn":
            compare_eri = "off"

        if backend == "veloxchem" and multiplicity != 1:
            pytest.skip("VeloxChem does not support UHF.")

        scfres = cached_backend_hf(backend, molecule, basis, multiplicity)
        compare_refstate_with_reference(data, reference, case, scfres,
                                        compare_orbcoeff=False,
                                        compare_eri=compare_eri)

    def template_generic(self, case, backend):
        self.base_test(case, backend, "gen")

    def template_cvs(self, case, backend):
        self.base_test(case, backend, "cvs")

    # TODO Could test FC / FV and other imports as well
