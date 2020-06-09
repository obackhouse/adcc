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
from .LazyMp import LazyMp
from .timings import Timer
from .AdcMethod import AdcMethod
from .functions import evaluate
from .Intermediates import Intermediates
from .AmplitudeVector import AmplitudeVector

from adc_pp import matrix as ppmatrix

import libadcc

# TODO This is a bit of a hack to be able to use the python expressions
#      from AdcMatrix.py


class AdcMatrixPython:
    def __init__(self, method, hf_or_mp, block_orders=None):
        if isinstance(hf_or_mp, (libadcc.ReferenceState,
                                 libadcc.HartreeFockSolution_i)):
            hf_or_mp = LazyMp(hf_or_mp)
        if not isinstance(hf_or_mp, libadcc.LazyMp):
            raise TypeError("mp_results is not a valid object. It needs to be "
                            "either a LazyMp, a ReferenceState or a "
                            "HartreeFockSolution_i.")

        if not isinstance(method, AdcMethod):
            method = AdcMethod(method)

        self.method = method
        self.ground_state = hf_or_mp
        self.reference_state = hf_or_mp.reference_state
        self.mospaces = hf_or_mp.reference_state.mospaces
        self.is_core_valence_separated = method.is_core_valence_separated

        # Determine block orders
        new_orders = {
            #             ph_ph=0, ph_pphh=None, pphh_ph=None, pphh_pphh=None),
            "adc0":  dict(ph_ph=0, ph_pphh=None, pphh_ph=None, pphh_pphh=None),  # noqa: E501
            "adc1":  dict(ph_ph=1, ph_pphh=None, pphh_ph=None, pphh_pphh=None),  # noqa: E501
            "adc2":  dict(ph_ph=2, ph_pphh=1,    pphh_ph=1,    pphh_pphh=0),     # noqa: E501
            "adc2x": dict(ph_ph=2, ph_pphh=1,    pphh_ph=1,    pphh_pphh=1),     # noqa: E501
            "adc3":  dict(ph_ph=3, ph_pphh=2,    pphh_ph=2,    pphh_pphh=1),     # noqa: E501
        }[method.base_method]
        if block_orders is not None:
            new_orders.update(block_orders)
        block_orders = new_orders

        # Sanity checks on block_orders
        for block in block_orders.keys():
            if block not in ("ph_ph", "ph_pphh", "pphh_ph", "pphh_pphh"):
                raise ValueError(f"Invalid block order key: {block}")
        if block_orders["ph_pphh"] != block_orders["pphh_ph"]:
            raise ValueError("ph_pphh and pphh_ph should always have "
                             "the same order")
        if block_orders["ph_pphh"] is not None \
           and block_orders["pphh_pphh"] is None:
            raise ValueError("pphh_pphh cannot be node if ph_pphh isn't.")

        # Build the blocks
        variant = None
        if method.is_core_valence_separated:
            variant = "cvs"
        self.intermediates = Intermediates()

        self.__blocks = {
            block: ppmatrix.block(hf_or_mp, block.split("_"), order=order,
                                  intermediates=self.intermediates,
                                  variant=variant)
            for block, order in block_orders.items() if order is not None
        }

        # TODO Remove this
        # automatically get the shape and subspace
        # blocks of the matrix from the orders
        # TODO once the AmplitudeVector object support missing blocks
        #      this would be self.diagonal.blocks and self.diagonal.ph.subspaces
        #      and self.diagonal.pphh.subspaces
        self.__block_spaces = {}
        if variant == "cvs":
            self.__block_spaces["s"] = ["o2", "v1"]
            if block_orders.get("pphh_pphh", None) is not None:
                self.__block_spaces["d"] = ["o1", "o2", "v1", "v1"]
        else:
            self.__block_spaces["s"] = ["o1", "v1"]
            if block_orders.get("pphh_pphh", None) is not None:
                self.__block_spaces["d"] = ["o1", "o1", "v1", "v1"]
        self.blocks = list(self.__block_spaces.keys())

    @property
    def timer(self):
        return Timer()  # TODO Implement properly

    def diagonal(self, block):
        if block not in self.blocks:
            raise ValueError("block not in blocks")
        # TODO The block argument should be brought more
        #      in line with the ph, pphh stuff

        # TODO Once the AmplitudeVector object support missing blocks this would
        #      just be return sum(bl.diagonal for bl in self.__blocks)

        if block == "s":
            return evaluate(sum(bl.diagonal.ph
                                for bl in self.__blocks.values()
                                if bl.diagonal and "s" in bl.diagonal.blocks
                                and bl.diagonal.ph))
        else:
            return evaluate(sum(bl.diagonal.pphh
                                for bl in self.__blocks.values()
                                if bl.diagonal and "d" in bl.diagonal.blocks
                                and bl.diagonal.pphh))

    def to_cpp(self):
        # TODO Super-dirty hack, since guesses are not yet py-side
        self.cppmat = libadcc.AdcMatrix(self.method.name, self.ground_state)
        for key, value in self.intermediates.items():
            getattr(self.cppmat.intermediates, f"set_{key}")(value)
        return self.cppmat

    def compute_apply(self, block, tensor):
        # TODO A lot of junk code to get compatibility to the old interface
        key = {"ss": "ph_ph", "sd": "ph_pphh",
               "ds": "pphh_ph", "dd": "pphh_pphh"}[block]
        if block[1] == "s":
            ampl = AmplitudeVector(ph=tensor)
        else:
            ampl = AmplitudeVector(pphh=tensor)
        ret = self.__blocks[key].apply(ampl)
        if block[0] == "s":
            return ret.ph
        else:
            return ret.pphh

    def compute_matvec(self, ampl):
        # TODO Once properly supported in AmplitudeVector, this should be
        # return sum(bl.apply(ampl) for bl in self.__blocks.values())

        res = [bl.apply(ampl) for bl in self.__blocks.values()]
        ph = sum(v.ph for v in res if "s" in v.blocks and v.ph)
        pphh = None
        if "d" in self.blocks:
            pphh = sum(v.pphh for v in res if "d" in v.blocks and v.pphh)
        return AmplitudeVector(ph=ph, pphh=pphh)

    def has_block(self, block):  # TODO This should be deprecated
        return block in self.blocks

    @property
    def shape(self):
        length = 0
        for bl in self.blocks:
            prod = 1
            for sp in self.block_spaces(bl):
                prod *= self.mospaces.n_orbs(sp)
            length += prod
        return (length, length)

    def block_spaces(self, block):  # TODO This should be deprecated
        return self.__block_spaces[block]
