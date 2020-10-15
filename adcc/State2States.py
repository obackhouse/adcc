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
import numpy as np

from . import adc_pp
from .misc import cached_property
from .timings import timed_member_call

from .Excitation import mark_excitation_property
from .ElectronicTransition import ElectronicTransition


class State2States(ElectronicTransition):
    """
    Documentation
    """
    def __init__(self, parent_state, initial=0, final=None):
        self.parent_state = parent_state
        self.reference_state = self.parent_state.reference_state
        self.ground_state = self.parent_state.ground_state
        self.matrix = self.parent_state.matrix
        self.property_method = self.parent_state.property_method
        self.operators = self.parent_state.operators

        self.initial = initial
        self.final = final

        self.initial_excitation_vector = \
            self.parent_state.excitation_vector[self.initial]
        if self.final is None:
            other_excitation_energy = np.delete(
                self.parent_state.excitation_energy.copy(), self.initial
            )
        else:
            other_excitation_energy = \
                np.array([self.parent_state.excitation_energy[self.final]])
        self.excitation_energy = other_excitation_energy -\
            self.parent_state.excitation_energy[self.initial]
    # TODO: describe?!

    @cached_property
    @mark_excitation_property(transform_to_ao=True)
    @timed_member_call(timer="_property_timer")
    def transition_dm(self):
        """
        List of transition density matrices from
        initial state to final state/s
        """
        # TODO: only states above self.initial
        if self.final is None:
            return [
                adc_pp.state2state_transition_dm(self.property_method,
                                                 self.ground_state,
                                                 self.initial_excitation_vector,
                                                 evec,
                                                 self.matrix.intermediates)
                for i, evec in enumerate(self.parent_state.excitation_vector)
                if i != self.initial
            ]
        else:
            final_vec = self.parent_state.excitation_vector[self.final]
            return [
                adc_pp.state2state_transition_dm(self.property_method,
                                                 self.ground_state,
                                                 self.initial_excitation_vector,
                                                 final_vec,
                                                 self.matrix.intermediates)
            ]


class State2StatesLazy(ElectronicTransition):
    """
    Documentation
    """
    def __init__(self, parent_state):
        self.parent_state = parent_state
        self.reference_state = self.parent_state.reference_state
        self.ground_state = self.parent_state.ground_state
        self.matrix = self.parent_state.matrix
        self.property_method = self.parent_state.property_method
        self.operators = self.parent_state.operators
    
    def __getitem__(self, *args):
        assert len(args) <= 2
        