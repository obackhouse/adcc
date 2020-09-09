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
import warnings
import numpy as np

from . import adc_pp
from .misc import cached_property
from .timings import Timer, timed_member_call
from .AdcMethod import AdcMethod
from .FormatIndex import (FormatIndexAdcc, FormatIndexBase,
                          FormatIndexHfProvider, FormatIndexHomoLumo)
from .visualisation import ExcitationSpectrum
from .OneParticleOperator import product_trace
from .FormatDominantElements import FormatDominantElements

from adcc import dot

from scipy import constants
from matplotlib import pyplot as plt
from .solver.SolverStateBase import EigenSolverStateBase
from .Excitation import Excitation, mark_excitation_property

class ElectronicTransition:
    """
    Documentation
    """
    @cached_property
    @mark_excitation_property()
    @timed_member_call(timer="_property_timer")
    def transition_dipole_moment(self):
        """List of transition dipole moments of all computed states"""
        if self.property_method.level == 0:
            warnings.warn("ADC(0) transition dipole moments are known to be "
                          "faulty in some cases.")
        dipole_integrals = self.operators.electric_dipole
        return np.array([
            [product_trace(comp, tdm) for comp in dipole_integrals]
            for tdm in self.transition_dm
        ])

    @cached_property
    @mark_excitation_property()
    @timed_member_call(timer="_property_timer")
    def transition_dipole_moment_velocity(self):
        """List of transition dipole moments in the
        velocity gauge of all computed states"""
        if self.property_method.level == 0:
            warnings.warn("ADC(0) transition velocity dipole moments "
                          "are known to be faulty in some cases.")
        dipole_integrals = self.operators.nabla
        return np.array([
            [product_trace(comp, tdm) for comp in dipole_integrals]
            for tdm in self.transition_dm
        ])

    @cached_property
    @mark_excitation_property()
    @timed_member_call(timer="_property_timer")
    def transition_magnetic_dipole_moment(self):
        """List of transition magnetic dipole moments of all computed states"""
        if self.property_method.level == 0:
            warnings.warn("ADC(0) transition magnetic dipole moments "
                          "are known to be faulty in some cases.")
        mag_dipole_integrals = self.operators.magnetic_dipole
        return np.array([
            [product_trace(comp, tdm) for comp in mag_dipole_integrals]
            for tdm in self.transition_dm
        ])

    @cached_property
    @mark_excitation_property()
    def oscillator_strength(self):
        """List of oscillator strengths of all computed states"""
        return 2. / 3. * np.array([
            np.linalg.norm(tdm)**2 * np.abs(ev)
            for tdm, ev in zip(self.transition_dipole_moment,
                               self.excitation_energy)
        ])

    @cached_property
    @mark_excitation_property()
    def oscillator_strength_velocity(self):
        """List of oscillator strengths in
        velocity gauge of all computed states"""
        return 2. / 3. * np.array([
            np.linalg.norm(tdm)**2 / np.abs(ev)
            for tdm, ev in zip(self.transition_dipole_moment_velocity,
                               self.excitation_energy)
        ])

    @cached_property
    @mark_excitation_property()
    def rotatory_strength(self):
        """List of rotatory strengths of all computed states"""
        return np.array([
            np.dot(tdm, magmom) / ee
            for tdm, magmom, ee in zip(self.transition_dipole_moment_velocity,
                                       self.transition_magnetic_dipole_moment,
                                       self.excitation_energy)
        ])

    @property
    @mark_excitation_property()
    def cross_section(self):
        """List of one-photon absorption cross sections of all computed states"""
        # TODO Source?
        fine_structure = constants.fine_structure
        fine_structure_au = 1 / fine_structure
        prefac = 2.0 * np.pi ** 2 / fine_structure_au
        return prefac * self.oscillator_strength

    def plot_spectrum(self, broadening="lorentzian", xaxis="eV",
                      yaxis="cross_section", width=0.01, **kwargs):
        """One-shot plotting function for the spectrum generated by all states
        known to this class.

        Makes use of the :class:`adcc.visualisation.ExcitationSpectrum` class
        in order to generate and format the spectrum to be plotted, using
        many sensible defaults.

        Parameters
        ----------
        broadening : str or None or callable, optional
            The broadening type to used for the computed excitations.
            A value of None disables broadening any other value is passed
            straight to
            :func:`adcc.visualisation.ExcitationSpectrum.broaden_lines`.
        xaxis : str
            Energy unit to be used on the x-Axis. Options:
            ["eV", "au", "nm", "cm-1"]
        yaxis : str
            Quantity to plot on the y-Axis. Options are "cross_section",
            "osc_strength", "dipole" (plots norm of transition dipole),
            "rotational_strength" (ECD spectrum with rotational strength)
        width : float, optional
            Gaussian broadening standard deviation or Lorentzian broadening
            gamma parameter. The value should be given in atomic units
            and will be converted to the unit of the energy axis.
        """
        if xaxis == "eV":
            eV = constants.value("Hartree energy in eV")
            energies = self.excitation_energy * eV
            width = width * eV
            xlabel = "Energy (eV)"
        elif xaxis in ["au", "Hartree", "a.u."]:
            energies = self.excitation_energy
            xlabel = "Energy (au)"
        elif xaxis == "nm":
            hc = constants.h * constants.c
            Eh = constants.value("Hartree energy")
            energies = hc / (self.excitation_energy * Eh) * 1e9
            xlabel = "Wavelength (nm)"
            if broadening is not None and not callable(broadening):
                raise ValueError("xaxis=nm and broadening enabled is "
                                 "not supported.")
        elif xaxis in ["cm-1", "cm^-1", "cm^{-1}"]:
            towvn = constants.value("hartree-inverse meter relationship") / 100
            energies = self.excitation_energy * towvn
            width = width * towvn
            xlabel = "Wavenumbers (cm^{-1})"
        else:
            raise ValueError("Unknown xaxis specifier: {}".format(xaxis))

        if yaxis in ["osc", "osc_strength", "oscillator_strength", "f"]:
            absorption = self.oscillator_strength
            ylabel = "Oscillator strengths (au)"
        elif yaxis in ["dipole", "dipole_norm", "μ"]:
            absorption = np.linalg.norm(self.transition_dipole_moment, axis=1)
            ylabel = "Modulus of transition dipole (au)"
        elif yaxis in ["cross_section", "σ"]:
            absorption = self.cross_section
            ylabel = "Cross section (au)"
        elif yaxis in ["rot", "rotational_strength", "rotatory_strength"]:
            absorption = self.rotatory_strength
            ylabel = "Rotatory strength (au)"
        else:
            raise ValueError("Unknown yaxis specifier: {}".format(yaxis))

        sp = ExcitationSpectrum(energies, absorption)
        sp.xlabel = xlabel
        sp.ylabel = ylabel
        if not broadening:
            plots = sp.plot(style="discrete", **kwargs)
        else:
            kwdisc = kwargs.copy()
            kwdisc.pop("label", "")
            plots = sp.plot(style="discrete", **kwdisc)

            kwargs.pop("color", "")
            sp_broad = sp.broaden_lines(width, shape=broadening)
            plots.extend(sp_broad.plot(color=plots[0].get_color(),
                                       style="continuous", **kwargs))

        if xaxis in ["nm"]:
            # Invert x axis
            plt.xlim(plt.xlim()[::-1])
        return plots


class State2StateTransition(ElectronicTransition):
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

        self.initial_excitation_vector = self.parent_state.excitation_vector[self.initial]
        if self.final is None:
            other_excitation_energy = np.delete(
                self.parent_state.excitation_energy.copy(), self.initial
            )
        else:
            other_excitation_energy = np.array([self.parent_state.excitation_energy[self.final]])
        self.excitation_energy = other_excitation_energy -\
            self.parent_state.excitation_energy[self.initial]
    # TODO: describe?!

    @cached_property
    @mark_excitation_property(transform_to_ao=True)
    @timed_member_call(timer="_property_timer")
    def transition_dm(self):
        """List of transition density matrices from initial state to final state/s"""
        if self.final is None:
            return [adc_pp.state2state_transition_dm(self.property_method, self.ground_state,
                                                    self.initial_excitation_vector, evec,
                                                    self.matrix.intermediates)
                    for i, evec in enumerate(self.parent_state.excitation_vector) if i != self.initial]
        else:
            return [adc_pp.state2state_transition_dm(self.property_method, self.ground_state,
                                                    self.initial_excitation_vector,
                                                    self.parent_state.excitation_vector[self.final],
                                                    self.matrix.intermediates)]
