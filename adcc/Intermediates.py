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
from adcc.functions import evaluate


class Intermediates(dict):
    """
    TODO docme
    """
    # Make caching configurable
    # Timings?

    def push(self, key, expression):
        if not self.__contains__(key):
            self.__setitem__(key, evaluate(expression))
        return self.__getitem__(key)

    def __getattr__(self, key):
        if self.__contains__(key):
            return self.__getitem__(key)
        else:
            raise AttributeError
