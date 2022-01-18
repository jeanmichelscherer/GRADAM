#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Last edited 2021-02-10

Anisotropic polycrystal multiphase field: useful functions on tensors,
adapted from phase_field_composites by Jérémy Bleyer,
https://zenodo.org/record/1188970

This file is part of phase_field_polycrystals based on FEniCS project 
(https://fenicsproject.org/)

phase_field_polycrystals (c) by Jean-Michel Scherer, 
Ecole des Ponts ParisTech, 
Laboratoire Navier (ENPC,IFSTTAR,CNRS UMR 8205) & 
Ecole Polytechnique, 
Laboratoire de Mécanique des Solides, Institut Polytechnique

phase_field_polycrystals is licensed under a
Creative Commons Attribution-ShareAlike 4.0 International License.

You should have received a copy of the license along with this
work. If not, see <http://creativecommons.org/licenses/by-sa/4.0/>.
"""
_author__ = "Jean-Michel Scherer"
__license__ = "CC BY-SA 4.0"
__email__ = "jean-michel.scherer@enpc.fr"

import numpy as np
import itertools
from dolfin import *

# copied and adapted from libAtoms/matscipy/elasticity.py library: 
# https://github.com/libAtoms/matscipy/blob/master/matscipy/elasticity.py

# The indices of the full stiffness matrix of (orthorhombic) interest
Voigt_notation = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]

def full_3x3_to_Voigt_6_index(i, j):
    if i == j:
        return i
    return 6-i-j

def Voigt_6x6_to_full_3x3x3x3(C):
    """
    Convert from the Voigt representation of the stiffness matrix to the full
    3x3x3x3 representation.
    Parameters
    ----------
    C : array_like
        6x6 stiffness matrix (Voigt notation).
    
    Returns
    -------
    C : array_like
        3x3x3x3 stiffness matrix.
    """
    
    C = np.asarray(C)
    C_out = np.zeros((3,3,3,3), dtype=float)
    for i, j, k, l in itertools.product(range(3), range(3), range(3), range(3)):
        Voigt_i = full_3x3_to_Voigt_6_index(i, j)
        Voigt_j = full_3x3_to_Voigt_6_index(k, l)
        C_out[i, j, k, l] = C[Voigt_i, Voigt_j]
    return C_out

def full_3x3x3x3_to_Voigt_6x6(C):
    """
    Convert from the full 3x3x3x3 representation of the stiffness matrix
    to the representation in Voigt notation. Checks symmetry in that process.
    """

    tol = 1e-3

    C = np.asarray(C)
    Voigt = np.zeros((6,6))
    for i in range(6):
        for j in range(6):
            k, l = Voigt_notation[i]
            m, n = Voigt_notation[j]
            Voigt[i,j] = C[k,l,m,n]

            #print '---'
            #print k,l,m,n, C[k,l,m,n]
            #print m,n,k,l, C[m,n,k,l]
            #print l,k,m,n, C[l,k,m,n]
            #print k,l,n,m, C[k,l,n,m]
            #print m,n,l,k, C[m,n,l,k]
            #print n,m,k,l, C[n,m,k,l]
            #print l,k,n,m, C[l,k,n,m]
            #print n,m,l,k, C[n,m,l,k]
            #print '---'

            # Check symmetries
            # assert abs(Voigt[i,j]-C[m,n,k,l]) < tol, \
            #     'Voigt[{},{}] = {}, C[{},{},{},{}] = {}' \
            #     .format(i, j, Voigt[i,j], m, n, k, l, C[m,n,k,l])
            # assert abs(Voigt[i,j]-C[l,k,m,n]) < tol, \
            #     'Voigt[{},{}] = {}, C[{},{},{},{}] = {}' \
            #     .format(i, j, Voigt[i,j], k, l, m, n, C[l,k,m,n])
            # assert abs(Voigt[i,j]-C[k,l,n,m]) < tol, \
            #     'Voigt[{},{}] = {}, C[{},{},{},{}] = {}' \
            #     .format(i, j, Voigt[i,j], k, l, n, m, C[k,l,n,m])
            # assert abs(Voigt[i,j]-C[m,n,l,k]) < tol, \
            #     'Voigt[{},{}] = {}, C[{},{},{},{}] = {}' \
            #     .format(i, j, Voigt[i,j], m, n, l, k, C[m,n,l,k])
            # assert abs(Voigt[i,j]-C[n,m,k,l]) < tol, \
            #     'Voigt[{},{}] = {}, C[{},{},{},{}] = {}' \
            #     .format(i, j, Voigt[i,j], n, m, k, l, C[n,m,k,l])
            # assert abs(Voigt[i,j]-C[l,k,n,m]) < tol, \
            #     'Voigt[{},{}] = {}, C[{},{},{},{}] = {}' \
            #     .format(i, j, Voigt[i,j], l, k, n, m, C[l,k,n,m])
            # assert abs(Voigt[i,j]-C[n,m,l,k]) < tol, \
            #     'Voigt[{},{}] = {}, C[{},{},{},{}] = {}' \
            #     .format(i, j, Voigt[i,j], n, m, l, k, C[n,m,l,k])
    #print(Voigt)
    return Voigt

