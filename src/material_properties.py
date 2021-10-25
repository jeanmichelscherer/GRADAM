#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Last edited 2021-03-02

Anisotropic polycrystal multiphase field: material properties per domain,
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

from dolfin import *
import numpy as np
#from .ELtensor import *
#import matscipy

def make_cubic_elasticity_stiffness_tensor(dim,
                                           moduli, #=[[E1,nu1,G1], [E2,nu2,G2], ..., [En,nun,Gn]]
                                           mesh,
                                           mf):
          
    # interpolate elasticity moduli
    class YOUNG(UserExpression):
        def __init__(self, mf, **kwargs):
            super().__init__(**kwargs)
            self.mf = mf
        def eval_cell(self, values, x, cell):
            k = self.mf[cell.index]-1
            values[0] = moduli[k][0]
        def value_shape(self):
            return () 
    class POISSON(UserExpression):
        def __init__(self, mf, **kwargs):
            super().__init__(**kwargs)
            self.mf = mf
        def eval_cell(self, values, x, cell):
            k = self.mf[cell.index]-1
            values[0] = moduli[k][1]
        def value_shape(self):
            return () 
    class SHEAR(UserExpression):
        def __init__(self, mf, **kwargs):
            super().__init__(**kwargs)
            self.mf = mf
        def eval_cell(self, values, x, cell):
            k = self.mf[cell.index]-1
            values[0] = moduli[k][2]
        def value_shape(self):
            return () 
            
    V0 = FunctionSpace(mesh, "DG", 0)
    E_, nu_, G_ = YOUNG(mf), POISSON(mf), SHEAR(mf)
    E, nu, G = Function(V0, name='E'), Function(V0, name='nu'), Function(V0, name='G')
    E.interpolate(E_)
    nu.interpolate(nu_)
    G.interpolate(G_)
    y1111 = E*(1.-nu**2)/(1.-3.*nu**2-2.*nu**3)
    y1122 = E*nu*(1.+nu)/(1.-3.*nu**2-2.*nu**3)
    y1212 = G 
    C = as_matrix( [[y1111, y1122, y1122, 0.,    0.,    0.   ],
                    [y1122, y1111, y1122, 0.,    0.,    0.   ],
                    [y1122, y1122, y1111, 0.,    0.,    0.   ],
                    [0.,    0.,    0.,    y1212, 0.,    0.,  ],
                    [0.,    0.,    0.,    0.,    y1212, 0.,  ],
                    [0.,    0.,    0.,    0.,    0.,    y1212]])
    return C, y1111, y1122, y1212   

def make_orthotropic_elasticity_stiffness_tensor(dim,
                                                 moduli, #=[[E1,E2,E3,nu12,nu21,nu13,nu31,nu23,nu32,G12,G13,G23], ...]
                                                 mesh,
                                                 mf):
          
    # interpolate elasticity moduli
    class YOUNG1(UserExpression):
        def __init__(self, mf, **kwargs):
            super().__init__(**kwargs)
            self.mf = mf
        def eval_cell(self, values, x, cell):
            k = self.mf[cell.index]-1
            values[0] = moduli[k][0]
        def value_shape(self):
            return () 
    class YOUNG2(UserExpression):
        def __init__(self, mf, **kwargs):
            super().__init__(**kwargs)
            self.mf = mf
        def eval_cell(self, values, x, cell):
            k = self.mf[cell.index]-1
            values[0] = moduli[k][1]
        def value_shape(self):
            return ()
    class YOUNG3(UserExpression):
        def __init__(self, mf, **kwargs):
            super().__init__(**kwargs)
            self.mf = mf
        def eval_cell(self, values, x, cell):
            k = self.mf[cell.index]-1
            values[0] = moduli[k][2]
        def value_shape(self):
            return ()
    class POISSON12(UserExpression):
        def __init__(self, mf, **kwargs):
            super().__init__(**kwargs)
            self.mf = mf
        def eval_cell(self, values, x, cell):
            k = self.mf[cell.index]-1
            values[0] = moduli[k][3]
        def value_shape(self):
            return () 
    class POISSON21(UserExpression):
        def __init__(self, mf, **kwargs):
            super().__init__(**kwargs)
            self.mf = mf
        def eval_cell(self, values, x, cell):
            k = self.mf[cell.index]-1
            values[0] = moduli[k][4]
        def value_shape(self):
            return () 
    class POISSON13(UserExpression):
        def __init__(self, mf, **kwargs):
            super().__init__(**kwargs)
            self.mf = mf
        def eval_cell(self, values, x, cell):
            k = self.mf[cell.index]-1
            values[0] = moduli[k][5]
        def value_shape(self):
            return () 
    class POISSON31(UserExpression):
        def __init__(self, mf, **kwargs):
            super().__init__(**kwargs)
            self.mf = mf
        def eval_cell(self, values, x, cell):
            k = self.mf[cell.index]-1
            values[0] = moduli[k][6]
        def value_shape(self):
            return () 
    class POISSON23(UserExpression):
        def __init__(self, mf, **kwargs):
            super().__init__(**kwargs)
            self.mf = mf
        def eval_cell(self, values, x, cell):
            k = self.mf[cell.index]-1
            values[0] = moduli[k][7]
        def value_shape(self):
            return ()
    class POISSON32(UserExpression):
        def __init__(self, mf, **kwargs):
            super().__init__(**kwargs)
            self.mf = mf
        def eval_cell(self, values, x, cell):
            k = self.mf[cell.index]-1
            values[0] = moduli[k][8]
        def value_shape(self):
            return ()
    class SHEAR12(UserExpression):
        def __init__(self, mf, **kwargs):
            super().__init__(**kwargs)
            self.mf = mf
        def eval_cell(self, values, x, cell):
            k = self.mf[cell.index]-1
            values[0] = moduli[k][9]
        def value_shape(self):
            return ()
    class SHEAR13(UserExpression):
        def __init__(self, mf, **kwargs):
            super().__init__(**kwargs)
            self.mf = mf
        def eval_cell(self, values, x, cell):
            k = self.mf[cell.index]-1
            values[0] = moduli[k][10]
        def value_shape(self):
            return ()
    class SHEAR23(UserExpression):
        def __init__(self, mf, **kwargs):
            super().__init__(**kwargs)
            self.mf = mf
        def eval_cell(self, values, x, cell):
            k = self.mf[cell.index]-1
            values[0] = moduli[k][11]
        def value_shape(self):
            return () 
            
    V0 = FunctionSpace(mesh, "DG", 0)
    E1_,E2_,E3_ = YOUNG1(mf), YOUNG2(mf), YOUNG3(mf)
    nu12_,nu21_,nu13_,nu31_,nu23_,nu32_ = POISSON12(mf), POISSON21(mf),\
        POISSON13(mf), POISSON31(mf), POISSON23(mf), POISSON32(mf)
    G12_,G13_,G23_ = SHEAR12(mf), SHEAR13(mf), SHEAR23(mf)
    E1, E2, E3 = Function(V0, name='E1'), Function(V0, name='E2'), Function(V0, name='E3')
    nu12, nu21, nu13, nu31, nu23, nu32 = Function(V0, name='nu12'), Function(V0, name='nu21'),\
        Function(V0, name='nu13'), Function(V0, name='nu31'), Function(V0, name='nu23'), Function(V0, name='nu32')
    G12, G13, G23 = Function(V0, name='G12'), Function(V0, name='G13'), Function(V0, name='G23')
    E1.interpolate(E1_)
    E2.interpolate(E2_)
    E3.interpolate(E3_)
    nu12.interpolate(nu12_)
    nu21.interpolate(nu21_)
    nu13.interpolate(nu13_)
    nu31.interpolate(nu31_)
    nu23.interpolate(nu23_)
    nu32.interpolate(nu32_)
    G12.interpolate(G12_)
    G13.interpolate(G13_)
    G23.interpolate(G23_)
    delta = (1.-nu23*nu32-nu31*nu13-nu12*nu21-2.*nu23*nu31*nu12)/(E1*E2*E3)
    y1111 = (1.-nu23*nu32)/(delta*E2*E3)
    y1122 = (nu21+nu31*nu23)/(delta*E2*E3)
    y1133 = (nu31+nu21*nu32)/(delta*E2*E3)
    y2211 = (nu12+nu13*nu32)/(delta*E1*E3)
    y2222 = (1.-nu31*nu13)/(delta*E1*E3)
    y2233 = (nu32+nu31*nu12)/(delta*E1*E3)
    y3311 = (nu13+nu12*nu23)/(delta*E1*E2)
    y3322 = (nu23+nu21*nu13)/(delta*E1*E2)
    y3333 = (1.-nu12*nu21)/(delta*E1*E2)
    y2323 = G23
    y1313 = G13
    y1212 = G12
    C = as_matrix( [[y1111, y1122, y1133, 0.,    0.,    0.   ],
                    [y2211, y2222, y2233, 0.,    0.,    0.   ],
                    [y3311, y3322, y3333, 0.,    0.,    0.   ],
                    [0.,    0.,    0.,    y2323, 0.,    0.,  ],
                    [0.,    0.,    0.,    0.,    y1313, 0.,  ],
                    [0.,    0.,    0.,    0.,    0.,    y1212]])
    return C 

def make_fracture_properties_per_domain(dim,
                                        mesh,
                                        mf,
                                        damage_dim,
                                        Gc_,
                                        l0_,
                                        dub_):
          
    # interpolate fracture properties
    class GC(UserExpression):
        def __init__(self, mf, **kwargs):
            super().__init__(**kwargs)
            self.mf = mf
        def eval_cell(self, values, x, cell):
            k = self.mf[cell.index]-1
            for i in range(damage_dim):
                values[i] = Gc_[k][i]
        def value_shape(self):
            if (damage_dim==1):
                return ()
            else:
                return (damage_dim,) 
    class L0(UserExpression):
        def __init__(self, mf, **kwargs):
            super().__init__(**kwargs)
            self.mf = mf
        def eval_cell(self, values, x, cell):
            k = self.mf[cell.index]-1
            for i in range(damage_dim):
                values[i] = l0_[k][i]
        def value_shape(self):
            if (damage_dim==1):
                return ()
            else:
                return (damage_dim,)
    # damage upper bound (=1 if damageable else =0)
    class DUB(UserExpression): 
        def __init__(self, mf, **kwargs):
            super().__init__(**kwargs)
            self.mf = mf
        def eval_cell(self, values, x, cell):
            k = self.mf[cell.index]-1
            for i in range(damage_dim):
                values[i] = dub_[k][i]
        def value_shape(self):
            if (damage_dim==1):
                return ()
            else:
                return (damage_dim,)
    
    
    if (damage_dim == 1):        
        V0 = FunctionSpace(mesh, "DG", 1)
        Vd = FunctionSpace(mesh, "CG", 1)
    else:
        V0 = VectorFunctionSpace(mesh, "DG", damage_dim)
        #Vd = VectorFunctionSpace(mesh, "CG", damage_dim)
        Vd = VectorFunctionSpace(mesh, "CG",1, dim=damage_dim)
    GC_, L0_, DUB_ = GC(mf), L0(mf), DUB(mf)
    Gc, l0, dub = Function(V0, name='Gc'), Function(V0, name='l0'), Function(Vd, name='Damage upper bound')
    Gc.interpolate(GC_)
    l0.interpolate(L0_)
    dub.interpolate(DUB_)

    return Gc, l0, dub


def transfer_function(function, function_space):
    temp = Function(function_space, name=function.name())
    # function.set_allow_extrapolation(True)
    A = PETScDMCollection.create_transfer_matrix(function.ufl_function_space(),
                                                 function_space)
    temp.vector()[:] = A*function.vector()
    return temp