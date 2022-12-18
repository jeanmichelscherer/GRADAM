"""
Created on Mon Feb 14 09:23:17 2022

Anisotropic polycrystal multiphase field: Mfront material models,
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

import meshio
from dolfin import *
from .material_properties import *
from .rotation_matrix import *
from .tensors import *
import numpy as np
from ufl import sign
import ufl
import mgis.fenics as mf

class MfrontBehaviour:
    """ Non-linear material behaviour based on Mfront implementations and the MGIS.fenics interface """ 
    def __init__(self,behaviour,mfront_library='./src/libBehaviour.so',hypothesis='3d',mp={},quadrature_degree=2):
        self.behaviour = behaviour
        self.mfront_library = mfront_library
        self.hypothesis = hypothesis
        self.mat_prop = mp
        self.quadrature_degree = quadrature_degree
        
    def set_material_properties(self,dim,mesh,mf,mat_prop):
        for key in self.mat_prop.keys():
            if (isinstance(self.mat_prop[key],list)):
                self.mat_prop[key] = make_property_per_domain(dim,mesh,mf,key,self.mat_prop[key])
            else:
                self.mat_prop[key] = self.mat_prop[key] #["function"]
                #self.mat_prop[key] = make_evolving_property_per_domain(dim,mesh,mf,self.mat_prop[key]["function"],key,self.mat_prop[key])
        self.mat_prop = dict(self.mat_prop, **mat_prop) # concatenation of both dicts
        
    def set_rotation_matrix(self,R):
        if ("Isotropic" in self.behaviour):
            self.rotation_matrix=None
        else:
            self.rotation_matrix = R
    
    def create_material(self):
        material = mf.MFrontNonlinearMaterial(self.mfront_library,
                                              self.behaviour,
                                              hypothesis=self.hypothesis,
                                              material_properties=self.mat_prop,
                                              rotation_matrix=self.rotation_matrix)
        return material