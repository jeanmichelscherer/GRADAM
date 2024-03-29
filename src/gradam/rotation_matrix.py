#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Last edited 2021-02-10

Anisotropic polycrystal multiphase field: useful functions for rotation matrices,
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
import os
from scipy.spatial.transform import Rotation

def make_rotation_matrix_from_euler(dim,
                         neper_mesh,
                         mesh,
                         mf,
                         orientation): 
    
    # get orientations generated by neper with bunge-euler convention
    if os.path.isfile(neper_mesh+".ori"):
        print(".ori file found!")
        ori = np.loadtxt(neper_mesh+".ori")
    else:
        print(".ori file not found! using phi1=0, Phi=0, phi2=0")
        ori = np.array([0., 0., 0.])
    if orientation=='from_euler':
        if (ori.size == 3):
            ori = [ori, ori] # duplicate single crystal orientation in order to get a 2D list as well
    elif orientation=='from_vector':
        if (ori.size == 6):
            ori = [ori, ori] # duplicate single crystal orientation in order to get a 2D list as well
        angles = []
        for o in ori:
            normv1 = sqrt( o[0]**2 + o[1]** 2 + o[2]**2)
            normv2 = sqrt( o[3]**2 + o[4]** 2 + o[5]**2)
            normv3 = sqrt( o[0]**2 + o[1]** 2 + o[2]**2) * \
                     sqrt( o[3]**2 + o[4]** 2 + o[5]**2)
            v11 = o[0] / normv1
            v12 = o[1] / normv1
            v13 = o[2] / normv1
            v21 = o[3] / normv2
            v22 = o[4] / normv2
            v23 = o[5] / normv2
            v31 = (o[1]*o[5] - o[2]*o[4]) / normv3
            v32 = (o[3]*o[2] - o[0]*o[5]) / normv3
            v33 = (o[0]*o[4] - o[1]*o[3]) / normv3
            R_ = [[v11, v21, v31], 
                  [v12, v22, v32], 
                  [v13, v23, v33]]
            r = Rotation.from_matrix(np.transpose(R_)) # I take the transpose because of the way "as_euler" function works
            phi = r.as_euler('ZXZ', degrees=True)
            angles.append(phi)
            #print(R_)
        ori = angles
        #print(ori)
        
    # compute crystal rotation (sample to crystal, i.e. V_crystal = Rotation*V_sample)
    class PHI1(UserExpression):
        def __init__(self, mf, **kwargs):
            super().__init__(**kwargs)
            self.mf = mf
        def eval_cell(self, values, x, cell):
            k = self.mf[cell.index]-1
            values[0] = pi*ori[k][0]/180.
        def value_shape(self):
            return () 
    class PHI(UserExpression):
        def __init__(self, mf, **kwargs):
            super().__init__(**kwargs)
            self.mf = mf
        def eval_cell(self, values, x, cell):
            k = self.mf[cell.index]-1
            values[0] = pi*ori[k][1]/180.
        def value_shape(self):
            return () 
    class PHI2(UserExpression):
        def __init__(self, mf, **kwargs):
            super().__init__(**kwargs)
            self.mf = mf
        def eval_cell(self, values, x, cell):
            k = self.mf[cell.index]-1
            values[0] = pi*ori[k][2]/180.
        def value_shape(self):
            return () 
            
    V0 = FunctionSpace(mesh, "DG", 0)
    phi1_, Phi_, phi2_ = PHI1(mf), PHI(mf), PHI2(mf)
    phi1, Phi, phi2 = Function(V0, name='phi1'), Function(V0, name='Phi'), Function(V0, name='phi2')
    phi1.interpolate(phi1_)
    Phi.interpolate(Phi_)
    phi2.interpolate(phi2_)
    # phi1.assign(local_project(phi1, V0))
    # Phi.assign(local_project(Phi, V0))
    # phi2.assign(local_project(phi2, V0))
    
    # rotation matrix (sample to crystal, i.e. V_crystal = R*V_sample)
    if (dim==2 and orientation=='from_euler'):
        R = as_matrix([[ cos(phi1),  sin(phi1), 0.],
                       [-sin(phi1),  cos(phi1), 0.], 
                       [        0.,         0., 1.]])
    else:
        R = as_matrix([[cos(phi1)*cos(phi2)-sin(phi1)*sin(phi2)*cos(Phi), 
                        sin(phi1)*cos(phi2)+cos(phi1)*sin(phi2)*cos(Phi),
                        sin(phi2)*sin(Phi)],
                       [-cos(phi1)*sin(phi2)-sin(phi1)*cos(phi2)*cos(Phi),
                        -sin(phi1)*sin(phi2)+cos(phi1)*cos(phi2)*cos(Phi),
                        cos(phi2)*sin(Phi)],
                       [sin(phi1)*sin(Phi),
                        -cos(phi1)*sin(Phi),
                        cos(Phi)]])

    return R, phi1, Phi, phi2   

def make_rotation_matrix_from_V1V2V3(dim,
                         neper_mesh,
                         mesh,
                         mf):
    
    # get coordinates of sample axes in crystal frame (e.g. X1 = [111])
    ori = np.loadtxt(neper_mesh+".ori")
    if (ori.size == 6):
        ori = [ori, ori] # duplicate single crystal orientation in order to get a 2D list as well
    
    # compute crystal rotation (sample to crystal, i.e. V_crystal = Rotation*V_sample)
    class V1(UserExpression):
        def __init__(self, mf, **kwargs):
            super().__init__(**kwargs)
            self.mf = mf
        def eval_cell(self, values, x, cell):
            k = self.mf[cell.index]-1
            norm = sqrt( ori[k][0]**2 + ori[k][1]** 2 + ori[k][2]**2)
            values[0] = ori[k][0] / norm
            values[1] = ori[k][1] / norm
            values[2] = ori[k][2] / norm
        def value_shape(self):
            return (3,) 
    class V2(UserExpression):
        def __init__(self, mf, **kwargs):
            super().__init__(**kwargs)
            self.mf = mf
        def eval_cell(self, values, x, cell):
            k = self.mf[cell.index]-1
            norm = sqrt( ori[k][3]**2 + ori[k][4]** 2 + ori[k][5]**2)
            values[0] = ori[k][3] / norm
            values[1] = ori[k][4] / norm
            values[2] = ori[k][5] / norm
        def value_shape(self):
            return (3,)
    class V3(UserExpression):
        def __init__(self, mf, **kwargs):
            super().__init__(**kwargs)
            self.mf = mf
        def eval_cell(self, values, x, cell):
            k = self.mf[cell.index]-1
            norm = sqrt( ori[k][0]**2 + ori[k][1]** 2 + ori[k][2]**2) * \
                   sqrt( ori[k][3]**2 + ori[k][4]** 2 + ori[k][5]**2)
            values[0] = (ori[k][1]*ori[k][5] - ori[k][2]*ori[k][4]) / norm
            values[1] = (ori[k][3]*ori[k][2] - ori[k][0]*ori[k][5]) / norm
            values[2] = (ori[k][0]*ori[k][4] - ori[k][1]*ori[k][3]) / norm
        def value_shape(self):
            return (3,) 
            
    V0 = VectorFunctionSpace(mesh, "DG", 0, dim=3)   
    v1_, v2_, v3_ = V1(mf), V2(mf), V3(mf)
    v1, v2, v3 = Function(V0, name='v1'), Function(V0, name='v2'), Function(V0, name='v3')   
    v1.interpolate(v1_)
    v2.interpolate(v2_)
    v3.interpolate(v3_)
    # v1.assign(local_project(v1, V0))
    # v2.assign(local_project(v2, V0))
    # v3.assign(local_project(v3, V0))
    
    # rotation matrix (sample to crystal, i.e. V_crystal = R*V_sample)
    R_ = [[v1[0], v2[0], v3[0]], 
          [v1[1], v2[1], v3[1]], 
          [v1[2], v2[2], v3[2]]]
    #r = Rotation.from_matrix(R_)
    #phi1, Phi, phi2 = r.as_euler('zxz', degrees=False)
    R = as_matrix(R_)
    
    return R, v1, v2, v3 
