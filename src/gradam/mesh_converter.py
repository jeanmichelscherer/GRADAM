#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Last edited 2021-02-10

Anisotropic polycrystal multiphase field: .msh to .xdmf mesh conversion with meshio,
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

import os
import meshio
from dolfin import *
import numpy as np

class MeshioMsh2Xdmf:
    """ Convert a mesh in GMSH file format .msh to a FEniCS readable .xdmf mesh by using meshio """
    def __init__(self,dimension,msh_file,extras=''):
        self.dimension = dimension
        self.msh_file = msh_file
        if (MPI.rank(MPI.comm_world)==0):
            self.msh_mesh = meshio.read(self.msh_file+".msh")
        else:
            self.msh_mesh = None
        self.msh_mesh = MPI.comm_world.bcast(self.msh_mesh, root=0)
        self.extras = extras
        #self.write_xdmf_mesh()
        #self.read_xdmf_mesh()
        
    def make_meshio_mesh(self,cell_type,point_data={},prune_z=False):
        cells = np.vstack([cell.data for cell in self.msh_mesh.cells if cell.type==cell_type])
        if "gmsh:physical" in self.msh_mesh.cell_data_dict.keys():
            cell_data = np.hstack([self.msh_mesh.cell_data_dict["gmsh:physical"][key]
                                   for key in self.msh_mesh.cell_data_dict["gmsh:physical"].keys() if key==cell_type])
        # Remove z-coordinates from mesh if we have a 2D cell and all points have the same third coordinate
        points= self.msh_mesh.points
        if prune_z:
            points = points[:,:2]
        if "gmsh:physical" in self.msh_mesh.cell_data_dict.keys():
            self.meshio_mesh = meshio.Mesh(points=points, cells={cell_type: cells}, point_data=point_data, cell_data={"grains":[cell_data]})
        else:
            self.meshio_mesh = meshio.Mesh(points=points, cells={cell_type: cells}, point_data=point_data)
            
    def write_xdmf_mesh(self,out='', point_data={}):
        if (out==''):
            out = self.msh_file
        if (self.dimension == 2):
            prune_z=True
            if ('quad' in self.extras):
                elttype = 'quad'
            else:
                elttype = 'triangle'
        elif (self.dimension == 3):
            prune_z=False
            if ('hex' in self.extras):
                elttype = 'hexahedron'
            else:
                elttype = 'tetra'
        self.make_meshio_mesh(elttype, point_data=point_data, prune_z=prune_z)
        if (MPI.rank(MPI.comm_world)==0):
            meshio.write("%s.xdmf" % out, self.meshio_mesh)
        #meshio.xdmf.write("%s.xdmf" % out, self.meshio_mesh)
        
    def read_xdmf_mesh(self):
        xdmf_file = XDMFFile(MPI.comm_world, "%s.xdmf" % self.msh_file)
        self.mesh = Mesh()
        xdmf_file.read(self.mesh);
        self.mesh_dim = self.mesh.geometric_dimension()
        
        # make MeshFunction for grains subdomains
        self.mvc = MeshValueCollection("size_t", self.mesh, self.mesh_dim) 
        if "gmsh:physical" in self.msh_mesh.cell_data_dict.keys():
            xdmf_file.read(self.mvc, "grains")
        self.mf = cpp.mesh.MeshFunctionSizet(self.mesh, self.mvc)
        #self.dx = Measure("dx", domain=self.mesh, subdomain_data=self.mf)
        self.dx = Measure("dx", subdomain_data=self.mf)
        
        # exterior facets MeshFunction
        self.facets = MeshFunction("size_t", self.mesh, self.mesh_dim-1)
        self.facets.set_all(0)


