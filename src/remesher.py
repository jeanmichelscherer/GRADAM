#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Last edited 2021-03-27

Anisotropic polycrystal multiphase field: class computing metric and writing .sol files for MMG remeshing,
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
from dolfin import *
import numpy as np
import meshio
from mpi4py import MPI as pyMPI
import matplotlib.pyplot as plt
# import subprocess

class Remesher:
    def __init__(self,mesh_path='',mesh_file='',sol_file='',alpha=None,delta=None,number_of_nodes_index=None,\
                 sol_min=None,sol_max=None,save_files=False):
        self.mesh_path = mesh_path
        self.mesh_file = mesh_file
        self.sol_file = sol_file
        self.alpha = alpha
        self.delta = delta
        self.number_of_nodes_index = number_of_nodes_index # index of the nodes number in msh files (depends on msh version)
        self.sol_min = sol_min
        self.sol_max = sol_max
        self.save_files = save_files

    def diffusion(self,v,V):
        dv = TrialFunction(V)
        v_ = TestFunction(V)
        a = dv*v_*dx + self.delta**2*dot(grad(dv), grad(v_))*dx
        L = v*v_*dx #inner(v, v_)*dx
        u = Function(V)
        solve(a == L, u)
        return u
    
    def metric(self,damage_dim,d,Vd,remeshing_index):       
        VV = Vd
        if (damage_dim>1):
             VV = Vd.sub(0).collapse()
        diffuse = Function(VV,name="Diffused damage")
        metric_field = Function(VV,name="v:metric")
        metric_field.vector()[:] = diffuse.vector()[:]
        if (damage_dim>1):
            for k in range(damage_dim):
                diffuse = self.diffusion(d.sub(k),VV)
                metric_field.vector()[:] = np.maximum(metric_field.vector()[:], diffuse.vector()[:]) #element wise maximum #or # += diffuse.compute_vertex_values() #sum
        else: 
            diffuse = self.diffusion(d,VV)
            metric_field.vector()[:] = np.maximum(metric_field.vector()[:], diffuse.vector()[:])
        mini, maxi = min(metric_field.vector()), max(metric_field.vector())
        pyMPI.COMM_WORLD.barrier()
        mini, maxi = pyMPI.COMM_WORLD.allreduce(mini, op=pyMPI.MIN), pyMPI.COMM_WORLD.allreduce(maxi, op=pyMPI.MAX)
        metric_field.vector()[:] = (metric_field.vector()[:] - mini)/(max(maxi - mini,1.e-6))
        xdmf = XDMFFile(pyMPI.COMM_WORLD, self.mesh_path+"metric_%s.xdmf" % remeshing_index)
        xdmf.write(metric_field)
        xdmf.close()
        
    def write_uniform_sol(self,uniform_metric):
        s = open(self.mesh_path+self.mesh_file+'_remeshed_0.sol','w')
        f = open(self.mesh_path+self.mesh_file+'.msh', "r")
        lines = f.readlines()
        f.close()
        i = 0
        number_of_nodes = 0
        while (number_of_nodes==0):
            if lines[i].startswith("$Nodes"):
                number_of_nodes = int(lines[i+1].split()[self.number_of_nodes_index])
            i+=1  
        s.write(\
        'MeshVersionFormatted 2\n\\n\Dimension 3\n\\n\SolAtVertices\n\%s\n\1 1\n\\n'%number_of_nodes)
        for i in range(number_of_nodes):
            s.write('%s\n' % uniform_metric)   
        s.write('\nEND')    
        s.close()

    def write_sol(self,metric_field,number_of_nodes,remeshing_index):
        '''
        f = open(self.mesh_path+self.mesh_file+'_remeshed_%s.msh' % (remeshing_index-1), "r")
        lines = f.readlines()
        f.close()
        i = 0
        number_of_nodes = 0
        while (number_of_nodes==0):
            if lines[i].startswith("$Nodes"):
                number_of_nodes = int(lines[i+1].split()[self.number_of_nodes_index])
            i+=1
        '''
        s  = open(self.mesh_path+self.sol_file+'_remeshed_%s.sol' % (remeshing_index-1),'w')
        s.write('MeshVersionFormatted 2\n\nDimension 3\n\nSolAtVertices\n%s\n1 1\n\n'%number_of_nodes)
        for i in range(number_of_nodes):
            #new_sol = min( max( self.sol_max*(1. - 2.*metric_field[i]) , self.sol_min) , self.sol_max )
            #new_sol = min( max( self.sol_max*(1. - 5.*metric_field[i]) , self.sol_min) , self.sol_max )
            
            new_sol = max( self.sol_max*max((1. - self.alpha*metric_field[i]),0.)**0.5 , self.sol_min)
            s.write( '%s\n' % format(new_sol, '.4f') )   
        s.write('\nEND')   
        s.close()
    
    def remesh(self,dim,geo_tmpl,nbgrains,remeshing_index):
        if (remeshing_index==1):
            self.convert_msh2mesh(dim,remeshing_index-1)      
        oldmesh = self.mesh_file+"_remeshed_%s" % (remeshing_index-1)
        newmesh = self.mesh_file+"_remeshed_%s" % (remeshing_index) #+1)
        medit = ""
        if (dim==2):
            medit = "-3dMedit %s" % dim
        command = "mmg%sd_O3 %s %s" % (dim,medit,self.mesh_path+oldmesh+'.mesh')
        print("\nCalling MMG to perform remeshing: %s \n" % command )
        os.system(command)
        #subprocess.call(["mmg%sd_O3" % dim, "-3dMedit", "%s" % dim,  "%s" % (mesh_path+oldmesh+'.mesh')] )
        f = open(self.mesh_path+geo_tmpl+'.geo.tmpl','r')
        lines = f.readlines()
        f.close()
        geo = self.mesh_path+geo_tmpl+'.geo'
        out = open(geo,'w')
        for line in lines:
            new_line = line
            if "!oldmesh" in line:
                new_line = line.replace("!oldmesh",oldmesh)
            if "!newmesh" in line:
                new_line = line.replace("!newmesh",newmesh)
            if "!nbgrains" in line:
                new_line = line.replace("!nbgrains",str(nbgrains))
            out.write(new_line)
        out.close()
        print("\nCalling GMSH inline to save MMG output as new mesh and future MMG input\n")
        os.system("gmsh -%s %s" % (dim,geo))
        #subprocess.call(["gmsh", "-%s" % dim, "%s" % geo])
        if (not self.save_files):
            self.cleanup_files(remeshing_index-1)

    def convert_msh2mesh(self,dim,remeshing_index):
        geo = self.mesh_path+'convert_0.geo'
        c = open(geo,'w')
        c.write('Merge "%s_remeshed_%s.msh";\n' % (self.mesh_file,remeshing_index))
        c.write('Save "%s_remeshed_%s.mesh";' % (self.mesh_file,remeshing_index))
        c.close()
        os.system("gmsh -%s %s" % (dim,geo))
        
    def cleanup_files(self,remeshing_index):
        files = [f for f in os.listdir(self.mesh_path) if '_'+str(remeshing_index) in f]
        if (remeshing_index==0):
            files.remove(self.mesh_file+'_remeshed_0.msh')
        for f in files:
            os.system("rm %s%s" % (self.mesh_path,f))
        #os.system("rm %s*_%s.*" % (self.mesh_path,remeshing_index))
        