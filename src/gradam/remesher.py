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
from .hybrid_linear_solver import *
from time import sleep

# import subprocess

class Remesher:
    def __init__(self,mesh_path='',mesh_file='',sol_file='',beta=None,delta=None,number_of_nodes_index=None,\
                 sol_min=None,sol_max=None,save_files=False,metric_type="damage",slurm_job=False,nnodes=1,
                 nprocs=1,mpirun='/home/jmscherer/anaconda3/envs/fenics/bin/mpirun'):
        self.mesh_path = mesh_path
        self.mesh_file = mesh_file
        self.sol_file = sol_file
        self.beta = beta
        self.delta = delta
        self.number_of_nodes_index = number_of_nodes_index # index of the nodes number in msh files (depends on msh version)
        self.sol_min = sol_min
        self.sol_max = sol_max
        self.save_files = save_files
        self.metric_type = metric_type
        self.slurm_job = slurm_job
        self.nnodes = nnodes
        self.nprocs = nprocs
        self.mpirun = mpirun

    def diffusion(self,v,V):
        dv = TrialFunction(V)
        v_ = TestFunction(V)
        a = dv*v_*dx + self.delta**2*dot(grad(dv), grad(v_))*dx
        L = v*v_*dx #inner(v, v_)*dx
        u = Function(V)
        solver_metric = HybridLinearSolver(a,L,u,bcs=[],parameters={"iteration_switch": 5,"user_switch": True})
        #solve(a == L, u)
        if (MPI.rank(MPI.comm_world)==0):
            print("Solving the pseudo heat equation to compute the remeshing metric field")
        solver_metric.solve()
        return u
    
    def metric_damage(self,previous_metric,damage_dim,d,Vd,remeshing_index):       
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
        metric_field.vector()[:] = np.maximum(metric_field.vector()[:], previous_metric.vector()[:]) #/!\ prevents mesh to coarsen
        xdmf = XDMFFile(pyMPI.COMM_WORLD, self.mesh_path+"metric_%s.xdmf" % remeshing_index)
        xdmf.write(metric_field)
        xdmf.close()
        return metric_field
    
    def metric_elastic_energy_density(self,previous_metric,stored_energy_density,remeshing_index): #,Vmetric):
        #metric_field = Function(Vmetric,name="v:metric")
        #metric_field.vector()[:] = stored_energy.vector()[:]
        metric_field = stored_energy_density
        mini, maxi = min(metric_field.vector()), max(metric_field.vector())
        pyMPI.COMM_WORLD.barrier()
        mini, maxi = pyMPI.COMM_WORLD.allreduce(mini, op=pyMPI.MIN), pyMPI.COMM_WORLD.allreduce(maxi, op=pyMPI.MAX)
        metric_field.vector()[:] = (metric_field.vector()[:] - mini)/(max(maxi - mini,1.e-6))
        metric_field.vector()[:] = np.maximum(metric_field.vector()[:], previous_metric.vector()[:]) #/!\ prevents mesh to coarsen
        xdmf = XDMFFile(pyMPI.COMM_WORLD, self.mesh_path+"metric_%s.xdmf" % remeshing_index)
        xdmf.write(metric_field)
        xdmf.close()
        return metric_field

    def metric_damage_and_elastic_energy_density(self,previous_metric,damage_dim,d,Vd,stored_energy_density,remeshing_index):       
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
        metric_field.vector()[:] = np.maximum(metric_field.vector()[:], previous_metric.vector()[:]) #/!\ prevents mesh to coarsen

        # refine also where the stored energy density is large
        mini, maxi = min(stored_energy_density.vector()), max(stored_energy_density.vector())
        pyMPI.COMM_WORLD.barrier()
        mini, maxi = pyMPI.COMM_WORLD.allreduce(mini, op=pyMPI.MIN), pyMPI.COMM_WORLD.allreduce(maxi, op=pyMPI.MAX)
        stored_energy_density.vector()[:] = (stored_energy_density.vector()[:] - mini)/(max(maxi - mini,1.e-6))
        metric_field.vector()[:] = np.maximum(stored_energy_density.vector()[:], metric_field.vector()[:]) #/!\ prevents mesh to coarsen

        xdmf = XDMFFile(pyMPI.COMM_WORLD, self.mesh_path+"metric_%s.xdmf" % remeshing_index)
        xdmf.write(metric_field)
        xdmf.close()
        return metric_field

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
            new_sol = max( self.sol_max*max((1. - self.beta*metric_field[i]),0.)**0.5 , self.sol_min)
            s.write( '%s\n' % format(new_sol, '.4f') )   
        s.write('\nEND')   
        s.close()
    
    def remesh(self,dim,geo_tmpl,nbgrains,remeshing_index):
        if (remeshing_index==1):
            self.convert_msh2mesh(dim,remeshing_index-1)      
        oldmesh = self.mesh_file+"_remeshed_%s" % (remeshing_index-1)
        newmesh = self.mesh_file+"_remeshed_%s" % (remeshing_index) #+1)
        if (dim==2):
            medit = "-3dMedit %s" % dim
            command = "mmg%sd_O3 %s %s" % (dim,medit,self.mesh_path+oldmesh+'.mesh')
        else:
            medit = ""
            if self.slurm_job:
                slurm = self.create_slurm_script(remeshing_index,oldmesh)
                command = "sbatch %s" % slurm
                os.system('touch %s' % self.mesh_path+'remeshing_state')
                #np.savetxt(self.mesh_path+'remeshing_state.txt',[1])
            else:
                command = "%s -n %s parmmg_O3 %s" % (self.mpirun,self.nprocs,self.mesh_path+oldmesh+'.mesh')
        print("\nCalling MMG to perform remeshing: %s \n" % command )
        os.system(command)
        #subprocess.call(["mmg%sd_O3" % dim, "-3dMedit", "%s" % dim,  "%s" % (mesh_path+oldmesh+'.mesh')] )
        tt = 0
	
        #while np.loadtxt(self.mesh_path+'remeshing_state.txt')==1: #not os.path.exists(self.mesh_path+oldmesh+'.o.mesh'):
        while os.path.isfile(self.mesh_path+'remeshing_state'): #not os.path.exists(self.mesh_path+oldmesh+'.o.mesh'):
            sleep(1)
            tt += 1
            print("\n Waiting for remeshing to finish: %s s \n" % tt)
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

    def create_slurm_script(self,remeshing_index,oldmesh):
        with open(self.mesh_path+'remesh.slurm','w') as f:
            slurm = [
                "#!/bin/bash\n",
                "#SBATCH --partition=hpe-xeon #all-nodes\n",
                "#SBATCH --ntasks=%s               # Number of MPI tasks (i.e. processes)\n" % (int(self.nprocs*self.nnodes)),
                "#SBATCH --cpus-per-task=1         # Number of cores per MPI task\n",
                "#SBATCH --nodes=%s                 # Maximum number of nodes to be allocated\n" % self.nnodes,
                "#SBATCH --ntasks-per-node=%s         # Maximum number of tasks on each node\n" % self.nprocs,
                "##SBATCH --exclude=node002         # Nodes to exclude for allocation\n",
                "\n",
                "#SBATCH --mem 250Go              # Memory needed => TO ADAPT FOR YOUR COMPUTATION\n",
                "#SBATCH --job-name=poly3D        # Your job name which appear in the squeue\n",
                "#SBATCH -o %sremesh_%s_%%j.o\n" % (self.mesh_path,remeshing_index),
                "#SBATCH -e %sremesh_%s_%%j.e\n" % (self.mesh_path,remeshing_index),
                "\n",
                "/home/users02/jmscherer/anaconda3/envs/fenicsproject/bin/mpirun -n $SLURM_NTASKS parmmg_O3 %s\n" % (self.mesh_path+oldmesh+'.mesh'),
                "rm %s\n" % (self.mesh_path+'remeshing_state'),
                #"python -c \"import numpy as np; np.savetxt('%s'+'remeshing_state.txt',[0])\"\n" % (self.mesh_path)
            ]
            f.writelines(slurm)
        return self.mesh_path+'remesh.slurm'

    def convert_msh2mesh(self,dim,remeshing_index):
        geo = self.mesh_path+'convert_0.geo'
        c = open(geo,'w')
        c.write('Merge "%s_remeshed_%s.msh";\n' % (self.mesh_file,remeshing_index))
        c.write('Save "%s_remeshed_%s.mesh";' % (self.mesh_file,remeshing_index))
        c.close()
        os.system("gmsh -%s %s" % (dim,geo))
        
    def cleanup_files(self,remeshing_index):
        files = [f for f in os.listdir(self.mesh_path) if '_'+str(remeshing_index)+'.' in f]
        if (remeshing_index==0):
            files.remove(self.mesh_file+'_remeshed_0.msh')
            if self.mesh_file+'_remeshed_0.ori' in files:
                files.remove(self.mesh_file+'_remeshed_0.ori')
        for f in files:
            os.system("rm %s%s" % (self.mesh_path,f))
        #os.system("rm %s*_%s.*" % (self.mesh_path,remeshing_index))
        
