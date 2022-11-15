#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Last edited 2021-02-10

Anisotropic polycrystal multiphase field: model problem formulation,
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
import matplotlib.pyplot as plt
import numpy as np
from ufl import replace
from .material_models import *
from .hybrid_linear_solver import *
from time import time
from time import sleep
import os
from .remesher import *
from .mesh_converter import *
import types
import scipy as sp
import scipy.ndimage
import sys
import subprocess
#import h5py
from mpi4py import MPI as pyMPI
from .j_integral import *
from .version_date import *

## Setting up damage-part optimization solver
class DamageProblem(OptimisationProblem):
    def __init__(self, Wtot,DW_d,D2W_d,d):
        OptimisationProblem.__init__(self)
        self.obj_func = Wtot
        self.residual = DW_d
        self.Jac = D2W_d
        self.var = d
        self.bcs = []

    def f(self, x):
        self.var.vector()[:] = x
        return assemble(self.obj_func)

    def F(self, b, x):
        self.var.vector()[:] = x
        assemble(self.residual, b, self.bcs)

    def J(self, A, x):
        self.var.vector()[:] = x
        assemble(self.Jac, A, self.bcs)
        
class FractureProblem:
    def __init__(self,mesh,facets,mat,prefix,loads=[[0,Constant(0.)]],Jcontours=[],mf=None,mvc=None):
        self.staggered_solver = dict({"iter_max":500,"tol":1e-4,"accelerated":False,"history_function":False})
        self.comm = MPI.comm_world
        self.rank = MPI.rank(self.comm)  
        self.mesh = mesh
        self.mf = mf
        self.mvc = mvc
        self.u_degree = 1
        self.d_degree = 1
        self.num_vertices = self.mesh.num_vertices() # NOK in parallel, use metric size instead
        self.facets = facets
        self.dim = mesh.geometric_dimension()
        self.mat = mat
        self.prefix = prefix
        if (not os.path.isdir(self.prefix)):
            os.system("mkdir %s" % self.prefix)
        self.bcs = []
        self.bc_d_lb = []
        self.bc_d_ub = []
        self.Uimp = [Expression("t", t=0, degree=0)]
        self.incr = 0
        self.incr_save = 1
        self.t = 0.
        self.dtime = 1.e-4
        self.desired_dincr = 1.e-2
        self.max_dincr = 1.e-2
        self.min_dtime = 1.e-5
        self.max_dtime = 1.e-3
        self.final_time = 1.e-2
        self.niter_tot = 0
        self.niter_TAO = 0
        self.niter_iterative = 0
        self.niter_direct = 0
        self.set_functions()
        self.dx = Measure("dx")
        self.ds = Measure("ds")
        self.loads = loads
        self.load_boundaries = [self.ds]
        self.Wext = self.define_external_work()
        self.resultant = self.sig[1,1]*self.ds      
        self.results = XDMFFile(MPI.comm_world,self.prefix+"output.xdmf")
        self.results.parameters["flush_output"] = True
        self.results.parameters["functions_share_mesh"] = True
        self.J_results = XDMFFile(MPI.comm_world,self.prefix+"J_integral.xdmf")
        self.J_results.parameters["flush_output"] = True
        self.J_results.parameters["functions_share_mesh"] = True
        self.checkpoints = XDMFFile(MPI.comm_world,self.prefix+"checkpoint.xdmf")
        self.checkpoints.parameters["flush_output"] = True
        self.checkpoints.parameters["functions_share_mesh"] = True
        self.save_all = True
        self.save_intermediate = False
        self.save_checkpoints = False
        self.write_checkpoint_count = 0
        self.use_hybrid_solver = False
        self.normal = FacetNormal(mesh)
        self.dsJ = Jcontours
        self.J = []
        self.remesh = False
        self.use_remeshing = False
        self.remeshing_criterion = 1.e-2
        self.remeshing_index = 1
        self.nbgrains  = -1
        self.remesher = None
        self.boundaries = []
        self.markers = []
        self.domains = []
        self.domains_markers = []
        self.myBCS = self.BCS(self.Vu,self.Vd,self.facets)
        self.myResultant = self.Resultant(self.mat,self.u,self.d,self.P1pos,self.P2pos,self.P3pos,self.ds)
        self.gaussian_filter_sigma = 1.
        self.no_residual_stiffness=[False,0.99]
        self.JIntegral = None
        self.timings = True
        self.null_space_basis = None
        self.rigid_body_motion=[]

    class BCS():
        def __init__(self, Vu, Vd, facets):
            self.Vu = Vu
            self.Vd = Vd
            self.facets = facets
    class Resultant():
        def __init__(self, mat, u, d, P1pos, P2pos, P3pos, ds):
            self.mat = mat
            self.u = u
            self.d = d
            self.P1pos = P1pos
            self.P2pos = P2pos
            self.P3pos = P3pos
            self.ds = ds
        
    def set_functions(self):
        # Definition of functions spaces and test/trial functions
        self.Vu = VectorFunctionSpace(self.mesh, "CG", self.u_degree, dim=self.dim)
        if self.mat.damage_dim == 1:
            self.Vd = FunctionSpace(self.mesh, "CG", self.d_degree)
        else:
            self.Vd = VectorFunctionSpace(self.mesh, "CG", self.d_degree, dim=self.mat.damage_dim)
        self.V0 = FunctionSpace(self.mesh, "DG", 0)
        self.Vsig = TensorFunctionSpace(self.mesh, "DG", 0, shape=(3,3)) #self.u_degree, shape=(3,3))
        self.VV = VectorFunctionSpace(self.mesh, "DG", 0, dim=3)
        self.Vr = TensorFunctionSpace(self.mesh, "DG", 0, shape=(3,3))
        self.Vmetric = FunctionSpace(self.mesh, "CG", self.d_degree)  
        
        self.u_ = TestFunction(self.Vu)
        self.du = TrialFunction(self.Vu)
        self.d_ = TestFunction(self.Vd)
        self.dd = TrialFunction(self.Vd)
        
        # Definition of functions
        self.u = Function(self.Vu,name="Displacement")
        #self.u_prev = Function(self.Vu,name="Previous displacement")
        self.d = Function(self.Vd,name="Damage")
        self.d_prev = Function(self.Vd,name="Previous damage")
        self.d_prev_iter = Function(self.Vd,name="Previous damage in staggered minimization")
        #self.dold = Function(self.Vd,name="Old damage")
        self.d_lb = Function(self.Vd,name="Lower bound d_n")
        self.d_ub = Function(self.Vd,name="Damage upper bound") #"Upper bound 1")
        self.d_ar= Function(self.Vd,name="Damage field after remeshing") 
        self.d_ub = self.mat.dub
        self.sig = Function(self.Vsig,name="Stress")
        self.eel = Function(self.Vsig,name="ElasticStrain")
        self.epspos = Function(self.Vsig,name="Strain (+)")
        self.epsneg = Function(self.Vsig,name="Strain (-)")
        #self.V1 = Function(self.VV,name="V1")
        #self.V2 = Function(self.VV,name="V2")
        #self.V3 = Function(self.VV,name="V3")
        # if self.staggered_solver["accelerated"]:
        #     self.tmp_u = Function(self.Vu)
        #     self.tmp_d = Function(self.Vd)
        self.R = Function(self.Vr,name="Rotation matrix")
        self.dissipated = Function(self.V0,name="Plastic dissipation")
        self.stored = Function(self.V0,name="Stored energy")
        self.P1pos = Function(self.V0,name="P1pos")
        self.P2pos = Function(self.V0,name="P2pos")
        self.P3pos = Function(self.V0,name="P3pos")
        self.P1pos.interpolate(Constant(1.))
        self.P2pos.interpolate(Constant(1.))
        self.P3pos.interpolate(Constant(1.))
        self.Efrac_field = Function(self.V0,name="Efrac")
        self.d_eq_fiss = Function(self.Vd,name="deq")
        if self.mat.damage_dim==1:
            self.d_eq_fiss.interpolate(Constant((1.)))
            self.d_prev_iter.interpolate(Constant(0.))
        else:
            self.d_eq_fiss.interpolate(Constant((1.,)*self.mat.damage_dim))
            self.d_prev_iter.interpolate(Constant((0.,)*self.mat.damage_dim))
        #self.Vstiffness = TensorFunctionSpace(self.mesh, "CG", 1, shape=(6,6))
        #self.stiffness = Function(self.Vstiffness,name="Stiffness")
        self.metric = Function(self.Vmetric,name="Remeshing metric")
        self.metric.interpolate(Constant(0.))
           
    def set_load(self,u):
        L = self.loads[0][1]*u[self.loads[0][0]]*self.load_boundaries[0]
        for (load,load_boundary) in list(zip(self.loads[1:],self.load_boundaries[1:])):
            L += load[1]*u[load[0]]*load_boundary
        return L
 
    def define_external_work(self):
        return self.set_load(self.u)
        #return dot(self.load,self.u)*self.ds
        
    def set_energies(self):
        if (not self.mat.behaviour=="linear_elasticity"):
            self.mb = self.mat.mfront_behaviour.create_material()
            self.solver_u = mf.MFrontNonlinearProblem(self.u, self.mb, quadrature_degree=self.mat.mfront_behaviour.quadrature_degree, bcs=self.bcs)
            self.solver_u.register_external_state_variable("Damage", self.d)
            '''
            prm = self.solver_u.solver.parameters
            #prm['nonlinear_solver'] = 'newton'
            prm['linear_solver'] = 'gmres' #'mumps' #'minres' #'cg' #'cg' #'mumps' #'gmres' #'petsc' #'umfpack' #'tfqmr'
            #prm['preconditioner'] = 'petsc_amg' #'ilu' # 'sor' # 'icc' # 'petsc_amg'
            #prm['krylov_solver']['error_on_nonconvergence'] = True
            #prm['krylov_solver']['monitor_convergence'] = True
            #prm['krylov_solver']['absolute_tolerance'] = 1E-14
            #prm['krylov_solver']['relative_tolerance'] = 1E-14
            #prm['krylov_solver']['maximum_iterations'] = 10000
            prm['krylov_solver']['nonzero_initial_guess'] = True            
            prm['preconditioner'] = 'hypre_amg'
            prm['absolute_tolerance'] = 1E-6 #-9
            prm['relative_tolerance'] = 1E-8 #-8
            #prm['maximum_iterations'] = 1000 #25
            #prm['relaxation_parameter'] = 1.
            ##prm['krylov_solver']['gmres']['restart'] = 40
            ##prm['krylov_solver']['preconditioner']['ilu']['fill_level'] = 0
            prm["report"] = True
            #prm['lu_solver']['symmetric'] = True #False
            '''

            ''''''
            self.solver_u.solver = PETScSNESSolver('newtonls') #'newtontr') #'newtonls')
            prm = self.solver_u.solver.parameters
            #prm['nonlinear_solver'] = 'snes'
            #prm['line_search'] =  'bt' #'cp' #'cp' #'nleqerr' # 'bt' # 'basic' # 'l2'
            prm['line_search'] =  'nleqerr'
            #prm['linear_solver'] = 'mumps'
            prm['linear_solver'] = 'cg' #'gmres' #'cg' #'gmres'
            prm['preconditioner'] = 'amg' #'hypre_amg'
            prm['krylov_solver']['nonzero_initial_guess'] = False # True
            #prm['maximum_iterations'] = 50
            tol = 1.0E-6
            prm['absolute_tolerance'] = tol
            prm['relative_tolerance'] = tol
            prm['solution_tolerance'] = tol
            #prm['report'] = False #True
            ''''''
            
            self.load = self.set_load(self.u)
            self.solver_u.set_loading(self.load)
            self.dissipated.vector().set_local(self.mb.data_manager.s1.dissipated_energies)
            self.dissipated.vector().apply("insert")
            self.stored.vector().set_local(self.mb.data_manager.s1.stored_energies)
            self.stored.vector().apply("insert")
            #print(max(_dummy_function.vector()[:]))
            self.sigma()
            #self.eps_elas()
            ## self.Wel = 0.5*inner(self.sig,self.eel)*self.dx
            ## self.Wel = 0.5*(1.-self.d)**2*inner(self.sig,self.eel)*self.dx
            self.Wel = (1.-self.d)**2*self.stored*self.dx
            #self.Wel = 0.5*self.stored*self.dx
            self.Wdis = (1.-self.d)**2*self.dissipated*self.dx  
        else:  
            # Definition of energy densities            
            # self.Wel = 0.5*inner(self.mat.sigma(self.u,self.d,self.P1pos,self.P2pos),eps(self.u))*self.dx
            # self.Wel = 0.5*inner(self.mat.sigma(self.u,self.d,self.P1pos,self.P2pos,self.P3pos),eps(self.u,self.dim))*self.dx
            self.sigma()
            self.Wel = 0.5*inner(self.sig,eps(self.u,self.dim))*self.dx
        
        self.Efrac = self.mat.fracture_energy_density(self.d,self.d_prev_iter)
        self.Wfrac = sum(self.Efrac)*self.dx
        self.Wtot = self.Wel + self.Wfrac - self.Wext
        if (not self.mat.behaviour=="linear_elasticity"):
            self.Wtot += self.Wdis

        # Definition of J integral
        if (self.dim == 2):
            normal3  = as_vector([self.normal[0],self.normal[1],0.])
            sigma_n3 = dot(normal3, self.sig)
            sigma_n  = as_vector([sigma_n3[0],sigma_n3[1]])
        elif (self.dim == 3):
            normal3 = self.normal
            sigma_n = dot(normal3, self.sig)
            
        if (not self.dsJ==[]):
            self.J=[]
            for c in self.dsJ:
                #self.J.append( (0.5*inner(self.sig,eps(self.u,self.dim))*self.normal[1] \
                #               - inner(sigma_n, grad(self.u)[:,0]) ) * c ) # for outer boundaries
                self.J.append( (0.5*inner(self.sig,eps(self.u,self.dim))*self.normal[1]) * c )
                self.J.append( (- inner(sigma_n, grad(self.u)[:,0]) ) * c ) # for outer boundaries
            for c in self.dsJ:
                #self.J.append( (0.5*inner(self.sig,eps(self.u,self.dim))*self.normal[0] \
                #               - inner(sigma_n, grad(self.u)[:,1]) ) * c ) # for outer boundaries
                self.J.append( (0.5*inner(self.sig,eps(self.u,self.dim))*self.normal[0] ) * c)
                self.J.append( (- inner(sigma_n, grad(self.u)[:,1]) ) * c ) # for outer boundaries
                # self.J.append( (0.5*inner(self.sig,eps(self.u,self.dim))*self.normal[1] \
                #                - inner(sigma_n, grad(self.u)[:,0]) )('-') * c ) # for inner boundaries
        
        # Definition of energy derivatives
        self.DW_u = derivative(self.Wtot,self.u,self.u_)
        self.D2W_u = derivative(self.DW_u,self.u,self.du)
        self.DW_d = derivative(self.Wtot,self.d,self.d_)
        self.D2W_d = derivative(self.DW_d,self.d,self.dd)

    def set_problems(self):
        if not self.rigid_body_motion==[]:
            null_space = [interpolate(n, self.Vu).vector() for n in self.rigid_body_motion]
            # Make into unit vectors
            [normalize(n, 'l2') for n in null_space]
            # Create null space basis object
            self.null_space_basis = VectorSpaceBasis(null_space)

        # Setting up displacement-part linear solver 
        # LinearVariationalProblem(lhs(self.D2W_u),replace(self.Wext,{self.u:self.u_}), self.u, self.bcs)
        if (self.mat.behaviour=='linear_elasticity'):
            self.load = self.set_load(self.u_)
            if (self.use_hybrid_solver):
                #self.solver_u = HybridLinearSolver(lhs(self.D2W_u),dot(self.load,self.u_)*self.ds,\
                self.solver_u = HybridLinearSolver(lhs(self.D2W_u),self.load,\
                                                   self.u,bcs=self.bcs,parameters={"iteration_switch": 5,\
                                                   "user_switch": True},null_space_basis=self.null_space_basis) #not self.remesh or (self.niter>0)})
            else:
                if (not self.mat.tension_compression_asymmetry):
                    #self.problem_u = LinearVariationalProblem(lhs(self.D2W_u),dot(self.load,self.u_)*self.ds,self.u,self.bcs)
                    self.problem_u = LinearVariationalProblem(lhs(self.D2W_u),self.load,self.u,self.bcs)
                    self.solver_u = LinearVariationalSolver(self.problem_u)
                    self.solver_u.parameters["linear_solver"] = "mumps"
                else:
                    self.problem_u = NonlinearVariationalProblem(self.DW_u,self.u,self.bcs,J=self.D2W_u)
                    self.solver_u = NonlinearVariationalSolver(self.problem_u)
                    prm = self.solver_u.parameters
                    prm['nonlinear_solver'] = 'newton'
                    prm['newton_solver']['linear_solver'] = 'mumps' #'gmres' #'mumps' #'petsc'
                    
                    prm['newton_solver']['error_on_nonconvergence'] = False #True
                    prm['newton_solver']['absolute_tolerance'] = 1E-9
                    prm['newton_solver']['relative_tolerance'] = 1E-8
                    prm['newton_solver']['maximum_iterations'] = 25 #10000 #25
                    prm['newton_solver']['relaxation_parameter'] = 1.0
                    
                    prm['newton_solver']['lu_solver']['report'] = True
                    #prm['newton_solver']['lu_solver']['reuse_factorization'] = False
                    #prm['newton_solver']['lu_solver']['same_nonzero_pattern'] = False
                    prm['newton_solver']['lu_solver']['symmetric'] = False
                    
                    prm['newton_solver']['krylov_solver']['error_on_nonconvergence'] = True
                    prm['newton_solver']['krylov_solver']['absolute_tolerance'] = 1E-7
                    prm['newton_solver']['krylov_solver']['relative_tolerance'] = 1E-5
                    prm['newton_solver']['krylov_solver']['maximum_iterations'] = 1000
                    prm['newton_solver']['krylov_solver']['nonzero_initial_guess'] = True
                    if prm['newton_solver']['linear_solver'] == 'gmres':
                        prm['newton_solver']['preconditioner'] = 'ilu'
                    #self.solver_u.parameters["newton_solver"]["linear_solver"] = "mumps"              
        
        self.solver_d = PETScTAOSolver()
        self.solver_d.parameters["method"] = "tron"
        self.solver_d.parameters["line_search"] = "gpcg"
        self.solver_d.parameters["linear_solver"] = "cg" #"mumps" #"mumps" #'cg", "gltr", "gmres", "nash"
        #self.solver_d.parameters["preconditioner"] = "hypre_amg"
        self.solver_d.parameters["maximum_iterations"] = 5000
        # self.solver_d.parameters["gradient_absolute_tol"] = self.staggered_solver["tol"]
        # self.solver_d.parameters["gradient_relative_tol"] = self.staggered_solver["tol"]
        # self.solver_d.parameters["gradient_t_tol"] = self.staggered_solver["tol"]
        self.solver_d.parameters["gradient_absolute_tol"] = 1.e-4
        self.solver_d.parameters["gradient_relative_tol"] = 1.e-4
        self.solver_d.parameters["gradient_t_tol"] = 1.e-4
           
    #@profile 
    def staggered_solve(self):
        DeltaE = 1.
        self.niter = 0
        if(self.remesh == False):
            self.d_prev.assign(self.d) 
        
        # boundary conditions for damage problem
        for bc in self.bc_d_lb:
            bc.apply(self.d_lb.vector())
        for bc in self.bc_d_ub:
            bc.apply(self.d_ub.vector())
        
        while (DeltaE>self.staggered_solver["tol"]) and (self.niter<self.staggered_solver["iter_max"]): 
            if self.rank == 0:
                print("    Iteration %i "%(self.niter))

            # u-solve : gives u_{n+1}^{pred} from d_{n}
            tic_u = time()
            if self.mat.behaviour=='linear_elasticity':
                count = self.solver_u.solve()
                self.sigma()
            else:
                self.solver_u.dt = self.dtime
                count = self.solver_u.solve(self.u.vector())
                self.dissipated.vector().set_local(self.mb.data_manager.s1.dissipated_energies)
                self.dissipated.vector().apply("insert")
                self.stored.vector().set_local(self.mb.data_manager.s1.stored_energies)
                self.stored.vector().apply("insert")
                self.sigma()
            self.runtime_u += time() - tic_u
            if self.use_hybrid_solver:
                self.niter_iterative += count[0]
                self.niter_direct += count[1]
            else:
                self.niter_direct += 1
            '''            
            # Update pos/neg indicators        
            Eloc = strain2voigt(dot(self.mat.R,dot(eps(self.u,self.dim),self.mat.R.T)))
            #self.P1pos = Heav(Eloc[0])
            self.P1pos.assign(local_project(Heav(Eloc[0]),self.V0))
            #self.P2pos = Heav(Eloc[1])
            self.P2pos.assign(local_project(Heav(Eloc[1]),self.V0))
            #self.P3pos = Heav(Eloc[2])
            self.P3pos.assign(local_project(Heav(Eloc[2]),self.V0))
            # print("\nP1pos = ", np.sum(self.P1pos.vector()[:]))
            '''

            # u-update
            # self.u_prev.assign(self.u)
            
            tic_assemble = time()
            Etot_old = assemble(self.Wtot)
            self.runtime_assemble += time() - tic_assemble

            # d-solve : gives d_{n+1}^{pred} from u^{n+1}
            #self.dold.assign(self.d)   
            dam_prob = DamageProblem(self.Wtot,self.DW_d,self.D2W_d,self.d)
            tic_d = time()
            self.niter_TAO += self.solver_d.solve(dam_prob, self.d.vector(), self.d_lb.vector(), self.d_ub.vector())[0]
            self.runtime_d += time() - tic_d
            
            self.d_prev_iter.assign(self.d)

            if (not self.mat.behaviour=='linear_elasticity'):
                self.sigma()

            # Energy computation
            tic_assemble = time()
            Etot = assemble(self.Wtot)
            self.runtime_assemble += time() - tic_assemble

            if (not Etot==0.):
                DeltaE = abs(Etot_old/Etot-1)
            else:
                DeltaE = abs(Etot_old - Etot)
            
            self.niter += 1
            self.niter_tot += 1
            if self.rank == 0:
                print("        Energy variation : %.5e"%(DeltaE))
            
            # if self.save_intermediate == True:
            #     self.user_postprocess()
        if self.mat.damage_dim==2 and self.no_residual_stiffness==True:
            d1,d2 = self.equalize_damage(self.d,self.d_ub,0.99)
            assign(self.d.sub(0),d1)
            assign(self.d.sub(1),d2)
        '''
        # Update lower bound to account for irreversibility
        self.d_lb.assign(self.d)
        self.max_dincr = norm(self.d.vector()-self.d_prev.vector(),'linf')
        if (self.max_dincr==0.):
            self.max_dincr = self.desired_dincr/2.
        '''

        # Update pos/neg indicators        
        Eloc = strain2voigt(dot(self.mat.R,dot(eps(self.u,self.dim),self.mat.R.T)))
        #self.P1pos = Heav(Eloc[0])
        self.P1pos.assign(local_project(Heav(Eloc[0]),self.V0))
        #self.P2pos = Heav(Eloc[1])
        self.P2pos.assign(local_project(Heav(Eloc[1]),self.V0))
        #self.P3pos = Heav(Eloc[2])
        self.P3pos.assign(local_project(Heav(Eloc[2]),self.V0))
        # print("\nP1pos = ", np.sum(self.P1pos.vector()[:]))
        # print("\nP2pos = ", np.sum(self.P2pos.vector()[:]))

    def equalize_damage(self, d, d_ub, threshold):
        d1,d2 = self.d.split(deepcopy = True)
        d1_ub,d2_ub = d_ub.split(deepcopy = True)
        d1_ = d1.vector()[:]
        d2_ = d2.vector()[:]
        d1_ub_ = d1_ub.vector()[:]
        d2_ub_ = d2_ub.vector()[:]
        np.place(d1_, d2_>threshold, d1_ub_)
        np.place(d2_, d1_>threshold, d2_ub_)
        d1.vector()[:] = d1_
        d2.vector()[:] = d2_
        return d1, d2
        
    def export_damage(self,t):
        tic_export = time()
        self.results.write(self.d,t)
        #self.results.write(self.d_eq_fiss,t)
        if ((not self.write_checkpoint_count % 10) and self.save_checkpoints):
            self.results.write_checkpoint(self.d,"Damage",t,append=True)
            self.write_checkpoint_count = 1
        self.write_checkpoint_count += 1
        self.runtime_export += time() - tic_export

    def export_J(self,t):
        tic_export = time()
        self.sigma()
        #os.system('rm %s' % self.prefix+"J_integral.xdmf")
        if (self.mat.behaviour=="linear_elasticity"):
            sigma = Function(self.Vsig,name="Stress")
            sigma.assign(local_project(self.sig,self.Vsig)) #, solver_type='cg', preconditioner_type='hypre_amg'))
            self.J_results.write(sigma,t) 
        else:
            #sigma = Function(self.Vsig,name="Stress")
            #sigma.assign(local_project((1.-self.d)**2*self.sig,self.Vsig))
            #self.results.write(sigma,t)
            flux_names = self.solver_u.material.get_flux_names()
            stress_name = "MissingStress?"
            for name in flux_names:
                if "Stress" in name:
                    stress_name = name
            self.J_results.write(self.solver_u.get_flux(stress_name, project_on=("DG", 0), as_tensor=True),t)
            #self.results.write((1.-self.d)**2*self.sig,t)
        self.J_results.write(self.u,t)
        #self.J_results.write(self.sig,t)
        self.runtime_export += time() - tic_export
            
    def export_all(self,t):
        tic_export = time()
        self.sigma()
        if (self.mat.behaviour=="linear_elasticity"):
            sigma = Function(self.Vsig,name="Stress")
            sigma.assign(local_project(self.sig,self.Vsig))
            self.results.write(sigma,t) 
        else:
            #sigma = Function(self.Vsig,name="Stress")
            #sigma.assign(local_project((1.-self.d)**2*self.sig,self.Vsig)) #, solver_type='cg', preconditioner_type='hypre_amg'))
            #Vsig = TensorFunctionSpace(self.mesh, "DG", 0, shape=(3,3))
            #sigma.assign(local_project(self.sig,Vsig))
            #self.results.write(sigma,t)
            flux_names = self.solver_u.material.get_flux_names()
            stress_name = "MissingStress?"
            for name in flux_names:
                if "Stress" in name:
                    stress_name = name
            #sigma = Function(self.Vsig,name=stress_name)
            ##sig = (1.-self.d)**2*self.solver_u.get_flux(stress_name, project_on=("DG", 0), as_tensor=True)
            #sigma.assign(local_project(self.sig,self.Vsig))
            self.results.write(self.solver_u.get_flux(stress_name, project_on=("DG", 0), as_tensor=True),t)
            #self.results.write(self.solver_u.get_flux(stress_name, project_on=("DG", 0), as_tensor=True),t)
            #self.results.write((1.-self.d)**2*self.sig,t) 
        self.results.write(self.u,t)
        ##self.epspos.assign(local_project(dot(self.mat.R.T,dot(voigt2strain(self.mat.eps_crystal_pos),self.mat.R)),self.Vsig)) #, solver_type='cg', preconditioner_type='hypre_amg'))
        ##self.epsneg.assign(local_project(dot(self.mat.R.T,dot(voigt2strain(self.mat.eps_crystal_neg),self.mat.R)),self.Vsig)) #, solver_type='cg', preconditioner_type='hypre_amg'))
        ##self.results.write(self.epspos,t)
        ##self.results.write(self.epsneg,t)
        # write rotation matrix
        #self.R.assign(project(self.mat.R,self.Vr)) #, solver_type='cg', preconditioner_type='hypre_amg'))
        #self.results.write(self.R,t)
        self.results.write(self.mat.phi1,t)
        self.results.write(self.mat.Phi,t)
        self.results.write(self.mat.phi2,t)
        
        if (not self.mat.behaviour=='linear_elasticity'):
            for var in self.mb.get_internal_state_variable_names():
                as_tensor = False
                if ("Deformation" in var) or ("ElasticStrain"==var) or ("PlasticStrain"==var):
                    as_tensor = True
                self.results.write(self.solver_u.get_state_variable(var,project_on=('DG',0),as_tensor=as_tensor),t)
            
        #self.V1.assign(local_project(self.mat.R[0,:],self.VV))
        #self.V2.assign(local_project(self.mat.R[1,:],self.VV))
        #self.V3.assign(local_project(self.mat.R[2,:],self.VV))
        #self.results.write(self.V1,t)
        #self.results.write(self.V2,t)
        #self.results.write(self.V3,t)
        #self.results.write(self.P1pos,t)
        #self.results.write(self.P2pos,t)
        #self.results.write(self.P3pos,t)
        self.Efrac_field.assign(local_project(sum(self.Efrac),self.V0))
        self.results.write(self.Efrac_field,t)
        #self.stiffness.assign(local_project(self.mat.C,self.Vstiffness))
        #self.results.write(self.stiffness,t)

        if ((not self.write_checkpoint_count % 10) and self.save_checkpoints):
            self.checkpoints.write_checkpoint(self.u,self.u.name(),t,append=True)
        self.runtime_export += time() - tic_export
                   
    def solve(self):
        self.startup_message()
        #log = [[0]*13]
        if (self.rank==0):
            f = open(self.prefix+"results.txt","a")
            Jint_cols = ''
            if (not self.dsJ==[]):
                Jint_cols = "[15-%s]_J_integrals " % (16+len(self.dsJ)-1)
            f.write("#1_incr 2_time 3_F 4_Eel 5_Ed 6_Ep 7_Etot 8_niter 9_niter_tot 10_niter_TAO 11_niter_iterative 12_niter_direct 13_runtime 14_runtime_u 15_runtime_d "\
                    + Jint_cols + "[%s-%s]_Efrac_i " % (16+len(self.dsJ),16+len(self.dsJ)+self.mat.damage_dim-1) + "\n")
            f.close()
            if (not self.JIntegral==None):
                for contour in self.JIntegral.keys():
                    J_file = open(self.prefix+'J_integral_%s.txt' % contour,'a')
                    J_file.write("#1_incr 2_time J_left J_bot J_right J_top J_tot\n")
                    J_file.close()
            if (self.timings==True):
                f = open(self.prefix+"timings.txt","a")
                f.write("#1_incr 2_time 3_runtime_u 4_runtime_d 5_runtime_assemble "+\
                        "6_runtime_export 7_runtime_JIntegral 8_runtime_remeshing_mmg 9_runtime_remeshing_interpolation 10_runtime_tot "+\
                        "11_number_of_vertices\n")
                f.close()
                
        self.runtime   = 0. #time()
        self.runtime_u = 0.
        self.runtime_d = 0.
        self.runtime_assemble = 0.
        self.runtime_export = 0.
        self.runtime_remeshing_mmg = 0.
        self.runtime_remeshing_interpolation = 0.
        self.runtime_JIntegral = 0.
        self.set_energies()
        self.set_problems()
    
        while (self.t < self.final_time):
            tic = time()
            if (self.remesh == False):
                self.dtime = max(self.dtime*self.desired_dincr/self.max_dincr,self.min_dtime)
                self.dtime = min(self.dtime,self.max_dtime)
                if ((self.t + self.dtime) > self.final_time):
                    self.dtime = self.final_time - self.t
                if (self.incr==0):
                    self.dtime = 0
                for load in self.loads:
                    load[1].t = self.t + self.dtime
                for uimp in self.Uimp:
                    uimp.t = self.t + self.dtime
            if self.rank == 0:
                print( "Increment %i | Time : %.5e | dt : %.5e"%(self.incr,self.t+self.dtime,self.dtime))

            self.staggered_solve()

            if (self.use_remeshing):
                self.remeshing()
            if self.remesh == False:
                # Update lower bound to account for irreversibility
                self.d_lb.assign(self.d)
                self.max_dincr = norm(self.d.vector()-self.d_prev.vector(),'linf')
                if (self.max_dincr==0.):
                    self.max_dincr = self.desired_dincr/2.
                self.t += self.dtime
                
                if ((not self.incr % self.incr_save) or (self.incr < 1)):
                    if self.save_all:
                        self.export_all(self.t/self.final_time)
                    self.export_damage(self.t/self.final_time)
                    
                if(not self.JIntegral==None):
                    self.export_J(self.t/self.final_time)
    
                tic_assemble = time()
                F = assemble(self.resultant)
                Eel = assemble(self.Wel)
                Ed = assemble(self.Wfrac)
                Ep = 0.
                if (not self.mat.behaviour=="linear_elasticity"):
                    Ep = assemble(self.Wdis)
                Etot = assemble(self.Wtot)
                Efrac = [assemble(Efrac_i*self.dx) for Efrac_i in self.Efrac]
                Jint=[]
                if (not self.J==[]):
                    for j in self.J:
                        Jint.append( assemble(j) )
                self.runtime_assemble += time() - tic_assemble       
                
                if ((not self.JIntegral==None) and self.rank==0):
                        tic_JIntegral = time()
                        for contour in self.JIntegral.keys():
                            if ((not self.incr % self.JIntegral[contour].incr_save) or (self.incr < 1)):
                                self.JIntegral[contour].compute_J_integral(contour,self.incr, self.t)
                        self.runtime_JIntegral += time() - tic_JIntegral
                    
                self.runtime += time() - tic
                if (self.rank==0):
                    log = ([self.incr,self.t,F,Eel,Ed,Ep,Etot,self.niter,self.niter_tot, \
                            self.niter_TAO,self.niter_iterative,self.niter_direct,self.runtime,\
                            self.runtime_u,self.runtime_d] + Jint + Efrac)
                    f = open(self.prefix+"results.txt","a")
                    f.write(' '.join(map(str, log))+"\n")
                    f.close()
                    
                    if (self.timings==True):
                        timings = ([self.incr,self.t,self.runtime_u,self.runtime_d,self.runtime_assemble,\
                                    self.runtime_export,self.runtime_JIntegral,self.runtime_remeshing_mmg,self.runtime_remeshing_interpolation,self.runtime,\
                                    self.num_vertices])
                        f = open(self.prefix+"timings.txt","a")
                        f.write(' '.join(map(str, timings))+"\n")
                        f.close()
                    
                self.incr += 1
            
        if (self.rank==0):
             f = open(self.prefix+"results.txt","a")
             f.write("# elapsed time in solve()     = %.3f seconds\n" % self.runtime)
             f.write("# elapsed time in solve_u()   = %.3f seconds\n" % self.runtime_u)
             f.write("# elapsed time in solve_d()   = %.3f seconds\n" % self.runtime_d)
             f.write("# elapsed time in assemble()   = %.3f seconds\n" % self.runtime_assemble)
             f.write("# elapsed time in export()   = %.3f seconds\n" % self.runtime_export)
             f.write("# elapsed time in JIntegral() = %.3f seconds\n" % self.runtime_JIntegral)
             f.write("# elapsed time in remeshing_mmg() = %.3f seconds\n" % self.runtime_remeshing_mmg)
             f.write("# elapsed time in remeshing_interpolation() = %.3f seconds\n" % self.runtime_remeshing_interpolation)
             f.close()
             
    def remeshing(self):
        tic_remeshing_mmg = time()
        self.d_var = Function(self.Vd,name="Damage variation")
        self.d_var.vector()[:] = self.d.vector()[:] - self.d_ar.vector()[:]           
        max_d_var = max(self.d_var.vector()) #norm(self.d_var.vector(),'linf') #max(self.d_var.vector())
        self.comm.barrier()
        max_d_var = self.comm.allreduce(max_d_var, op=pyMPI.MAX)
        if (max_d_var > self.remeshing_criterion):
        #if (np.hstack(self.comm.allgather(self.d_var.compute_vertex_values())).max() > self.remeshing_criterion):
            self.remesh = True
            #self.d_var_smooth = filter_function(self.d,self.d_var,self.Vd,self.gaussian_filter_sigma,\
            #                                    self.mat.damage_dim,self.dim)
            #self.metric_field = np.array(self.d_var_smooth.compute_vertex_values())
            self.metric = self.remesher.metric(self.metric,self.mat.damage_dim,self.d,self.Vd,self.remeshing_index)
            if (self.rank == 0):
                metric = meshio.xdmf.read(self.remesher.mesh_path+"metric_%s.xdmf" % self.remeshing_index).point_data["v:metric"][:,0]
                self.num_vertices = metric.size
                self.remesher.write_sol(metric,self.num_vertices,self.remeshing_index)
                geo_tmpl = 'mmg_tmp'
                self.remesher.remesh(self.dim,geo_tmpl,self.nbgrains,self.remeshing_index)
            xdmf = MeshioMsh2Xdmf(self.dim,self.remesher.mesh_path+self.remesher.mesh_file+'_remeshed_%s'\
                                   % self.remeshing_index,extras='')
            xdmf.write_xdmf_mesh()
            xdmf.read_xdmf_mesh()
            self.mesh = xdmf.mesh
            #self.num_vertices = self.mesh.num_vertices() # NOK in parallel, use metric size insteadn.num
            self.mf = xdmf.mf
            self.facets = xdmf.facets
            self.mvc = xdmf.mvc
            self.dx = xdmf.dx
            self.normal = FacetNormal(self.mesh)
            hmin, hmax = self.mesh.hmin(), self.mesh.hmax()
            self.comm.barrier()
            hmin, hmax = self.comm.allreduce(hmin, op=pyMPI.MIN), self.comm.allreduce(hmax, op=pyMPI.MAX)
            if (self.rank == 0):
                print("max cell size =", hmax)
                print("min cell size =", hmin)
            self.runtime_remeshing_mmg += time() - tic_remeshing_mmg
            tic_remeshing_interpolation = time()
                
            for (boundary,marker) in list(zip(self.boundaries,self.markers)):
                boundary().mark(self.facets, marker)
            for (domain,domain_marker) in list(zip(self.domains,self.domains_markers)):
                domain().mark(self.mf, domain_marker)
            if (not self.dsJ==[]):
                mfJ = MeshFunction("size_t", self.mesh, 1)
                dsj = Measure("ds", subdomain_data=mfJ)
                #dSJ = Measure("dS", subdomain_data=mfJ)
                for (i,(Jcontour,Jmarker)) in enumerate(list(zip(self.jcontours,self.jmarkers))):
                    Jcontour().mark(mfJ, Jmarker)
                    self.dsJ[i] = dsj(Jmarker)           
            self.ds = Measure("ds", subdomain_data=self.facets)
            self.mat = EXD(self.mat.dim,self.mat.damage_dim,self.mat.mp,\
                           self.mesh,self.mf,self.mat.geometry,behaviour=self.mat.behaviour,mfront_behaviour=self.mat.mfront_behaviour,\
                           damage_model=self.mat.damage_model,anisotropic_elasticity=self.mat.anisotropic_elasticity,\
                           damage_tensor=self.mat.damage_tensor)
            
            # Re-Definition of functions spaces
            self.Vu = VectorFunctionSpace(self.mesh, "CG", self.u_degree, dim=self.dim)
            if self.mat.damage_dim == 1:
                self.Vd = FunctionSpace(self.mesh, "CG", self.d_degree)
            else:
                self.Vd = VectorFunctionSpace(self.mesh, "CG", self.d_degree, dim=self.mat.damage_dim)
            self.V0 = FunctionSpace(self.mesh, "DG", 0)    
            self.Vsig = TensorFunctionSpace(self.mesh, "CG", 1, shape=(3,3))
            self.VV = VectorFunctionSpace(self.mesh, "DG", 0, dim=3)
            self.Vr = TensorFunctionSpace(self.mesh, "DG", 0, shape=(3,3))
            #self.Vstiffness = TensorFunctionSpace(self.mesh, "CG", 1, shape=(6,6))
            self.Vmetric = FunctionSpace(self.mesh, "CG", self.d_degree)    
                
            self.u_ = TestFunction(self.Vu)
            self.du = TrialFunction(self.Vu)
            self.d_ = TestFunction(self.Vd)
            self.dd = TrialFunction(self.Vd)
            
            # Interpolation of functions onto the new function spaces
            tmp = self.u
            self.u = Function(self.Vu,name="Displacement") 
            LagrangeInterpolator.interpolate(self.u,tmp)

            #tmp = self.u_prev
            #self.u_prev = Function(self.Vu,name="Previous displacement") 
            #LagrangeInterpolator.interpolate(self.u_prev,tmp)

            tmp = self.d
            self.d = Function(self.Vd,name="Damage") 
            LagrangeInterpolator.interpolate(self.d,tmp)
            tmp = self.d_prev
            self.d_prev = Function(self.Vd,name="Previous Damage") 
            LagrangeInterpolator.interpolate(self.d_prev,tmp)
            tmp = self.d_prev_iter
            self.d_prev_iter = Function(self.Vd,name="Previous damage in staggered minimization")
            LagrangeInterpolator.interpolate(self.d_prev_iter,tmp)
            #tmp = self.dold
            #self.dold = Function(self.Vd,name="Old Damage") 
            #LagrangeInterpolator.interpolate(self.dold,tmp)
            tmp = self.d_ub
            self.d_ub = Function(self.Vd,name="Damage upper bound") 
            #LagrangeInterpolator.interpolate(self.d_ub,tmp)
            self.d_ub = self.mat.dub
            tmp = self.d_lb
            self.d_lb = Function(self.Vd,name="Lower bound d_n") 
            LagrangeInterpolator.interpolate(self.d_lb,tmp)
            self.d_ar= Function(self.Vd,name="Damage field after remeshing") 
            self.d_ar.vector()[:] = self.d.vector()[:]
            #LagrangeInterpolator.interpolate(self.d_ar,self.d)
            tmp = self.metric
            self.metric = Function(self.Vmetric,name="Remeshing metric") 
            LagrangeInterpolator.interpolate(self.metric,tmp)

            #tmp = self.sig
            self.sig = Function(self.Vsig,name="Stress")
            self.eel = Function(self.Vsig,name="ElasticStrain") 
            #LagrangeInterpolator.interpolate(self.sig,tmp)
            #tmp = self.V1
            #self.V1 = Function(self.VV,name="V1") 
            #LagrangeInterpolator.interpolate(self.V1,tmp)           
            #tmp = self.V2
            #self.V2 = Function(self.VV,name="V2") 
            #LagrangeInterpolator.interpolate(self.V2,tmp)  
            #tmp = self.V3
            #self.V3 = Function(self.VV,name="V3") 
            #LagrangeInterpolator.interpolate(self.V3,tmp)  
            #tmp = self.R
            self.R = Function(self.Vr,name="Rotation matrix")
            #LagrangeInterpolator.interpolate(self.R,tmp)  

            self.myBCS.Vu = self.Vu
            self.myBCS.Vd = self.Vd
            self.myBCS.facets = self.facets
            self.bcs, self.bc_d_lb, self.bc_d_ub = self.myBCS.make_bcs(self.dim,self.mat.damage_dim)

            self.myResultant.u = self.u
            self.myResultant.d = self.d
            self.myResultant.P1pos = self.P1pos
            self.myResultant.P2pos = self.P2pos
            self.myResultant.P3pos = self.P3pos
            self.myResultant.ds = self.ds
            
            self.resultant = self.sig[1,1]*self.ds(5)
            #self.resultant = self.mat.sigma(self.u,self.d,self.P1pos,self.P2pos,self.P3pos)[1,1]*self.dsJ[3]
            #self.resultant = self.myResultant.make_resultant() #*Measure("ds", subdomain_data=self.facets)
            
            self.Wext = self.define_external_work()
            self.set_energies()
            self.set_problems()
            
            self.remeshing_index += 1
            self.runtime_remeshing_interpolation += time() - tic_remeshing_interpolation
        else:
            self.remesh = False
            self.set_problems() #self.solver_u.params["user_switch"] = (not self.remesh) #
    
    def sigma(self):
        if (self.mat.behaviour=="linear_elasticity"):
            self.sig = self.mat.sigma(self.u,self.d,self.P1pos,self.P2pos,self.P3pos)
        else:
            flux_names = self.solver_u.material.get_flux_names()
            stress_name = "MissingStress?"
            for name in flux_names:
                if "Stress" in name:
                    stress_name = name
            
            #sig = (1-self.d)**2*self.solver_u.get_flux(stress_name, project_on=("DG", 0), as_tensor=True) # NOK because the (1-d)^2 factor on stored energy must be added in the fenics programm, not the mfront behaviour (seen as variable not a constant)
            sig = self.solver_u.get_flux(stress_name, project_on=("DG", 0), as_tensor=True)
            ''' not needed if as_tensor = True
            if stress_name=="Stress":
                if self.dim==2:
                    sig = as_matrix([[s[0],          s[3]/sqrt(2.), 0.],\
                                     [s[3]/sqrt(2.), s[1],          0.],\
                                     [0.,              0.,        s[2]]])
                else:
                    sig = as_matrix([[s[0],          s[3]/sqrt(2.), s[4]/sqrt(2.)],\
                                     [s[3]/sqrt(2.), s[1],          s[5]/sqrt(2.)],\
                                     [s[4]/sqrt(2.), s[5]/sqrt(2.),          s[2]]])
            elif stress_name == "FirstPiolaKirchhoffStress":
                if self.dim==2:
                    sig = as_matrix([[s[0],          s[3],          0.],\
                                     [s[3],          s[1],          0.],\
                                     [0.,              0.,        s[2]]])
                else:
                    sig = as_matrix([[s[0],          s[3],        s[5]],\
                                     [s[4],          s[1],        s[7]],\
                                     [s[6],          s[8],        s[2]]])
            '''
            #self.sig.vector()[:] = sig.vector()
            #self.sig = Function(self.Vsig,name="Stress")
            self.sig.assign(local_project(sig, self.Vsig))
            #self.sig *= (1.-self.d)**2
    
    def eps_elas(self):
        if (not self.mat.behaviour=="linear_elasticity"):
            e = self.solver_u.get_state_variable("ElasticStrain", project_on=("DG", 0), as_tensor=True)
            ''' not needed if as_tensor = True
            if self.dim==2:
                e = as_matrix([[e[0],             e[3]/sqrt(2.),          0.],\
                                 [e[3]/sqrt(2.),           e[1],          0.],\
                                 [0.,                        0.,        e[2]]])
            else:
                e = as_matrix([[e[0],            e[3]/sqrt(2.),          e[4]/sqrt(2.)],\
                                 [e[3]/sqrt(2.),          e[1],          e[5]/sqrt(2.)],\
                                 [e[4]/sqrt(2.), e[5]/sqrt(2.),                   e[2]]])
            '''
            self.eel.assign(local_project(e, self.Vsig))
    
    def startup_message(self):
        if (self.rank==0):
            version, date = get_version_date(package_name='gradam')
            print(' ##################################################################\n',\
                   '##########                 gradam-%s                 ##########\n' % version,\
                   '##########            Jean-Michel Scherer (C)           ##########\n',\
                   '##########              scherer@caltech.edu             ##########\n',\
                   '##########    Installed on: %s    ##########\n' % date,\
                   '##################################################################\n')
        
    # def user_postprocess(self):
    #     if (self.niter % 10) ==0:
    #         intermediate_loading = self.load_steps[self.incr-1]+self.niter/float(self.staggered_solver["iter_max"])*(self.load_steps[self.incr]-self.load_steps[self.incr-1])
    #         self.export_damage(intermediate_loading/self.final_time)
    #         if self.save_all:
    #             self.export_all(intermediate_loading/self.final_time)
            
def local_project(v,V,u=None):
    dv = TrialFunction(V)
    v_ = TestFunction(V)
    a_proj = inner(dv, v_)*dx
    b_proj = inner(v, v_)*dx
    solver = LocalSolver(a_proj, b_proj)
    solver.factorize()
    if u is None:
        u = Function(V)
        solver.solve_local_rhs(u)
        return u
    else:
        solver.solve_local_rhs(u)
        return

def gaussian_filter(v,sigma):
    values = v.vector()[:].T #recover values of d as numpy array
    values = sp.ndimage.filters.gaussian_filter(values, sigma, mode='constant')
    filtered = Function(v.function_space()) #  generate a new function
    filtered.vector()[:] = values.T
    return filtered

def filter_function(u,v,V,sigma,damage_dim,dim):
    xyz = V.tabulate_dof_coordinates()
    xyz = xyz.reshape((int(len(xyz)/damage_dim),damage_dim,dim))[:,0,:]
    x = xyz[:,0]
    y = xyz[:,1]
    #z = xyz[:,2]
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    Ly, Lx = xmax-xmin, ymax-ymin # Lx and Ly are inverted volontarily
    N = x.shape[0]
    #zmin, zmax = z.min(), z.max()
    #grid_x, grid_y, grid_z = np.mgrid[xmin:xmax:(x.shape*1j), ymin:ymax:(y.shape*1j), zmin:zmax:(z.shape*1j)]
    # xi = range(x.shape[0]) #np.linspace(0,x.shape[0])
    # yi = range(y.shape[0]) #np.linspace(0,y.shape[0])   
    grid_x, grid_y = np.mgrid[xmin:xmax:(int(((xmax-xmin)/(ymax-ymin))*np.sqrt(N*Lx/Ly))*1j),
                              ymin:ymax:(int(((ymax-ymin)/(xmax-xmin))*np.sqrt(N*Ly/Lx))*1j)]
    #grid_x, grid_y = np.mgrid[xmin:xmax:(100*1j), ymin:ymax:(100*1j)]
    field  = u.vector().get_local()
    field  = field.reshape((int(len(field)/damage_dim),damage_dim))
    field  = LA.norm(field,ord=np.inf,axis=1)
    values = u.vector().get_local()
    values  = values.reshape((int(len(values)/damage_dim),damage_dim))
    values  = LA.norm(values,ord=np.inf,axis=1)    
    for (i,value) in enumerate(field):
        if (field[i]>0.5):
            values[i] = max(values[i],values.max()/10.)
    from scipy.interpolate import griddata
    image = griddata(xyz, values, (grid_x, grid_y), method='cubic') #nearest
    image = values.max()*image/image.max()
    image_filtered = sp.ndimage.filters.gaussian_filter(image, sigma, mode='constant')
    image_filtered = values.max()*image_filtered/image_filtered.max()
    # x_coords = {value: index for index, value in list(zip(xi,x))}
    # y_coords = {value: index for index, value in list(zip(yi,y))}
#    plt.imshow(image)
    values_filtered = np.zeros_like(values)
    q,p=image_filtered.shape
    for (i,(xic,yic)) in enumerate(list(zip(x,y))):
        #print(int(np.round(xic*(x.shape[0]-1))),int(np.round(yic*(y.shape[0]-1))))
        values_filtered[i] = image_filtered[min(max(int(np.floor(((xic-xmin)/(ymax-ymin))*np.floor(np.sqrt(N*Lx/Ly)))),0),q-1),\
                                            min(max(int(np.floor(((yic-ymin)/(xmax-xmin))*np.floor(np.sqrt(N*Ly/Lx)))),0),p-1)]
        # values_filtered[i] = image_filtered[int(np.floor(((xic-xmin)/(ymax-ymin))*np.sqrt(N-1))),\
        #                                     int(np.floor(((yic-ymin)/(xmax-xmin))*np.sqrt(N-1)))]
        # if (field[i]==1.):
        #     values_filtered[i] = values.max()
    #values_filtered = image_filtered.T
    values_filtered[values_filtered<values_filtered.max()/100.]=0
#    plt.imshow( griddata(xyz, values_filtered, (grid_x, grid_y), method='cubic') )
    v_filtered = Function(v.function_space().sub(0).collapse()) #  generate a new function
    v_filtered.vector()[:] = values_filtered
    return v_filtered
