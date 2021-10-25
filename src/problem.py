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
import os
from .remesher import *
from .mesh_converter import *
import types
import scipy as sp
import scipy.ndimage

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
    def __init__(self,mesh,facets,mat,prefix,load = Constant((0.,0.)),Jcontours=[]):
        self.staggered_solver = dict({"iter_max":500,"tol":1e-4,"accelerated":False,"history_function":False})
        self.mesh = mesh
        self.num_vertices = self.mesh.num_vertices()
        self.facets = facets
        self.dim = mesh.geometric_dimension()
        self.mat = mat
        self.prefix = prefix
        if (not os.path.isdir(self.prefix)):
            os.system("mkdir %s" % self.prefix)
#        self.load_steps = load_steps
        self.load = load
        self.bcs = []
        self.bc_d =[]
        self.Uimp = Expression("t", t=0, degree=0)
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
        self.Wext = self.define_external_work()
        self.resultant = self.sig[1,1]*self.ds
        self.comm = MPI.comm_world
        self.rank = MPI.rank(self.comm)        
        self.results = XDMFFile(MPI.comm_world,self.prefix+"output.xdmf")
        self.results.parameters["flush_output"] = True
        self.results.parameters["functions_share_mesh"] = True
        self.checkpoints = XDMFFile(MPI.comm_world,self.prefix+"checkpoint.xdmf")
        self.checkpoints.parameters["flush_output"] = True
        self.checkpoints.parameters["functions_share_mesh"] = True
        self.save_all = True
        self.save_intermediate = False
        self.save_checkpoints = False
        self.write_checkpoint_count = 0
        self.use_hybrid_solver = False
        self.normal = FacetNormal(mesh)
        self.dsJ = Jcontours #[] #self.ds
        self.J = []
        self.remesh = False
        self.use_remeshing = False
        self.remeshing_criterion = 1.e-2
        self.sol_min = 0.01
        self.sol_max = 0.1
        self.remeshing_index = 1
        self.mesh_path = ''
        self.mesh_file = ''
        self.sol_file  = ''
        self.nbgrains  = -1
        self.number_of_nodes_index = 0
        self.boundaries = []
        self.markers = []
        self.domains = []
        self.domains_markers = []
        self.myBCS = self.BCS(self.Vu,self.Vd,self.facets)
        self.myResultant = self.Resultant(self.mat,self.u,self.d,self.P1pos,self.P2pos,self.P3pos,self.ds)
        self.gaussian_filter_sigma = 1.
        self.no_residual_stiffness=[False,0.99]
        
        # if (not Jcontours==None):
        #     for c in Jcontours:
        #         self.dsJ.append(c) 

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
        self.Vu = VectorFunctionSpace(self.mesh, "CG", 1, dim=self.dim)
        if self.mat.damage_dim == 1:
            self.Vd = FunctionSpace(self.mesh, "CG",1)
        else:
            self.Vd = VectorFunctionSpace(self.mesh, "CG",1, dim=self.mat.damage_dim)
        #self.Vd0 = FunctionSpace(self.mesh, "CG",1)
        self.V0 = FunctionSpace(self.mesh, "DG", 0)    
        self.Vsig = TensorFunctionSpace(self.mesh, "CG", 1, shape=(3,3))
        self.VV = VectorFunctionSpace(self.mesh, "DG", 0, dim=3)
        
        self.u_ = TestFunction(self.Vu)
        self.du = TrialFunction(self.Vu)
        self.d_ = TestFunction(self.Vd)
        self.dd = TrialFunction(self.Vd)
        
        # Definition of functions
        self.u = Function(self.Vu,name="Displacement")
        self.u_prev = Function(self.Vu,name="Previous displacement")
        self.d = Function(self.Vd,name="Damage")
        self.d_prev = Function(self.Vd,name="Previous damage")
        self.dold = Function(self.Vd,name="Old damage")
        self.d_lb = Function(self.Vd,name="Lower bound d_n")
        self.d_ub = Function(self.Vd,name="Damage upper bound") #"Upper bound 1")
        self.d_ar= Function(self.Vd,name="Damage field after remeshing") 
        #self.d_list = [Function(self.Vd0,name="Damage-"+str(i)) for i in range(self.mat.damage_dim)]
        # if self.mat.damage_dim == 1:
        #     self.d_ub.interpolate(Constant(1.))
        # else:
        #     self.d_ub.interpolate(Constant((1.,)*self.mat.damage_dim))
        self.d_ub = self.mat.dub
        self.sig = Function(self.Vsig,name="Stress")
        self.epspos = Function(self.Vsig,name="Strain (+)")
        self.epsneg = Function(self.Vsig,name="Strain (-)")
        self.V1 = Function(self.VV,name="V1")
        self.V2 = Function(self.VV,name="V2")
        self.V3 = Function(self.VV,name="V3")
        # if self.staggered_solver["accelerated"]:
        #     self.tmp_u = Function(self.Vu)
        #     self.tmp_d = Function(self.Vd)
        # self.R1 = Function(self.V0)
        # self.R2 = Function(self.V0)
        self.P1pos = Function(self.V0,name="P1pos")
        self.P2pos = Function(self.V0,name="P2pos")
        self.P3pos = Function(self.V0,name="P3pos")
        self.P1pos.interpolate(Constant(1.))
        self.P2pos.interpolate(Constant(1.))
        self.P3pos.interpolate(Constant(1.))
        self.Efrac_field = Function(self.V0,name="Efrac")
        self.Vd0 = FunctionSpace(self.mesh, "CG",1)
        self.d_eq_fiss = Function(self.Vd,name="deq")
        if self.mat.damage_dim==1:
            self.d_eq_fiss.interpolate(Constant((1.)))
        else:
            self.d_eq_fiss.interpolate(Constant((1.,)*self.mat.damage_dim))
            
    def define_external_work(self):
        return dot(self.load,self.u)*self.ds
        
    def set_energies(self):
        # Definition of energy densities            
        # self.Wel = 0.5*inner(self.mat.sigma(self.u,self.d,self.P1pos,self.P2pos),eps(self.u))*self.dx
        self.Wel = 0.5*inner(self.mat.sigma(self.u,self.d,self.P1pos,self.P2pos,self.P3pos),eps(self.u,self.dim))*self.dx
        
        self.Efrac = self.mat.fracture_energy_density(self.d)
        self.Wfrac = sum(self.Efrac)*self.dx
        self.Wtot = self.Wel + self.Wfrac - self.Wext

        # Definition of J integral
        if (self.dim == 2):
            normal3  = as_vector([self.normal[0],self.normal[1],0.])
            sigma_n3 = dot(normal3, self.mat.sigma(self.u,self.d,self.P1pos,self.P2pos,self.P3pos))
            sigma_n  = as_vector([sigma_n3[0],sigma_n3[1]])
        elif (self.dim == 3):
            normal3 = self.normal
            sigma_n = dot(normal3, self.mat.sigma(self.u,self.d,self.P1pos,self.P2pos,self.P3pos))
            
        if (not self.dsJ==[]):
            self.J=[]
            for c in self.dsJ:
                self.J.append( (0.5*inner(self.mat.sigma(self.u,self.d,self.P1pos,self.P2pos,self.P3pos),eps(self.u,self.dim))*self.normal[1] \
                               - inner(sigma_n, grad(self.u)[:,0]) ) * c ) # for outer boundaries
            for c in self.dsJ:
                self.J.append( (0.5*inner(self.mat.sigma(self.u,self.d,self.P1pos,self.P2pos,self.P3pos),eps(self.u,self.dim))*self.normal[0] \
                               - inner(sigma_n, grad(self.u)[:,1]) ) * c ) # for outer boundaries
                # self.J.append( (0.5*inner(self.mat.sigma(self.u,self.d),eps(self.u,self.dim))*self.normal[1] \
                #                - inner(sigma_n, grad(self.u)[:,0]) )('-') * c ) # for inner boundaries
        
        # Definition of energy derivatives
        self.DW_u = derivative(self.Wtot,self.u,self.u_)
        self.D2W_u = derivative(self.DW_u,self.u,self.du)
        self.DW_d = derivative(self.Wtot,self.d,self.d_)
        self.D2W_d = derivative(self.DW_d,self.d,self.dd)

    def set_problems(self):
        # Setting up displacement-part linear solver 
        # LinearVariationalProblem(lhs(self.D2W_u),replace(self.Wext,{self.u:self.u_}), self.u, self.bcs)
        if (self.use_hybrid_solver):
            self.solver_u = HybridLinearSolver(lhs(self.D2W_u),dot(self.load,self.u_)*self.ds,self.u,bcs=self.bcs)
        else:
            if (not self.mat.tension_compression_asymmetry):
                self.problem_u = LinearVariationalProblem(lhs(self.D2W_u),dot(self.load,self.u_)*self.ds,self.u,self.bcs)
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
        self.solver_d.parameters["method"] = "tron" # "gpcg"
        self.solver_d.parameters["line_search"] = "gpcg"
        self.solver_d.parameters["linear_solver"] = "mumps" #'cg", "gltr", "gmres", "nash"
        #self.solver_d.parameters["preconditioner"] = "hypre_amg"
        self.solver_d.parameters["maximum_iterations"] = 5000
        # self.solver_d.parameters["gradient_absolute_tol"] = self.staggered_solver["tol"]
        # self.solver_d.parameters["gradient_relative_tol"] = self.staggered_solver["tol"]
        # self.solver_d.parameters["gradient_t_tol"] = self.staggered_solver["tol"]
        self.solver_d.parameters["gradient_absolute_tol"] = 1.e-4
        self.solver_d.parameters["gradient_relative_tol"] = 1.e-4
        self.solver_d.parameters["gradient_t_tol"] = 1.e-4
            
    def staggered_solve(self):
        DeltaE = 1.
        self.niter = 0
        self.d_prev.assign(self.d)
        
        # boundary conditions for damage problem
        for bc in self.bc_d:
            bc.apply(self.d_lb.vector())
        
        while (DeltaE>self.staggered_solver["tol"]) and (self.niter<self.staggered_solver["iter_max"]): 
            if self.rank == 0:
                print("    Iteration %i "%(self.niter))
            # u-solve : gives u_{n+1}^{pred} from d_{n}
            tic_u = time()
            count = self.solver_u.solve()
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
            self.u_prev.assign(self.u)

            Etot_old = assemble(self.Wtot)
            
            # d-solve : gives d_{n+1}^{pred} from u^{n+1}
            self.dold.assign(self.d)         
            dam_prob = DamageProblem(self.Wtot,self.DW_d,self.D2W_d,self.d)
            tic_d = time()
            self.niter_TAO += self.solver_d.solve(dam_prob, self.d.vector(), self.d_lb.vector(), self.d_ub.vector())[0]
            self.runtime_d += time() - tic_d
            
            # Energy computation
            Etot = assemble(self.Wtot)

            DeltaE = abs(Etot_old/Etot-1)
            
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
        '''
        di = self.d.split(deepcopy = True)
        di_ub = d_ub.split(deepcopy = True)
        for k in range(di):
            d_ = di[k].vector()[:]
            d_ub_ = di_ub[k].vector()[:]
            np.place(di_, d2_>threshold, d1_ub_)
        '''
        return d1, d2
        
    def export_damage(self,t):
        # if self.mat.damage_dim > 1:
        #     #for i in range(self.mat.damage_dim): # ??
        #     self.results.write(self.d,t)
        #         #self.results.write_checkpoint(self.d,self.d.name(),t,append=True)
        #         # self.assigner = FunctionAssigner(self.Vd0, self.Vd.sub(i))
        #         # self.assigner.assign(self.d_list[i], self.d.sub(i))
        #         # self.results.write(self.d_list[i],t)
        #     if (not self.write_checkpoint_count % 10):
        #         self.checkpoints.write_checkpoint(self.d,"Damage",t,append=True)
        #         self.write_checkpoint_count = 1
        # else:
        #     self.results.write(self.d,t)
        #     if (not self.write_checkpoint_count % 10):
        #         self.results.write_checkpoint(self.d,"Damage",t,append=True)
        #         #self.results.write_checkpoint(self.d,self.d.name(),t,append=True)
        #         self.write_checkpoint_count = 1
        #self.write_checkpoint_count += 1
        self.results.write(self.d,t)
        self.results.write(self.d_eq_fiss,t)
        if ((not self.write_checkpoint_count % 10) and self.save_checkpoints):
            self.results.write_checkpoint(self.d,"Damage",t,append=True)
            #self.results.write_checkpoint(self.d,self.d.name(),t,append=True)
            self.write_checkpoint_count = 1
        self.write_checkpoint_count += 1       
        
    
    def export_all(self,t):
        self.results.write(self.u,t)
        self.sig.assign(local_project(self.mat.sigma(self.u,self.d,self.P1pos,self.P2pos,self.P3pos),self.Vsig)) #, solver_type='cg', preconditioner_type='hypre_amg'))
        self.epspos.assign(local_project(dot(self.mat.R.T,dot(voigt2stress(self.mat.eps_crystal_pos),self.mat.R.T)),self.Vsig)) #, solver_type='cg', preconditioner_type='hypre_amg'))
        #self.epsneg.assign(local_project(dot(self.mat.R.T,dot(voigt2stress(self.mat.eps_crystal_neg),self.mat.R.T)),self.Vsig)) #, solver_type='cg', preconditioner_type='hypre_amg'))
        self.results.write(self.sig,t)
        self.results.write(self.epspos,t)
        #self.results.write(self.epsneg,t)
        self.results.write(self.mat.phi1,t)
        self.results.write(self.mat.Phi,t)
        self.results.write(self.mat.phi2,t)
        self.V1.assign(local_project(self.mat.R[0,:],self.VV))
        self.V2.assign(local_project(self.mat.R[1,:],self.VV))
        self.V3.assign(local_project(self.mat.R[2,:],self.VV))
        self.results.write(self.V1,t)
        self.results.write(self.V2,t)
        self.results.write(self.V3,t)
        self.results.write(self.P1pos,t)
        self.results.write(self.P2pos,t)
        self.results.write(self.P3pos,t)
        self.Efrac_field.assign(local_project(sum(self.Efrac),self.V0))
        self.results.write(self.Efrac_field,t)
        if ((not self.write_checkpoint_count % 10) and self.save_checkpoints):
            self.checkpoints.write_checkpoint(self.u,self.u.name(),t,append=True)
        # self.results.write_checkpoint(self.u,self.u.name(),t,append=True)
        # #self.checkpoints.write_checkpoint(self.u,self.u.name(),t,append=True)
        # self.sig.assign(local_project(self.mat.sigma(self.u,self.d),self.Vsig)) #, solver_type='cg', preconditioner_type='hypre_amg'))
        # self.results.write_checkpoint(self.sig,self.sig.name(),t,append=True)
        # self.results.write_checkpoint(self.mat.phi1,self.mat.phi1.name(),t,append=True)
        # self.results.write_checkpoint(self.mat.Phi,self.mat.Phi.name(),t,append=True)
        # self.results.write_checkpoint(self.mat.phi2,self.mat.phi2.name(),t,append=True)
        # self.V1.assign(local_project(self.mat.R[:,0],self.VV))
        # self.V2.assign(local_project(self.mat.R[:,1],self.VV))
        # self.V3.assign(local_project(self.mat.R[:,2],self.VV))
        # self.results.write_checkpoint(self.V1,self.V1.name(),t,append=True)
        # self.results.write_checkpoint(self.V2,self.V2.name(),t,append=True)
        # self.results.write_checkpoint(self.V3,self.V3.name(),t,append=True)
            
        
    def solve(self):
        #log = [[0]*13]
        if (self.rank==0):
            f = open(self.prefix+"results.txt","a")
            Jint_cols = ''
            if (not self.dsJ==[]):
                J_intcols = "[15-%s]_J_integrals " % (15+len(self.J)-1)
            f.write("#1_incr 2_t 3_F 4_Eel 5_Ed 6_Etot 7_niter 8_niter_tot 9_niter_TAO 10_niter_iterative 11_niter_direct 12_runtime 13_runtime_u 14_runtime_d "\
                    + Jint_cols + "[%s-%s]_Efrac_i\n" % (15+len(self.J),15+len(self.J)+self.mat.damage_dim-1))
            f.close()
        self.runtime   = time()
        self.runtime_u = 0.
        self.runtime_d = 0.
        self.set_energies()
        self.set_problems()
    
        while (self.t < self.final_time):
            self.dtime = max(self.dtime*self.desired_dincr/self.max_dincr,self.min_dtime)
            self.dtime = min(self.dtime,self.max_dtime)
            if ((self.t + self.dtime) > self.final_time):
                self.dtime = self.final_time - self.t
            if self.rank == 0:
                print( "Increment %i | Loading : %.5e"%(self.incr,self.t+self.dtime))
            if self.remesh == False:
                self.load.t = self.t + self.dtime
                self.Uimp.t = self.t + self.dtime
            
            self.staggered_solve()
            
            #
            # Apply remeshing: this needs cleaning work (use auxiliary functions, remesher class?)
            if (self.use_remeshing):
                diffuse = Function(self.Vd.sub(0).collapse(),name="Diffused damage")
                self.metric_field = np.zeros(self.num_vertices) # diffuse.compute_vertex_values()
                for k in range(self.mat.damage_dim):
                    diffuse = diffusion(self.d.sub(k),self.Vd.sub(k).collapse())
                    self.metric_field = np.maximum(self.metric_field, diffuse.compute_vertex_values()) #element wise maximum #or # += diffuse.compute_vertex_values() #sum
                mini, maxi = self.metric_field.min(), self.metric_field.max()
                self.metric_field = (self.metric_field - mini)/(max(maxi - mini,1.e-6))
                #self.d_var = np.array(self.d.compute_vertex_values()-self.d_prev.compute_vertex_values())
                self.d_var = Function(self.Vd,name="Damage variation")
                self.d_var.vector()[:] = self.d.vector()-self.d_ar.vector()
                
                
                #image_mesh = UnitSquareMesh(100, 100)
                #image_space = FunctionSpace(image_mesh, 'CG', 1)
                #self.d_var = interpolate(self.d_var,image_space)
                #self.d_var_smooth = interpolate(self.d_var_smooth,self.Vd)
                
                #self.d_var_image = function2image(self.d_var,self.Vd,self.gaussian_filter_sigma)
                #self.d_var_image_smooth = sp.ndimage.filters.gaussian_filter(self.d_var_image, self.gaussian_filter_sigma, mode='constant')
                #self.d_var_smooth = gaussian_filter(self.d_var, self.gaussian_filter_sigma) #diffusion(self.d_var,self.Vd)
                
                #if (self.metric_field.max() > self.remeshing_criterion):
                if (np.array(self.d_var.compute_vertex_values()).max() > self.remeshing_criterion):
                    self.remesh = True
                    self.d_var_smooth = filter_function(self.d,self.d_var,self.Vd,self.gaussian_filter_sigma,\
                                                        self.mat.damage_dim,self.dim)
                    #self.metric_field = np.array(self.d_var_smooth.compute_vertex_values())
                    #self.metric_field = np.array(diffuse.compute_vertex_values())
                    write_dvar_sol(self.mat.damage_dim,self.mesh_path+self.mesh_file,\
                                   self.mesh_path+self.sol_file,self.number_of_nodes_index,\
                                   self.metric_field,self.sol_min,self.sol_max,self.remeshing_index)
                    geo_tmpl = 'mmg_tmp' 
                    remesh(self.dim,self.mesh_path,geo_tmpl,self.mesh_file,\
                           self.nbgrains,self.remeshing_index)
                    
                    xdmf   = Meshio_msh2xdmf(self.dim,self.mesh_path+self.mesh_file+'_remeshed_%s' % self.remeshing_index,\
                                             extras='')
                    xdmf.write_xdmf_mesh()
                    xdmf.read_xdmf_mesh()
                    self.mesh = xdmf.mesh
                    self.num_vertices = self.mesh.num_vertices()
                    self.mf = xdmf.mf
                    self.facets = xdmf.facets
                    self.dx = xdmf.dx
                    #self.ds = Measure('ds')
                    self.ds = Measure("ds", subdomain_data=self.mf)
                    self.mat = EXD(self.mat.dim,self.mat.damage_dim,self.mat.mp,\
                                   self.mesh,self.mf,self.mat.geometry,damage_model=self.mat.damage_model)
                    self.normal = FacetNormal(self.mesh)
                        
                    for (boundary,marker) in list(zip(self.boundaries,self.markers)):
                        boundary().mark(self.facets, marker)
                    if (not self.dsJ==[]):
                        mfJ = MeshFunction("size_t", self.mesh, 1)
                        dsj = Measure("ds", subdomain_data=mfJ)
                        #dSJ = Measure("dS", subdomain_data=mfJ)
                        for (i,(Jcontour,Jmarker)) in enumerate(list(zip(self.jcontours,self.jmarkers))):
                            Jcontour().mark(mfJ, Jmarker)
                            self.dsJ[i] = dsj(Jmarker)
                    
                    # file = File("2D_precracked_plate/tmp/mesh"+"_facets%s.pvd" % self.remeshing_index)
                    # file << self.facets
                    
                    # Re-Definition of functions spaces
                    self.Vu = VectorFunctionSpace(self.mesh, "CG", 1, dim=self.dim)
                    if self.mat.damage_dim == 1:
                        self.Vd = FunctionSpace(self.mesh, "CG",1)
                    else:
                        self.Vd = VectorFunctionSpace(self.mesh, "CG",1, dim=self.mat.damage_dim)
                    #self.Vd0 = FunctionSpace(self.mesh, "CG",1)
                    self.V0 = FunctionSpace(self.mesh, "DG", 0)    
                    self.Vsig = TensorFunctionSpace(self.mesh, "CG", 1, shape=(3,3))
                    self.VV = VectorFunctionSpace(self.mesh, "DG", 0, dim=3)
                        
                    self.u_ = TestFunction(self.Vu)
                    self.du = TrialFunction(self.Vu)
                    self.d_ = TestFunction(self.Vd)
                    self.dd = TrialFunction(self.Vd)
                    
                    # Interpolation of functions onto the new function spaces
                    self.u.set_allow_extrapolation(True)
                    self.u = interpolate(self.u,self.Vu)
                    self.u.rename("Displacement","label")
                    self.u_prev.set_allow_extrapolation(True)
                    self.u_prev = interpolate(self.u_prev,self.Vu)
                
                    self.d.set_allow_extrapolation(True)
                    self.d = interpolate(self.d,self.Vd)
                    self.d.rename("Damage","label")
                    self.d_prev.set_allow_extrapolation(True)                    
                    self.d_prev = interpolate(self.d_prev,self.Vd)
                    self.dold.set_allow_extrapolation(True)
                    self.dold = interpolate(self.dold,self.Vd)
                    self.d_ub.set_allow_extrapolation(True)
                    self.d_ub = interpolate(self.d_ub,self.Vd)
                    self.d_lb.set_allow_extrapolation(True)
                    self.d_lb = interpolate(self.d_lb,self.Vd)
                    self.d_ar.set_allow_extrapolation(True)
                    self.d_ar = interpolate(self.d,self.Vd)

                    self.sig.set_allow_extrapolation(True)
                    self.sig = interpolate(self.sig,self.Vsig)
                    self.sig.rename("Stress","label")
                    self.V1.set_allow_extrapolation(True)
                    self.V1 = interpolate(self.V1,self.VV)
                    self.V1.rename("V1","label")
                    self.V2.set_allow_extrapolation(True)
                    self.V2 = interpolate(self.V2,self.VV)
                    self.V2.rename("V2","label")
                    self.V3.set_allow_extrapolation(True)
                    self.V3 = interpolate(self.V3,self.VV)
                    self.V3.rename("V3","label")
                                      
                    self.myBCS.Vu = self.Vu
                    self.myBCS.Vd = self.Vd
                    self.myBCS.facets = self.facets
                    self.bcs, self.bc_d = self.myBCS.make_bcs(self.dim,self.mat.damage_dim)

                    self.myResultant.u = self.u
                    self.myResultant.d = self.d
                    self.myResultant.P1pos = self.P1pos
                    self.myResultant.P2pos = self.P2pos
                    self.myResultant.P3pos = self.P3pos
                    self.myResultant.ds = self.ds
                    self.resultant = self.mat.sigma(self.u,self.d,self.P1pos,self.P2pos,self.P3pos)[1,1]*self.ds
                    #self.resultant =  self.myResultant.make_resultant()
                    
                    self.Wext = self.define_external_work()
                    self.set_energies()
                    self.set_problems()
                    
                    self.remeshing_index += 1
                else:
                    self.remesh = False

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
    
    #            self.sig.assign(project(self.mat.sigma(self.u,self.d,self.P1pos,self.P2pos),self.Vsig, solver_type='cg', preconditioner_type='hypre_amg'))
    #            self.sig.assign(project(self.mat.sigma(self.u,self.d),self.Vsig, solver_type='cg', preconditioner_type='hypre_amg'))
    #            self.results.write(self.u,t/self.final_time)
    #            self.results.write(self.sig,t/self.final_time)
                F = assemble(self.resultant)
                Eel = assemble(self.Wel)
                Ed = assemble(self.Wfrac)
                Etot = assemble(self.Wtot)
                Efrac = [assemble(Efrac_i*self.dx) for Efrac_i in self.Efrac]
                Jint=[]
                if (not self.J==[]):
                    for j in self.J:
                        Jint.append( assemble(j) )
                log = ([self.incr,self.t,F,Eel,Ed,Etot,self.niter,self.niter_tot, \
                        self.niter_TAO,self.niter_iterative,self.niter_direct,time()-self.runtime,self.runtime_u,self.runtime_d] + Jint + Efrac)
                #print(str(log))
                f = open(self.prefix+"results.txt","a")
                #np.savetxt(f,log)
                if (self.rank==0):
                    f.write(' '.join(map(str, log))+"\n")
                logn = np.array(log)
                    
                self.incr += 1
                self.runtime = time() - self.runtime  
        if (self.rank==0):
             f.write("# elapsed time in solve()   = %.3f seconds\n" % self.runtime)
             f.write("# elapsed time in solve_u() = %.3f seconds\n" % self.runtime_u)
             f.write("# elapsed time in solve_d() = %.3f seconds\n" % self.runtime_d)
    
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
    
    
def diffusion(v,V,u=None):
    # image manip
    dv = TrialFunction(V)
    v_ = TestFunction(V)
    dt = .001 #.01
    a = dv*v_*dx + dt*dot(grad(dv), grad(v_))*dx
    L = inner(v, v_)*dx # Constant(0.)*v_*dx # inner(v, v_)*dx
    # # setup solver
    # a = assemble(a)
    # b = assemble(l)
    u = Function(V)
    solve(a == L, u)
    return u
    # solver = KrylovSolver(a, 'cg', 'amg')
    # if u is None:
    #     u = Function(V)
    #     solver.solve(u,b)
    #     return u
    # else:
    #     solver.solve(u,b)
    #     return
