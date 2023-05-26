#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 13:58:43 2021

@author: bleyerj
"""
from dolfin import assemble_system, PETScMatrix, PETScVector, as_backend_type, \
                    PETScOptions, MPI
from petsc4py import PETSc
from time import time

class HybridLinearSolver:
    def __init__(self, a, L, u, p=None, bcs=None, parameters={"iteration_switch": 5, "user_switch": True},
                 direct_solver={"solver": "mumps", "type": "ksp", "blr": True},
                 iterative_solver={"solver": "cg"}, log=True, timings=False, null_space_basis=None):
        self.a = a
        self.L = L
        self.u = u
        self.p = p
        self.bcs = bcs
        self.parameters = parameters
        self.direct_solver = direct_solver
        self.iterative_solver = iterative_solver
        self.reuse_preconditioner = False

        self.null_space_basis = null_space_basis

        self.log = log
        self.timings = timings

        self.A = PETScMatrix()
        self.P = PETScMatrix()
        self.b = PETScVector()

        self.ksp = PETSc.KSP().create()

        PETScOptions.set('ksp_type', iterative_solver["solver"])
        PETScOptions.set('pc_factor_mat_solver_type', direct_solver["solver"])
        PETScOptions.set('pc_type', direct_solver["type"])

        PETScOptions.set('ksp_error_if_not_converged')
        PETScOptions.set('ksp_norm_type', 'natural')
        # use Block Low Rank compression
        if direct_solver["solver"]=="mumps" and direct_solver["blr"]:
            PETScOptions.set('mat_mumps_icntl_35', 1)
            PETScOptions.set('mat_mumps_cntl_7', 1e-10)

        self.ksp.setFromOptions()

        self.pc = self.ksp.getPC()
        self.pc.setReusePreconditioner(True)

    def print_log(self, m):
        if self.log and (MPI.rank(MPI.comm_world)==0):
            print(m)
    def print_timings(self, m):
        if self.timings:
            print(m, time()-self.tic)

    def assemble_operators(self):
        self.tic = time()
        assemble_system(self.a, self.L, self.bcs, A_tensor=self.A, b_tensor=self.b)
        self.print_timings("Operator assembly:")

        if (not self.null_space_basis==None):
            as_backend_type(self.A).set_nullspace(self.null_space_basis)
            self.null_space_basis.orthogonalize(self.b)

        if not self.reuse_preconditioner: # preconditioner is updated
            if self.p is not None:        # with a user-specified preconditioner p
                self.tic = time()
                assemble_system(self.p, self.L, self.bcs, A_tensor=self.P, b_tensor=self.b)
                self.print_timings("Preconditioner assembly:")
                self.ksp.setOperators(self.A.mat(), self.P.mat())
            else:                         # with the operator itself (direct solve)
                self.ksp.setOperators(self.A.mat(), self.A.mat())
        else:                             # preconditioner is not updated
            self.ksp.setOperators(self.A.mat())

    def solve(self, b=None, force_direct_solve=False):
        self.assemble_operators()

        self.tic = time()
        if force_direct_solve:
            self.pc.setReusePreconditioner(False)
        self.ksp.setUp()

        if b is None:
            bv = self.b.vec()
        else:
            bv = as_backend_type(b).vec()

        uv =  as_backend_type(self.u.vector()).vec()
        #self.ksp.solve(bv, uv)
        
        try:
            self.ksp.solve(bv, uv)
        except RuntimeError:  # force direct solver if previous solve fails
            print("Error has been catched")
            PETScOptions.set("ksp_type", "preonly")
            self.ksp.setFromOptions()
            self.ksp.solve(bv, uv)
            PETScOptions.set("ksp_type", self.iterative_solver["solver"])
            self.ksp.setFromOptions()
        
        self.print_timings("Solve time:")

        self.u.vector()[:] = uv
        it = self.ksp.getIterationNumber()
        self.print_log("        Converged in  {} iterations.".format(it))
        self.reuse_preconditioner = ((it < self.parameters["iteration_switch"]) and self.parameters["user_switch"])
        self.pc.setReusePreconditioner(self.reuse_preconditioner)
        it_direct = 0
        if self.reuse_preconditioner:
            self.print_log("        Preconditioner will be reused on next solve.")
        else:
            self.print_log("        Next solve will be a direct one with matrix factorization.")
            it_direct = 1
        return [it, it_direct]

