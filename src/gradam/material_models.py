#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Last edited 2021-02-10

Anisotropic polycrystal multiphase field: material models,
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
from .mfront_behaviour import *
import numpy as np
from ufl import sign
import ufl
import mgis.fenics as mf

parameters["form_compiler"]["representation"] = 'uflacs'
ufl.algorithms.apply_derivatives.CONDITIONAL_WORKAROUND = True # allows to use a TrialFunction in conditional( ) for spectral_decomposition

# residual stiffness
kres = Constant(1e-9)

def eps(v,dim):
    e = sym(grad(v))
    if (dim == 2):
        return as_matrix([[e[0,0],e[0,1],0.],[e[1,0],e[1,1],0.],[0.,0.,0.]])
    else:
        return e
def strain2voigt(e):
    #return as_vector([e[0,0],e[1,1],e[2,2],2*e[0,1],2*e[1,2],2*e[2,0]])
    return as_vector([e[0,0],e[1,1],e[2,2],2*e[1,2],2*e[0,2],2*e[0,1]])
def voigt2stress(s):
    return as_tensor([[s[0], s[5], s[4]],[s[5], s[1], s[3]],[s[4], s[3], s[2]]])
def voigt2strain(e):
    return as_tensor([[e[0], e[5]/2., e[4]/2.],[e[5]/2., e[1], e[3]/2.],[e[4]/2., e[3]/2., e[2]]])

ppos = lambda x: (abs(x)+x)/2.
pneg = lambda x: x-ppos(x)
Heav = lambda x: (sign(x)+1.)/2.
#Heav = lambda x: ((x/sqrt(x**2+1.e-30))+1.)/2.

class EXD:
    """ Elastic Crystal Damage (EXD) model with X variables and associated fracture energies """ 
    def __init__(self,dim,damage_dim,material_parameters,mesh,mf,geometry,behaviour="linear_elasticity",\
                 mfront_behaviour=None,damage_model="AT1",anisotropic_elasticity="cubic",damage_tensor=[False,0,0],orientation='from_euler'):
        self.mp = material_parameters
        self.dim = dim
        self.damage_dim = damage_dim
        self.damage_model = damage_model
        self.anisotropic_elasticity = anisotropic_elasticity
        self.mesh = mesh
        self.mf = mf
        self.geometry = geometry
        if (self.anisotropic_elasticity=="cubic"):
            E=self.mp["E"]
            nu=self.mp["nu"]
            G=self.mp["G"]
            self.moduli=list(zip(E,nu,G)) 
        elif (self.anisotropic_elasticity=="orthotropic"):
            E1,E2,E3=self.mp["E1"],self.mp["E2"],self.mp["E3"]
            nu12,nu21,nu13,nu31,nu23,nu32=self.mp["nu12"],\
                self.mp["nu21"],self.mp["nu13"],self.mp["nu31"],self.mp["nu23"],self.mp["nu32"]
            G12,G13,G23=self.mp["G12"],self.mp["G13"],self.mp["G23"]
            self.moduli=list(zip(E1,E2,E3,nu12,nu21,nu13,nu31,nu23,nu32,G12,G13,G23))
        if (self.damage_dim>0):
            if (not "static_phase_field" in self.mp):
                Gc_ = self.mp["Gc"]
                l0_ = self.mp["l0"]
                dub_= self.mp["dub"]
                self.Gc,self.l0,self.dub = make_fracture_properties_per_domain(self.dim,self.mesh,self.mf,self.damage_dim,Gc_,l0_,dub_)
            else:
                Gcbulk_ = self.mp["Gcbulk"]
                Gcboundary_ = self.mp["Gcboundary"]
                l0_ = self.mp["l0"]
                dub_= self.mp["dub"]
                static_phase_field_ = self.mp["static_phase_field"]
                self.Gc,self.l0,self.dub = make_fracture_properties_static_phase_field(self.dim,self.mesh,self.mf,self.damage_dim,Gcbulk_,Gcboundary_,l0_,dub_,static_phase_field_)
        #if ("B_crystal" in self.mp): 
        #    self.B_crystal = self.mp["B_crystal"]
        #el
        if ("alpha" in self.mp and "M" in self.mp):
            self.alpha = self.mp["alpha"]
            self.M = self.mp["M"]
            self.damage_induced_anisotropy = self.mp["damage_induced_anisotropy"]
        self.damage_tensor = damage_tensor # only implemented in 2D for orthogonal cleavage planes
        self.anisotropic_degradation = False
        if self.damage_tensor[0]:
            self.anisotropic_degradation = True
        if ("D_crystal" in self.mp):
            self.D_crystal = self.mp["D_crystal"]
            self.anisotropic_degradation = True
        self.tension_compression_asymmetry = False
        self.spheric_deviatoric_decomposition = [False]
        self.cleavage_planes_decomposition = False
        self.spectral_decomposition = False
        self.orientation = orientation
        self.C, self.R, self.phi1, self.Phi, self.phi2 = self.setup_behaviour()
        self.behaviour = behaviour
        self.mfront_behaviour = mfront_behaviour
        if (not self.behaviour=='linear_elasticity'):
            mat_prop = {}
            #if (self.anisotropic_elasticity=="cubic"):
                #mat_prop = {"YoungModulus": self.E, "PoissonRatio": self.nu} #,\
                #            #, "ShearModulus": self.G,\
            ##                 "YieldStrength": 1000., "HardeningSlope": 0.}
            ## elif (self.anisotropic_elasticity=="orthotropic"):
            ##     mat_prop = {#"YoungModulus1": self.E1, "YoungModulus2": self.E2, "YoungModulus3": self.E3,\
            ##                 #"PoissonRatio12": self.nu12, "PoissonRatio13": self.nu13, "PoissonRatio23": self.nu23,\
            ##                 #"ShearModulus12": self.G12, "ShearModulus13": self.G13, "ShearModulus23": self.G23,\
            ##                 "YieldStrength": 1000., "HardeningSlope": 0.}
            self.mfront_behaviour.set_material_properties(self.dim,self.mesh,self.mf,mat_prop)
            self.mfront_behaviour.set_rotation_matrix(self.R.T)
            

    def setup_behaviour(self):
        if (self.anisotropic_elasticity=="cubic"):
            C, self.y1111, self.y1122, self.y1212, self.E, self.nu, self.G = \
                make_cubic_elasticity_stiffness_tensor(self.dim,self.moduli,self.mesh,self.mf)
        elif (self.anisotropic_elasticity=="orthotropic"):
            C, self.E1, self.E2, self.E3, self.nu12, self.nu21, self.nu13, self.nu31, self.nu23, self.nu32,\
                self.G12, self.G13, self.G23 = make_orthotropic_elasticity_stiffness_tensor(self.dim,self.moduli,self.mesh,self.mf)
        if (self.anisotropic_degradation and ("D_crystal" in self.mp)):
            self.Cdam = []
            for n in range(self.damage_dim):
                D_array = np.array( self.D_crystal[n] ) 
                D = full_3x3x3x3_to_Voigt_6x6( D_array )
                self.Cdam.append( dot(dot(as_matrix(D),C),as_matrix(D)) )  
        
        # compute rotation matrix for the set of euler angles associated to the mesh
        #if self.orientation=='from_euler':
        R, phi1, Phi, phi2 = make_rotation_matrix_from_euler(self.dim, self.geometry, self.mesh, self.mf, self.orientation)
        #if self.geometry.endswith("single_crystal"):
        #elif self.orientation=='from_vector':
        #    R, v1, v2, v3  = make_rotation_matrix_from_V1V2V3(self.dim, self.geometry, self.mesh, self.mf)    
        return C, R, phi1, Phi, phi2  

    def sigma0(self,e):
        return dot(self.R.T,dot(voigt2stress(dot(self.C, e)),self.R))
    
    def sigma(self,v,d,P1,P2,P3):
        
        if (self.tension_compression_asymmetry):
            self.strain_decomposition(v,P1,P2,P3)
        else:
            self.eps_crystal_pos = strain2voigt(dot(self.R,dot(eps(v,self.dim),self.R.T)))
            #self.eps_crystal_neg = strain2voigt(dot(self.R,dot(as_tensor(np.zeros((3,3))),self.R.T)))
            self.eps_crystal_neg = 0.*self.eps_crystal_pos

        if (self.dim==2 and self.damage_dim==2 and self.anisotropic_degradation and self.damage_tensor[0]):
            q = self.damage_tensor[1]
            p = self.damage_tensor[2]
            degrad_type = self.damage_tensor[3]

            if (degrad_type == 'Lorentz'):            
                gamma = self.damage_tensor[4]
                r = self.damage_tensor[5]
                #d1 = ((1-d[0])/(1+gamma*d[0]))**q*((1-d[1])/(1+gamma*d[1]))**r + kres
                #d2 = ((1-d[1])/(1+gamma*d[1]))**q*((1-d[0])/(1+gamma*d[0]))**r + kres
                #dd = ((1-d[0])/(1+gamma*d[0]))**p*((1-d[1])/(1+gamma*d[1]))**p + kres
                d1 = ((1-d[0])/(1+gamma*d[0]))**q + kres
                d2 = ((1-d[1])/(1+gamma*d[1]))**q + kres
                dd = ((1-d[0])/(1+gamma*d[0]))**p*((1-d[1])/(1+gamma*d[1]))**p + kres
                #d1 = ((1-d[0])/(1+gamma*d[0])) + kres
                #d2 = ((1-d[1])/(1+gamma*d[1])) + kres
                #dd = ((1-d[0])/(1+gamma*d[0]))*((1-d[1])/(1+gamma*d[1])) + kres
                #d1 = ((1-d[0])) + kres
                #d2 = ((1-d[1])) + kres
                #dd = ((1-d[0]))*((1-d[1])) + kres
            elif (degrad_type=='tan'):
                gamma = self.damage_tensor[4]
                d1 = ((0.5/tan(gamma/2.))*tan(-gamma*(d[0]-0.5)) + 0.5)**q
                d2 = ((0.5/tan(gamma/2.))*tan(-gamma*(d[1]-0.5)) + 0.5)**q
                dd = ((0.5/tan(gamma/2.))*tan(-gamma*(d[0]-0.5)) + 0.5)**p*((0.5/tan(gamma/2.))*tan(-gamma*(d[1]-0.5)) + 0.5)**p
            else:
                d1 = (1-d[0])**q + kres
                d2 = (1-d[1])**q + kres
                dd = (1-d[0])**p*(1-d[1])**p + kres

            iD0 = as_tensor([[d1,0,0,0,0,0],[0,d2,0,0,0,0],[0,0,0,0,0,0],\
                             [0,0,0,dd,0,0],[0,0,0,0,dd,0],[0,0,0,0,0,dd]])
            degraded_stiffness = dot(iD0,dot(self.C,iD0))

            return dot(self.R.T,dot(voigt2stress(dot(degraded_stiffness,self.eps_crystal_pos)),self.R)) +\
                   self.sigma0(self.eps_crystal_neg)

        if (self.dim==3 and self.damage_dim==3 and self.anisotropic_degradation and self.damage_tensor[0]):
            q = self.damage_tensor[1]
            p = self.damage_tensor[2]
            degrad_type = self.damage_tensor[3]

            if (degrad_type == 'Lorentz'):            
                gamma = self.damage_tensor[4]
                r = self.damage_tensor[5]
                d1 = ((1-d[0])/(1+gamma*d[0]))**q + kres
                d2 = ((1-d[1])/(1+gamma*d[1]))**q + kres
                d3 = ((1-d[2])/(1+gamma*d[2]))**q + kres
                dd = ((1-d[0])/(1+gamma*d[0]))**p*((1-d[1])/(1+gamma*d[1]))**p*((1-d[2])/(1+gamma*d[2]))**p + kres
            elif (degrad_type=='tan'):
                gamma = self.damage_tensor[4]
                d1 = ((0.5/tan(gamma/2.))*tan(-gamma*(d[0]-0.5)) + 0.5)**q
                d2 = ((0.5/tan(gamma/2.))*tan(-gamma*(d[1]-0.5)) + 0.5)**q
                d3 = ((0.5/tan(gamma/2.))*tan(-gamma*(d[2]-0.5)) + 0.5)**q
                dd = ((0.5/tan(gamma/2.))*tan(-gamma*(d[0]-0.5)) + 0.5)**p*\
                     ((0.5/tan(gamma/2.))*tan(-gamma*(d[1]-0.5)) + 0.5)**p*\
                     ((0.5/tan(gamma/2.))*tan(-gamma*(d[2]-0.5)) + 0.5)**p
            else:
                d1 = (1-d[0])**q + kres
                d2 = (1-d[1])**q + kres
                d3 = (1-d[2])**q + kres
                dd = (1-d[0])**p*(1-d[1])**p*(1-d[2])**p + kres

            iD0 = as_tensor([[d1,0,0,0,0,0],[0,d2,0,0,0,0],[0,0,d3,0,0,0],\
                             [0,0,0,dd,0,0],[0,0,0,0,dd,0],[0,0,0,0,0,dd]])
            degraded_stiffness = dot(iD0,dot(self.C,iD0))

            return dot(self.R.T,dot(voigt2stress(dot(degraded_stiffness,self.eps_crystal_pos)),self.R)) +\
                   self.sigma0(self.eps_crystal_neg)

        if (self.damage_dim==0):
            return self.sigma0(self.eps_crystal_pos) + self.sigma0(self.eps_crystal_neg)
            #return self.sigma0(self.eps_crystal_neg)
        elif (self.damage_dim==1):
            return ((1.-d)**2 + kres)*self.sigma0(self.eps_crystal_pos) + self.sigma0(self.eps_crystal_neg)
            #return self.sigma0(self.eps_crystal_neg)
        else:
            if (self.anisotropic_degradation):
                g = []
                for n in range(self.damage_dim):
                    g.append( (1.-d[n])**2 )
                q = []
                for (i, gi) in enumerate(g):
                    #q.append( (1-gi) )
                    q.append( np.prod(g[:i]+g[i+1:])*(1-gi) )
                    
                degraded_stiffness = (np.prod(g) + kres)*self.C
                for n in range(self.damage_dim):
                    degraded_stiffness += q[n]*self.Cdam[n]
                return dot(self.R.T,dot(voigt2stress(dot(degraded_stiffness,\
                       self.eps_crystal_pos)),self.R)) +\
                       self.sigma0(self.eps_crystal_neg)
            else:
                g = 1.
                for n in range(self.damage_dim):
                    g*=(1.-d[n])**2
                return (g+kres)*self.sigma0(self.eps_crystal_pos) + self.sigma0(self.eps_crystal_neg)
                #return self.sigma0(self.eps_crystal_neg)        
        
    def strain_decomposition(self,v,P1,P2,P3):
        e = eps(v,self.dim)
        if (self.spheric_deviatoric_decomposition[0]==True):
            trace = tr(e)
            if (self.spheric_deviatoric_decomposition[1]=='Amor'):
                self.eps_crystal_pos = strain2voigt(e - (1./3.)*Heav(-trace)*trace*Identity(3))
                self.eps_crystal_neg = strain2voigt((1./3.)*Heav(-trace)*trace*Identity(3))
            else:
                self.eps_crystal_pos = strain2voigt(e - (1./3.)*trace*Identity(3))
                self.eps_crystal_neg = strain2voigt((1./3.)*trace*Identity(3))
        elif (self.spectral_decomposition and self.dim==2):
            def eig_plus(A): 
                return (tr(A) + sqrt(tr(A)**2-4*det(A)))/2
            def eig_minus(A): 
                return (tr(A) - sqrt(tr(A)**2-4*det(A)))/2
            # Diagonal matrix with positive and negative eigenvalues as the diagonal elements    
            def diag_eig(A):
                lambdap1 = 0.5*(eig_plus(A)+ abs(eig_plus(A)))
                lambdap2 = 0.5*(eig_minus(A) + abs(eig_minus(A)))
                lambdan1 = 0.5*(eig_plus(A) - abs(eig_plus(A)))
                lambdan2 = 0.5*(eig_minus(A) - abs(eig_minus(A)))
                matp = as_matrix(((lambdap1,0),(0,lambdap2)))
                matn = as_matrix(((lambdan1,0),(0,lambdan2)))
                return (matp,matn)    
            # Eigenvectors of the matrix arranged in columns of matrix     
            def eig_vecmat(A):
                 lambdap = eig_plus(A)
                 lambdan = eig_minus(A)
                 a = A[0,0]
                 b = A[0,1]
                 c = A[1,0]
                 d = A[1,1]
                 v11 = lambdap - d
                 v12 = lambdan - d
                 nv11 = sqrt(v11**2 + c**2)
                 nv12 = sqrt(v12**2 + c**2)
                 a1 = v11/nv11
                 b1 = v12/nv12
                 c1 = c/nv11
                 d1 = c/nv12
                 v21 = lambdap - a
                 v22 = lambdan - a
                 nv21 = sqrt(v21**2 + b**2)
                 nv22 = sqrt(v22**2 + b**2)
                 A1 = b/nv21
                 B1 = b/nv22
                 C1 = v21/nv21
                 D1 = v22/nv22
                 tol = 1.e-10
                 #if (gt(abs(c),tol)):
                 #    Eigvecmat = as_matrix(((a1,b1),(c1,d1)))
                 #else:
                 #    if (gt(abs(b),tol)):
                 #        Eigvecmat = as_matrix(((A1,B1),(C1,D1)))
                 #    else:
                 #        Eigvecmat = Identity(2)
                 Eigvecmat = conditional(gt(abs(c), tol) ,as_matrix(((a1,b1),(c1,d1))), conditional(gt(abs(b), tol),as_matrix(((A1,B1),(C1,D1))),Identity(2)))
                 return Eigvecmat
            def eps_split(A):
                P = eig_vecmat(A)
                (diag_eigp,diag_eign) = diag_eig(A)
                epsp = dot(P,dot(diag_eigp,P.T))
                epsn = dot(P,dot(diag_eign,P.T))
                epsp = strain2voigt(as_matrix([[epsp[0,0],epsp[0,1],0.],[epsp[1,0],epsp[1,1],0.],[0.,0.,0.]]))
                epsn = strain2voigt(as_matrix([[epsn[0,0],epsn[0,1],0.],[epsn[1,0],epsn[1,1],0.],[0.,0.,0.]]))
                return (epsp,epsn)  
            self.eps_crystal_pos, self.eps_crystal_neg = eps_split( as_matrix([[e[0,0],e[0,1]],[e[1,0],e[1,1]]]) )
        elif (self.cleavage_planes_decomposition):
            self.eps_crystal = dot(self.R,dot(e,self.R.T)) #eps(v,self.dim)
            self.eps_crystal_pos = as_tensor(np.zeros((3,3)))
            self.eps_crystal_neg = as_tensor(np.zeros((3,3)))
            self.n = [as_tensor([1.,0.,0.]),as_tensor([0.,1.,0.]),as_tensor([0.,0.,1.])]
            self.epsilon = []
            for i in range(3):
                #self.n.append(self.R[:,i])
                self.epsilon.append(dot(dot(self.eps_crystal,self.n[i]),self.n[i]))
                #print(self.n[i],self.epsilon[i])
                self.eps_crystal_pos += ppos(self.epsilon[i])*outer(self.n[i],self.n[i])
                self.eps_crystal_neg += pneg(self.epsilon[i])*outer(self.n[i],self.n[i])
            self.eps_crystal_pos = strain2voigt(self.eps_crystal_pos)
            self.eps_crystal_neg = strain2voigt(self.eps_crystal_neg)
        else:
            self.eps_crystal_pos = dot(as_tensor([[P1,0,0,0,0,0],[0,P2,0,0,0,0],[0,0,P3,0,0,0],\
                                                  [0,0,0,1.,0,0],[0,0,0,0,1.,0],[0,0,0,0,0,1.]]),strain2voigt(dot(self.R,dot(e,self.R.T))))
            self.eps_crystal_neg = dot(as_tensor([[1.-P1,0,0,0,0,0],[0,1.-P2,0,0,0,0],[0,0,1.-P3,0,0,0],\
                                                  [0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]]),strain2voigt(dot(self.R,dot(e,self.R.T)))) 
        # self.eps_crystal = dot(self.R,dot(e,self.R.T)) #eps(v,self.dim)
        # self.eps_crystal_pos = as_tensor(np.zeros((3,3)))
        # self.eps_crystal_neg = as_tensor(np.zeros((3,3)))
        # self.n = []
        # self.epsilon = []
        # for i in range(3):
        #     self.n.append(self.R[:,i])
        #     self.epsilon.append(dot(dot(self.eps_crystal,self.n[i]),self.n[i]))
        #     print(self.n[i],self.epsilon[i])
        #     self.eps_crystal_pos += ppos(self.epsilon[i])*outer(self.n[i],self.n[i])
        #     self.eps_crystal_neg += pneg(self.epsilon[i])*outer(self.n[i],self.n[i])

    def make_B_sample(self,d,d_prev_iter):
        if ("alpha" in self.mp and "M" in self.mp):
            I = np.eye(3)
            if (self.damage_dim==1):
                if (self.damage_induced_anisotropy==True):
                    k_dam = d_prev_iter
                else:
                    k_dam = 1.0
                self.B_crystal = as_tensor(I) + k_dam*self.alpha*(as_tensor(I) - outer(self.M,self.M))
            else:
                self.B_crystal = []
                for n in range(self.damage_dim):
                    if (self.damage_induced_anisotropy==True):
                        k_dam = d_prev_iter.sub(n)**2
                    else:
                        k_dam = 1.0
                    self.B_crystal.append( as_tensor(I) + k_dam*self.alpha*(as_tensor(I) - outer(self.M[n],self.M[n])) )
        else:
            self.B_crystal = as_tensor( np.eye(3) )

        if (self.damage_dim==1):
            self.B_sample = dot(self.R.T,dot(self.B_crystal,self.R))
            if (self.dim==2):
                self.B_sample = as_tensor([[self.B_sample[0,0], self.B_sample[0,1]],
                                           [self.B_sample[1,0], self.B_sample[1,1]]])
        else:
            self.B_sample = []
            for n in range(self.damage_dim):
                self.B_sample.append(dot(self.R.T,dot(self.B_crystal[n],self.R)))
                if (self.dim==2):
                    self.B_sample[n] = as_tensor([[self.B_sample[n][0,0], self.B_sample[n][0,1]],
                                                  [self.B_sample[n][1,0], self.B_sample[n][1,1]]])
                
    def fracture_energy_density(self,d,d_prev_iter):
        if (self.damage_dim==0):
            return [0.]
        else:
            if self.damage_model == "AT1":
                cw = Constant(8/3.)
                w = lambda d: d
            elif self.damage_model == "AT2":
                cw = Constant(2.)
                w = lambda d: d**2
            self.make_B_sample(d,d_prev_iter)
            if (self.damage_dim==1):
    #            self.B_sample = dot(self.R.T,dot(self.B_crystal,self.R))
    #            if (self.dim==2):
    #                self.B_sample = as_tensor([[self.B_sample[0,0], self.B_sample[0,1]],
    #                                           [self.B_sample[1,0], self.B_sample[1,1]]])
                return [self.Gc/cw*(w(d)/self.l0+self.l0*dot(dot(self.B_sample, grad(d)),grad(d)))]
            else:
                Efrac = []
    #            self.B_sample = []
                for n in range(self.damage_dim):
    #                self.B_sample.append(dot(self.R.T,dot(self.B_crystal[n],self.R)))
    #                if (self.B_crystal == as_tensor(np.eye(3))):
    #                    self.B_sample = as_tensor(np.eye(3))
    #                if (self.dim==2):
    #                    self.B_sample[n] = as_tensor([[self.B_sample[n][0,0], self.B_sample[n][0,1]],
    #                                                  [self.B_sample[n][1,0], self.B_sample[n][1,1]]])
                    Efrac.append( self.Gc[n]/cw*(w(d[n])/self.l0[n]+self.l0[n]*dot(dot(self.B_sample[n], grad(d[n])),grad(d[n]))) )
                return Efrac
