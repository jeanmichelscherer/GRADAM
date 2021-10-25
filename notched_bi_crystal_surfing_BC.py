#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Last edited 2021-02-19

Anisotropic polycrystal multiphase field: simulation of a single crystal bar in tension,
adapted from phase_field_composites by Jérémy Bleyer,
https://zenodo.org/record/1188970

This file is part of Gradam based on FEniCS project 
(https://fenicsproject.org/)

Gradam (c) by Jean-Michel Scherer, 
Ecole des Ponts ParisTech, 
Laboratoire Navier (ENPC,IFSTTAR,CNRS UMR 8205) & 
Ecole Polytechnique, 
Laboratoire de Mécanique des Solides, Institut Polytechnique

Gradam is licensed under a
Creative Commons Attribution-ShareAlike 4.0 International License.

You should have received a copy of the license along with this
work. If not, see <http://creativecommons.org/licenses/by-sa/4.0/>.
"""
_author__ = "Jean-Michel Scherer"
__license__ = "CC BY-SA 4.0"
__email__ = "jean-michel.scherer@enpc.fr"

import numpy as np
from dolfin import *
from mshr import *
from src import *

set_log_level(40)    # remove unnecessary console outputs

# Create mesh
mesh_folder = "meshes/"

dim=2
refinement_level = 0 #1
hole_spacing = 0
aspect_ratio = .1
L, W, R, c, e, w = 8., 1., np.sqrt(0.1/np.pi), 0.1, 0.05, 0.005
N = 1400
if(0):
  domain = Rectangle(Point(0, 0), Point(L, W)) \
  - Ellipse(Point(0., W/2.), c, 0.001, 100) 
  mesh = generate_mesh(domain, N)
  # mesh refinement
  for r in range(refinement_level):
      class CentralPart(SubDomain):
          def inside(self, x, on_boundary):
              return (0.275*W) <= x[1] <= (0.725*W)
      to_refine = MeshFunction("bool", mesh, 2)
      CentralPart().mark(to_refine, True)
      mesh = refine(mesh, to_refine)
      # plt.figure()
      # plot(mesh)
      # plt.show()

  xdmf_file = XDMFFile(MPI.comm_world, mesh_folder+"bi_crystal_notched_bar.xdmf")
  xdmf_file.write(mesh)
  #mesh_file = File(mesh_folder+"bi_crystal_notched_bar.xml")
  #mesh_file << mesh

# Load mesh
mesh_path = mesh_folder+"bi_crystal_notched_bar"
#mesh = Mesh(mesh_path+".xml")
xdmf_file = XDMFFile(MPI.comm_world, mesh_path+".xdmf")
mesh = Mesh()
xdmf_file.read(mesh)
print("number cells =", mesh.num_cells())
print("max cell size =", mesh.hmax())

# Define boundaries and boundary integration measure
facets = MeshFunction("size_t", mesh, 1)

delta = 0.
# create domains
class Grain1(SubDomain):
    def inside(self, x, on_boundary):
        return ((x[0]>e-.02) and ((x[0]+delta*(x[1]-W/2))<(((L)/4.))) and (x[1]>(e-0.02)) and (x[1]<(W-e+0.02)))
class Grain2(SubDomain):
    def inside(self, x, on_boundary):
        return (((x[0]+delta*(x[1]-W/2))>((L/4.)-0.02) and (x[0]<(L)) and (x[1]>e-0.02) and (x[1]<(W-e)+0.02)))
class Box(SubDomain):
    def inside(self, x, on_boundary):
        return ((x[0]<e) or  (x[0]>(L-e)) or  (x[1]<e) or  (x[1]>(W-e)))
mf     = MeshFunction("size_t", mesh, 2)
dsJ    = Measure("ds", subdomain_data=facets)
Grain1().mark(mf, 1)
Grain2().mark(mf, 2)
Box().mark(mf,3)

eps = 1.e-3
# create boundaries where to impose boundary conditions
class X0(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 0., eps)
class X1(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], L, eps)
class Y0(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], 0., eps)
class Y1(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], W, eps)
class Boundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary
class crack(SubDomain):
    def inside(self, x, on_boundary):
        return True if ((x[0]<(c+0.03)) & (x[0]>(c-0.02)) & (x[1]<((W/2.)+2.*w)) & (x[1]>((W/2)-2.*w))) and on_boundary else False # & on_boundary) else False
class Notch(SubDomain):
    def inside(self, x, on_boundary):
        return True if ((x[0]<(c-0.002)) & (x[1]<((W/2.)+2.*w)) & (x[1]>((W/2)-2.*w))) else False
Boundary().mark(facets,1)
crack().mark(facets,9)
Notch().mark(facets,10)

mfJ = MeshFunction("size_t", mesh, 1)
dsJ = Measure("ds", subdomain_data=mfJ)
dSJ = Measure("dS", subdomain_data=mfJ)
X0().mark(mfJ,10)
Y0().mark(mfJ,20)
X1().mark(mfJ,30)
Y1().mark(mfJ,40)

# save contours
file = File(mesh_path+".pvd")
file << mfJ
file = File(mesh_path+".pvd")
file << facets

# define material parameters
damage_tensor = True
q_, p_ = 1., 1.
q, p = Constant(q_), Constant(p_)
gamma_, r_ = 4., 0.0
gamma, r = Constant(gamma_), Constant(r_)
zener = 1.
E_, nu_ = 200., 0.3
G_ = zener*E_/(2.*(1.+nu_))
E  = [E_]*3
nu = [nu_]*3
G  = [G_]*3

# damage model parameters
damage_dim = 2
l0_= 2.e-2
l0 = [[l0_]*damage_dim]*2 + [[l0_]*damage_dim]
Gc_= 1.
Gc = [[Gc_]*damage_dim]*2 + [[Gc_]*damage_dim]
dub = [[.99]*damage_dim]*2 + [[0.]*damage_dim]

# make the fracture toughness anisotropy tensor B(3x3) in the crystal frame
I_ = np.eye(3)
I = as_tensor(I_)
if (damage_dim==1):
    B_crystal = I
else:
    B_crystal = []
    D_crystal = [] 
    alp       = 0.0 # set a value > 0 to use the AFE model
    alpha     = Constant(alp)
    M_        = [[1., 0., 0.], [0.,1.,0]]
    M         = [as_vector([1., 0., 0.]), as_vector([0.,1.,0])]
    P         = [ [[0., 1., 0.],[0., 0., 1.]], [[0., 0., 1.],[1., 0., 0.]] ]
    if dim == 3:
        M_.append([0., 0., 1.])
        M.append(as_vector([0., 0., 1.]))
        P.append([[1., 0., 0.],[0., 1., 0.]])
    for n in range(damage_dim):
        B_crystal.append(as_tensor(I) + alpha*(as_tensor(I) - outer(M[n],M[n])))
        D_crystal.append(0.5*( np.einsum('ij,kl->ikjl',I_,I_) + np.einsum('ij,kl->iljk',I_,I_) ) -\
                               np.einsum('i,j,k,l->ijkl',M_[n],M_[n],M_[n],M_[n]) -\
                               np.einsum('i,j,k,l->ijkl',M_[n],P[n][0],M_[n],P[n][0]) -\
                               np.einsum('i,j,k,l->ijkl',M_[n],P[n][1],M_[n],P[n][1]) -\
                               np.einsum('i,j,k,l->ijkl',P[n][0],M_[n],P[n][0],M_[n]) -\
                               np.einsum('i,j,k,l->ijkl',P[n][1],M_[n],P[n][1],M_[n]) )

# initialize material model class
material_parameters = {"E":E, "nu":nu, "G":G, "Gc":Gc, "l0": l0,\
                       "B_crystal": B_crystal, "D_crystal": D_crystal, "dub": dub}
mat = EXD(dim,damage_dim,material_parameters,mesh,mf,mesh_path,damage_model="AT1",\
          damage_tensor=[damage_tensor,q,p,'Lorentz',gamma,r])
suffix = mat.__class__.__name__
mat.anisotropic_degradation = True # set this to False and alp>0 to use the AFE model
#mat.tension_compression_asymmetry = False #True # not implemented yet

# initialize problem class
save_folder = "SurfingBCNotchedBicrystal/" # The path must exist
problem = FractureProblem(mesh,facets,mat,save_folder,load=Constant((0.,)*dim),\
                          Jcontours=[dsJ(10),dsJ(20),dsJ(30),dsJ(40)])
problem.dtime = 1.e-5
problem.min_dtime = 1.e-5
problem.max_dtime = 5.e-3
problem.final_time = 10.
problem.use_hybrid_solver = True
problem.incr_save = 40

# setup surfing boundary conditions
class surfingU(UserExpression):
    def __init__(self, t, Xc, ampl, mu, kappa,**kwargs):
        super().__init__(**kwargs) # This part is new!
        self.t = t
        self.Xc = Xc
        self.ampl = ampl
        self.mu = mu
        self.kappa = kappa
    def eval(self, value,x):
        r = sqrt( (x[0] - self.Xc[0] - self.t)**2 + (x[1] - self.Xc[1])**2 )
        theta = np.arctan2( x[1]-self.Xc[1], x[0]-self.Xc[0]-self.t )
        value[0] = self.ampl * np.sqrt(r / np.pi * .5) * np.cos(theta * .5) * (self.kappa - np.cos(theta))
        value[1] = self.ampl * np.sqrt(r / np.pi * .5) * np.sin(theta * .5) * (self.kappa - np.cos(theta))
    def value_shape(self):
        return (2,)
  

mu = E_ / (1. + nu_) * .5
planestress = 0
if planestress == 1:
    kappa = (3.0-nu_)/(1.0+nu_) 
    Ep = E_
else:
    kappa = 3.0-4.0*nu_
    Ep = E_ / (1.-nu_**2)
    
ampl = sqrt(Ep*Gc_)*(1.+nu_)/E_
Xc = [c, W/2] #position of the crack
t0 = 0.
    
problem.Uimp = surfingU(t0, Xc, ampl, mu, kappa)
problem.bcs  = [DirichletBC(problem.Vu, problem.Uimp, facets, 1)]
problem.bc_d = [DirichletBC(problem.Vd.sub(1), Constant(1.), facets, 9)]

# compute resultant force
problem.ds = Measure('ds')(subdomain_data=facets)
problem.resultant = problem.mat.sigma(problem.u,problem.d,problem.P1pos,problem.P2pos,problem.P3pos)[1,1]*problem.ds(5)

# Increase staggered solver accuracy
problem.staggered_solver["tol"]=1e-6

# solve problem
if (not os.path.isfile(save_folder+"output.xdmf")):
    print("START")
    problem.solve()
else:
    print("Please remove data from %s to launch a new simulation" % save_folder)
