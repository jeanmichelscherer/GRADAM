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
refinement_level = 3
hole_spacing = 0
aspect_ratio = 1.
L, W, R = 1., 1., np.sqrt(0.1/np.pi)
N = 500
if(0):
  domain = Rectangle(Point(0, 0), Point(L, W)) \
  - Ellipse(Point(L/2., W/2.), R/aspect_ratio, R, 100)
  mesh = generate_mesh(domain, N)
  # mesh refinement
  for r in range(refinement_level):
      class CentralPart(SubDomain):
          def inside(self, x, on_boundary):
              return (W/2.-.5*R) <= x[1] <= (W/2+.5*R)
      to_refine = MeshFunction("bool", mesh, 2)
      CentralPart().mark(to_refine, True)
      mesh = refine(mesh, to_refine)
      # plt.figure()
      # plot(mesh)
      # plt.show()

  xdmf_file = XDMFFile(MPI.comm_world, mesh_folder+"single_crystal_perforated_plate.xdmf")
  xdmf_file.write(mesh)
  #mesh_file = File(mesh_folder+"single_crystal_perforated_plate.xml")
  #mesh_file << mesh

# Load mesh
mesh_path = mesh_folder+"single_crystal_perforated_plate"
#mesh = Mesh(mesh_path+".xml")
xdmf_file = XDMFFile(MPI.comm_world, mesh_path+".xdmf")
mesh = Mesh()
xdmf_file.read(mesh)
print("number cells =", mesh.num_cells())
print("max cell size =", mesh.hmax())

# Define boundaries and boundary integration measure
facets = MeshFunction("size_t", mesh, 1)
facets.set_all(0)
class Left(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 0)
class Right(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], L)
class Bottom(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], 0)
class Top(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], W)
class Crack(SubDomain):
    def inside(self, x, on_boundary):
        return  True if (near(x[0],L/2,1e-3) and near(x[1],W/2,1e-3))  else False
Left().mark(facets, 1)
Right().mark(facets, 2)
Bottom().mark(facets, 3)
Top().mark(facets, 4)
Crack().mark(facets, 5)

# make MeshFunction for grain(s) subdomain(s)
one_func = MeshFunction("size_t",mesh,2)
for cell in cells(mesh):
    one_func[cell] = 1
mf = one_func
dx = Measure("dx", subdomain_data=mf)

# define material parameters
damage_tensor = True
q_, p_ = 1., 1.
q, p = Constant(q_), Constant(p_)
gamma_, r_ = 0., 0.0
gamma, r = Constant(gamma_), Constant(r_)
zener = 1.
E_, nu_ = 200., 0.3
G_ = zener*E_/(2.*(1.+nu_))
E  = [E_]*1
nu = [nu_]*1
G  = [G_]*1

# damage model parameters
damage_dim = 2
l0_= 1.e-2
l0 = [[l0_]*damage_dim]*1
Gc_= 1.
Gc = [[Gc_]*damage_dim]*1
dub = [[1.]*damage_dim]*1

# make the fracture toughness anisotropy tensor B(3x3) in the crystal frame
I_ = np.eye(3)
I = as_tensor(I_)
if (damage_dim==1):
    B_crystal = I
else:
    B_crystal = []
    D_crystal = [] 
    alp       = 100.0 # set a value > 0 to use the AFE model
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
mat.anisotropic_degradation = True # set this to True and alp=0 to use the ASD model
#mat.tension_compression_asymmetry = False #True # not implemented yet

# initialize problem class
save_folder = "TensionPerforated/" # The path must exist
problem = FractureProblem(mesh,facets,mat,save_folder,load=Constant((0.,)*dim))
problem.max_dtime = 1.e-3
problem.final_time = 1.
problem.use_hybrid_solver = True
problem.incr_save = 40

# setup boundary conditions
problem.Uimp = [Expression("t", t=0, degree=0)]
problem.bcs  = [DirichletBC(problem.Vu.sub(1), Constant(0.), facets, 3),
                DirichletBC(problem.Vu.sub(1), problem.Uimp[0], facets, 4)]
#problem.bc_d = [DirichletBC(problem.Vd, Constant((.1,)*damage_dim), facets, 5)]

# compute resultant force
problem.ds = Measure('ds')(subdomain_data=facets)
problem.resultant = problem.mat.sigma(problem.u,problem.d,problem.P1pos,problem.P2pos,problem.P3pos)[1,1]*problem.ds(4)

# Increase staggered solver accuracy
problem.staggered_solver["tol"]=1e-6

# solve problem
if (not os.path.isfile(save_folder+"output.xdmf")):
    print("START")
    problem.solve()
else:
    print("Please remove data from %s to launch a new simulation" % save_folder)
