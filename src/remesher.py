#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Last edited 2021-03-27

Anisotropic polycrystal multiphase field: writing .sol files for MMG remeshing,
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

from numpy import linalg as LA
import os
import scipy as sp
import scipy.ndimage

#mesh_file = 'mesh_100_crack'
#sol_file  = open(mesh_file+'.sol','w')
#node_number = 5585



def write_uniform_sol(mesh_file,sol_file,number_of_nodes_index,uniform_metric):
    sol_file  = open(mesh_file+'_remeshed_0.sol','w')
    f = open(mesh_file+'.msh', "r")
    lines = f.readlines()
    i = 0
    number_of_nodes = 0
    while (number_of_nodes==0):
        if lines[i].startswith("$Nodes"):
            number_of_nodes = int(lines[i+1].split()[number_of_nodes_index])
        i+=1
    f.close()
    
    sol_file.write(\
    'MeshVersionFormatted 2\n\
    \n\
    Dimension 3\n\
    \n\
    SolAtVertices\n\
    %s\n\
    1 1\n\
    \n'
    %number_of_nodes)
    
    for i in range(number_of_nodes):
        sol_file.write('%s\n' % uniform_metric)
    
    sol_file.write('\nEND')
    
    sol_file.close()


def write_dvar_sol(damage_dim,
                   mesh_file,
                   sol_file,
                   number_of_nodes_index,
                   metric_field,
                   sol_min,
                   sol_max,
                   remeshing_index):
    # find number of $Nodes in the mesh
    f = open(mesh_file+'_remeshed_%s.msh' % (remeshing_index-1), "r")
    lines = f.readlines()
    i = 0
    number_of_nodes = 0
    while (number_of_nodes==0):
        if lines[i].startswith("$Nodes"):
            number_of_nodes = int(lines[i+1].split()[number_of_nodes_index])
        i+=1
    f.close()

    #prev_sol_file = open(sol_file+'_remeshed_%s.o.sol' % (remeshing_index-1),'r')
    #prev_sols = prev_sol_file.readlines()

    sol_file  = open(sol_file+'_remeshed_%s.sol' % (remeshing_index-1),'w')
    # print("nb =" , number_of_nodes, "remeshing_index = ", remeshing_index)
    sol_file.write('MeshVersionFormatted 2\n\nDimension 3\n\nSolAtVertices\n%s\n1 1\n\n'%number_of_nodes)
    
    #metric_field = metric_field.reshape((number_of_nodes,damage_dim))
    #metric_field = LA.norm(metric_field,axis=1)
    
    # # Apply gaussian filter (not working because of nodes ordering)
    # #sigma_y = 3.0
    # #sigma_x = 3.0
    # sigma = 3.#[sigma_y, sigma_x]
    # #print(metric_field)
    # metric_field = sp.ndimage.filters.gaussian_filter(metric_field, sigma, mode='constant')
    # #print(metric_field)

    for i in range(number_of_nodes):
        #sol = float(prev_sols[i+9].split()[0])
        #sol_file.write('%s\n' % (sol*max(0.01,(1.-1000.*metric_field[i]))))
        #new_sol = min( max( sol*(1.1 - 200.*metric_field[i]) , sol_min) , sol_max )
        #new_sol = min( max( sol_max*(1. - 200.*metric_field[i]) , sol_min) , sol_max )
        #new_sol = min( max( sol_max*(1. - 10.*metric_field[i]) , sol_min) , sol_max )
        #new_sol = min( max( sol_max*(1. - 2.*metric_field[i]) , sol_min) , sol_max )
        new_sol = min( max( sol_max*(1. - 3.*metric_field[i]) , sol_min) , sol_max )
        sol_file.write( '%s\n' % new_sol )
    
    sol_file.write('\nEND')
    
    sol_file.close()
    
def remesh(dim,mesh_path,geo_tmpl,mesh_file,nbgrains,remeshing_index):
    oldmesh = mesh_file+"_remeshed_%s" % (remeshing_index-1)
    newmesh = mesh_file+"_remeshed_%s" % (remeshing_index) #+1)
    print("\nCalling MMG to perform remeshing \n")
    os.system("mmg%sd_O3 -3dMedit %s %s" % (dim,dim,mesh_path+oldmesh+'.mesh') )
    
    f = open(mesh_path+geo_tmpl+'.geo.tmpl','r')
    lines = f.readlines()
    geo = mesh_path+geo_tmpl+'.geo'
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
    os.system("gmsh %s" % geo)
    
    
