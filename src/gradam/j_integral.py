#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 12:30:35 2021

@author: scherer
"""

import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as axes3d
from matplotlib import cm
import sys
import os
import gradam
#from .j_contours_paraview import *
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

def trapz(a,b,c):
    product_y = [b_values*c_values for b_values,c_values in zip(b,c)]
    list = []
    for i in range(len(a)-1):
        list.append((abs(a[i+1]-a[i]))*((product_y[i+1]+product_y[i])/2))
    return (sum(list))

class JIntegral:
    def __init__(self,path=None, input_file=None, x_bot=None, x_top=None, y_bot=None, y_top=None, resolution=None, incr_save=1):
        self.path = path
        self.input_file = input_file
        self.x_bot = x_bot
        self.x_top = x_top
        self.y_bot = y_bot
        self.y_top = y_top
        self.resolution = resolution
        self.incr_save = incr_save

    def compute_J_integral(self, contour, step, time):
        #run_paraview_PP_j_contours(step, self.path, self.input_file, self.x_bot, self.x_top, self.y_bot, self.y_top, self.resolution )
        gradam_location = gradam.__file__[:-11]
        cmd = "pvbatch %sj_contours_paraview.py %s %s %s %s %s %s %s %s %s" %\
                    (gradam_location, step, self.path, self.input_file, contour, self.x_bot, self.x_top, self.y_bot, self.y_top, self.resolution)
        #print("Running: ", cmd)
        os.system(cmd)
        
        steps = 1
        data   = []
        e11    = []
        e22    = []
        e33    = []
        e12    = []
        e23    = []
        e31    = []
        e21    = []
        e32    = []
        e13    = []
        s11    = []
        s22    = []
        s33    = []
        s12    = []
        s23    = []
        s31    = []
        s21    = []
        s32    = []
        s13    = []
        
        J_file = open(self.path+'J_integral_%s.txt' % contour,'a')
        #J_file.write("#incr time J_left J_bot J_right J_top J_tot\n")
        #for i in range(steps):
        J = []
        sides = ['left','bot','right','top']
        for (s,side) in enumerate(sides):
            data = np.loadtxt(self.path+'%s_%s_stress_strain_coord_%s.csv' % (contour,side,step), delimiter=',',skiprows=1)
            #data.append( np.loadtxt('%s_stress_strain_coord_%s.csv' % (side,i), delimiter=',',skiprows=1) )
           
            #time = data[0,0]
            e11  = data[:,1]
            e12  = data[:,2]
            e13  = data[:,3]
            e21  = data[:,4]
            e22  = data[:,5]
            e23  = data[:,6]
            e31  = data[:,7]
            e32  = data[:,8]
            e33  = data[:,9]
            s11  = data[:,10]
            s12  = data[:,11]
            s13  = data[:,12]
            s21  = data[:,13]
            s22  = data[:,14]
            s23  = data[:,15]
            s31  = data[:,16]
            s32  = data[:,17]
            s33  = data[:,18]
            e11[np.isnan(e11)] = 0
            e12[np.isnan(e12)] = 0
            e13[np.isnan(e13)] = 0
            e21[np.isnan(e21)] = 0
            e22[np.isnan(e22)] = 0
            e23[np.isnan(e23)] = 0
            e31[np.isnan(e31)] = 0
            e32[np.isnan(e32)] = 0
            e33[np.isnan(e33)] = 0
            s11[np.isnan(s11)] = 0
            s12[np.isnan(s12)] = 0
            s13[np.isnan(s13)] = 0
            s21[np.isnan(s21)] = 0
            s22[np.isnan(s22)] = 0
            s23[np.isnan(s23)] = 0
            s31[np.isnan(s31)] = 0
            s32[np.isnan(s32)] = 0
            s33[np.isnan(s33)] = 0
            x    = data[:,19]
            y    = data[:,20]
            z    = data[:,21]
            
            if side == 'bot' or side == 'top':
                h = x
                dx2 = 0.
                if side == 'bot':
                    dx1 = 1.
                elif side == 'top':
                    dx1 = -1.
                J_Tgradu = (trapz(h,s12,e11) + trapz(h,s22,e21)) * dx1
            elif side == 'left' or side == 'right':
                h = y
                if side == 'left':
                    dx1 = -1.
                    dx2 = -1.
                elif side == 'right':
                    dx1 = 1.
                    dx2 = 1.
                J_Tgradu = -(trapz(h,s11,e11) + trapz(h,s12,e21)) * dx1
            
            J_W      = 0.5*( trapz(h,s11,e11) + trapz(h,s22,e22) + 2.*trapz(h,s12,e12) ) * dx2
            
            J.append( J_W + J_Tgradu ) 
            
            #print(step, ": ",  time, ": ", side, ": ", J_W, ", ", J_Tgradu)
    
        J_file.write("%s %s " % (step,time) + " ".join(map(str,J)) + " %s" % sum(J) +"\n")

        
        print("\t\t\tParaview computed J-integral on contour %s: %s " % (contour,sum(J)))
        
        J_file.close()
