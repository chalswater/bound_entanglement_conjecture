#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 12:55:42 2024

@author: carles
"""

import numpy as np
import mosek
import cvxpy as cp
import matplotlib.pyplot as plt
#from toqito.random import random_state_vector

from qutip import *

from cvxpy import *

X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0], [0, -1]])
id_2 = np.eye(2)

#Bell states
phi_p_vec = np.array([[1.0,0.0,0.0,1.0]])/np.sqrt(2.0)
phi_m_vec = np.array([[1.0,0.0,0.0,-1.0]])/np.sqrt(2.0)
psi_p_vec = np.array([[0.0,1.0,1.0,0.0]])/np.sqrt(2.0)
psi_m_vec = np.array([[0.0,1.0,-1.0,0.0]])/np.sqrt(2.0)

phi_p = np.kron(phi_p_vec,np.transpose(phi_p_vec))
phi_m = np.kron(phi_m_vec,np.transpose(phi_m_vec))
psi_p = np.kron(psi_p_vec,np.transpose(psi_p_vec))
psi_m = np.kron(psi_m_vec,np.transpose(psi_m_vec))

d = 5
level = 1
[nX0,nX1,nY0,nY1] = [d,d,d,d]
[nZ,nB] = [d+1,d]

#Locate the variables from the matrix of correlations
w_id = np.zeros(1,dtype=int)

cc = 0
w_id[0] = cc

i_rho_sigma = False
i_rho_M = True
i_sigma_M = False

#First level monomials
w_rho = np.zeros((nX0,nX1),dtype=int)
w_sigma = np.zeros((nY0,nY1),dtype=int)
w_M = np.zeros((nZ,nB),dtype=int)

#Second level monomials
w_rho_sigma = np.zeros((nX0,nX1,nY0,nY1),dtype=int)
w_rho_M = np.zeros((nX0,nX1,nZ,nB),dtype=int)
w_sigma_M = np.zeros((nY0,nY1,nZ,nB),dtype=int)

for x0 in range(nX0):
    for x1 in range(nX1):
        cc += 1
        w_rho[x0][x1] = cc

for y0 in range(nY0):
    for y1 in range(nY1):
        cc += 1
        w_sigma[y0][y1] = cc
        
for z in range(nZ):
    for b in range(nB):
        cc += 1
        w_M[z][b] = cc
        
print('First level',cc)

if i_rho_sigma == True:

    for x0 in range(nX0):
        for x1 in range(nX1):
            for y0 in range(nY0):
                for y1 in range(nY1):
                    cc += 1
                    w_rho_sigma[x0][x1][y0][y1] = cc
                    
if i_rho_M == True:
    for x0 in range(nX0):
        for x1 in range(nX1):
            for z in range(nZ):
                for b in range(nB):
                    cc += 1
                    w_rho_M[x0][x1][z][b] = cc
                    
if i_sigma_M == True:
                
    for y0 in range(nY0):
        for y1 in range(nY1):
            for z in range(nZ):
                for b in range(nB):
                    cc += 1
                    w_sigma_M[y0][y1][z][b] = cc
            
print('Second level',cc)

#dim= id + rho x id + id x sigma +  M_{z|b} 
dimG = 1 +  nX0*nX1 +  nY0*nY1   + nZ*nB  

if i_rho_sigma == True:
    dimG += nX0*nX1*nY0*nY1   #rho x sigma 
if i_rho_M == True:
    dimG += nX0*nX1*nZ*nB     #rho * M_{z|b}
if i_sigma_M == True:
    dimG += nY0*nY1*nZ*nB    #sigma * M_{z|b}
    
print('DimG:',dimG)

Gamma = cp.Variable((dimG,dimG),PSD=True)

#Store correlations in this variable
pbxyz = {}
for b in range(nB):
    pbxyz[b] = {}
    for x0 in range(nX0):
        pbxyz[b][x0] = {}
        for x1 in range(nX1):
            pbxyz[b][x0][x1] = {}
            for y0 in range(nY0):
                pbxyz[b][x0][x1][y0] = {}
                for y1 in range(nY1):
                    pbxyz[b][x0][x1][y0][y1] = {}
                    for z in range(nZ):
                        pbxyz[b][x0][x1][y0][y1][z] = cp.Variable(nonneg=True)
                        
                        
#List of rules (matrix)

# tr(I * I) = d**2
# tr(I * (R x I)) = d          tr((R x I) * (R x I)) = d
# tr(I * (I x S)) = d          tr((R x I) * (I x S)) = 1          tr((I x S) * (I x S)) = d
# sum_b tr(I * Mb) = d**2      sum_b tr((R x I) * Mb) = d         sum_b tr((I x S) * Mb) = d         tr(Mb * Mb') = ??
# tr(I * (R x S)) = 1          tr((R x I) * (R x S)) = 1          tr((I x S) * (R x S)) = 1          sum_b tr(Mb * (R x S)) = 1  tr((R x S) * (R x S)) = 1
# sum_b tr(I * Mb(R x I)) = d  sum_b tr((R x I) * Mb(R x I)) = d  sum_b tr((I x S) * Mb(R x I)) = 1  tr(Mb * Mb'(R x I)) = ??    sum_b tr((R x S) * Mb(R x I)) = 1  tr(Mb(R x I) * Mb'(R x I)) = ??
# sum_b tr(I * Mb(I x S)) = d  sum_b tr((R x I) * Mb(I x S)) = 1  sum_b tr((I x S) * Mb(I x S)) = d  tr(Mb * Mb'(I x S)) = ??    sum_b tr((R x S) * Mb(I x S)) = 1  tr(Mb(R x I) * Mb'(I x S)) = ??  tr(Mb(I x S) * Mb'(I x S)) = ??

#lvl 0 number of operators: 1 (identity)
#lvl 1 number of operators: 3 (rho, sigma, M)
#lvl 2 number of operators: 3 (rho x sigma, M * (rho x I), M * (I x sigma))


ct = []

#***************************
#lvl 0 column
#***************************

ct += [Gamma[w_id[0]][w_id[0]] == d**2]

ct += [Gamma[w_id[0]][w_rho[x0][x1]] == d for x0 in range(nX0) for x1 in range(nX1)]

ct += [Gamma[w_id[0]][w_sigma[y0][y1]] == d for y0 in range(nY0) for y1 in range(nY1)]

ct += [sum([Gamma[w_id[0]][w_M[z][b]] for b in range(nB)]) == d**2 for z in range(nZ)]

if i_rho_sigma == True:
    ct += [Gamma[w_id[0]][w_rho_sigma[x0][x1][y0][y1]] == 1.0 for x0 in range(nX0) for x1 in range(nX1) for y0 in range(nY0) for y1 in range(nY1)]

if i_rho_M == True:
    ct += [sum([Gamma[w_id[0]] [w_rho_M[x0][x1][z][b]] for b in range(nB)]) == Gamma[w_rho[x0][x1]][w_id[0]]  for z in range(nZ) for x0 in range(nX0) for x1 in range(nX1)]

if i_sigma_M == True:    
    ct += [sum([Gamma[w_id[0]][w_sigma_M[y0][y1][z][b]] for b in range(nB)]) == Gamma[w_sigma[y0][y1]][w_id[0]] for z in range(nZ) for y0 in range(nY0) for y1 in range(nY1)]

#***************************
#lvl 1  first column
#***************************

ct += [Gamma[w_rho[x0][x1]][w_rho[x0][x1]] == Gamma[w_id[0]][w_rho[x0][x1]] for x0 in range(nX0) for x1 in range(nX1)]

ct += [Gamma[w_rho[x0][x1]][w_sigma[y0][y1]] == 1.0 for x0 in range(nX0) for x1 in range(nX1) for y0 in range(nY0) for y1 in range(nY1)]

ct += [sum([Gamma[w_rho[x0][x1]][w_M[z][b]] for b in range(nB)]) == Gamma[w_rho[x0][x1]][w_id[0]] for z in range(nZ) for x0 in range(nX0) for x1 in range(nX1)]   

if i_rho_sigma == True:
    ct += [Gamma[w_rho[x0][x1]][w_rho_sigma[x0][x1][y0][y1]] == 1.0 for x0 in range(nX0) for x1 in range(nX1) for y0 in range(nY0) for y1 in range(nY1)]

if i_rho_M == True:
    ct += [sum([Gamma[w_rho[x0][x1]][w_rho_M[xx0][xx1][z][b]] for b in range(nB)]) == Gamma[w_rho[x0][x1]][w_rho[xx0][xx1]] for z in range(nZ) for x0 in range(nX0) for x1 in range(nX1) for xx0 in range(nX0) for xx1 in range(nX1)]

if i_sigma_M == True:
    ct += [sum([Gamma[w_rho[x0][x1]][w_sigma_M[y0][y1][z][b]] for b in range(nB)]) == Gamma[w_rho[x0][x1]][w_sigma[y0][y1]] for z in range(nZ) for x0 in range(nX0) for x1 in range(nX1) for y0 in range(nY0) for y1 in range(nY1)]
    ct += [Gamma[w_rho[x0][x1]][w_sigma_M[y0][y1][z][b]] == pbxyz[b][x0][x1][y0][y1][z] for b in range(nB) for z in range(nZ) for x0 in range(nX0) for x1 in range(nX1) for y0 in range(nY0) for y1 in range(nY1)]

#***************************
#lvl 1  second column
#***************************

ct += [Gamma[w_sigma[y0][y1]][w_sigma[y0][y1]] == Gamma[w_id[0]][w_sigma[y0][y1]] for y0 in range(nY0) for y1 in range(nY1)]

ct += [sum([Gamma[w_sigma[y0][y1]][w_M[z][b]] for b in range(nB)]) == Gamma[w_sigma[y0][y1]][w_id[0]] for z in range(nZ) for y0 in range(nY0) for y1 in range(nY1)]

if i_rho_sigma == True:
    ct += [Gamma[w_sigma[y0][y1]][w_rho_sigma[x0][x1][y0][y1]] == 1.0 for x0 in range(nX0) for x1 in range(nX1) for y0 in range(nY0) for y1 in range(nY1)]

if i_rho_M == True:
    ct += [sum([Gamma[w_sigma[y0][y1]][w_rho_M[x0][x1][z][b]] for b in range(nB)]) == Gamma[w_sigma[y0][y1]][w_rho[x0][x1]] for z in range(nZ) for x0 in range(nX0) for x1 in range(nX1) for y0 in range(nY0) for y1 in range(nY1)]
    ct += [Gamma[w_sigma[y0][y1]][w_rho_M[x0][x1][z][b]] == pbxyz[b][x0][x1][y0][y1][z] for b in range(nB) for z in range(nZ) for x0 in range(nX0) for x1 in range(nX1) for y0 in range(nY0) for y1 in range(nY1)]

if i_sigma_M == True:
    ct += [sum([Gamma[w_sigma[y0][y1]][w_sigma_M[yy0][yy1][z][b]] for b in range(nB)]) == Gamma[w_sigma[y0][y1]][w_sigma[yy0][yy1]] for z in range(nZ) for y0 in range(nY0) for y1 in range(nY1) for yy0 in range(nY0) for yy1 in range(nY1)]

#***************************
#lvl 1  third column
#***************************

ct += [sum([Gamma[w_M[z][b]][w_M[zz][bb]] for b in range(nB) for bb in range(nB)]) == Gamma[w_id[0]][w_id[0]] for z in range(nZ) for zz in range(nZ)]
#ct += [sum([Gamma[w_M[z][b]][w_M[zz][bb]] for b in range(nB)]) == Gamma[w_id[0]][w_M[zz][bb]] for bb in range(nB) for z in range(nZ) for zz in range(nZ)]
#ct += [sum([Gamma[w_M[z][b]][w_M[zz][bb]] for bb in range(nB)]) == Gamma[w_id[0]][w_M[z][b]] for b in range(nB) for z in range(nZ) for zz in range(nZ)]
#Orthogonal projector contraints!! -------
ct += [Gamma[w_M[z][b]][w_M[z][bb]] == deltaF(b,bb)*Gamma[w_M[z][b]][w_id[0]] for b in range(nB) for bb in range(nB) for z in range(nZ)]
#-----------------------------------------

if i_rho_sigma == True:
    ct += [sum([Gamma[w_M[z][b]][w_rho_sigma[x0][x1][y0][y1]] for b in range(nB)]) == Gamma[w_id[0]][w_rho_sigma[x0][x1][y0][y1]] for z in range(nZ) for x0 in range(nX0) for x1 in range(nX1) for y0 in range(nY0) for y1 in range(nY1)]
    ct += [Gamma[w_M[z][b]][w_rho_sigma[x0][x1][y0][y1]] == pbxyz[b][x0][x1][y0][y1][z] for b in range(nB) for z in range(nZ) for x0 in range(nX0) for x1 in range(nX1) for y0 in range(nY0) for y1 in range(nY1)]

if i_rho_M == True:
    #ct += [sum([Gamma[w_M[z][b]][w_rho_M[x0][x1][zz][bb]] for b in range(nB) for bb in range(nB)]) == Gamma[w_id[0]][w_rho[x0][x1]] for z in range(nZ) for zz in range(nZ) for x0 in range(nX0) for x1 in range(nX1)]
    #ct += [sum([Gamma[w_M[z][b]][w_rho_M[x0][x1][zz][bb]] for b in range(nB)]) == Gamma[w_id[0]][w_rho_M[x0][x1][zz][bb]] for bb in range(nB) for z in range(nZ) for zz in range(nZ) for x0 in range(nX0) for x1 in range(nX1)]
    #ct += [sum([Gamma[w_M[z][b]][w_rho_M[x0][x1][zz][bb]] for bb in range(nB)]) == Gamma[w_M[z][b]][w_rho[x0][x1]] for b in range(nB) for z in range(nZ) for zz in range(nZ) for x0 in range(nX0) for x1 in range(nX1)]
    #Orthogonal projector contraints!! -------
    ct += [Gamma[w_M[z][b]][w_rho_M[x0][x1][z][bb]] == deltaF(b,bb)*Gamma[w_M[z][b]][w_rho[x0][x1]] for b in range(nB) for bb in range(nB) for z in range(nZ) for x0 in range(nX0) for x1 in range(nX1)]
    #-----------------------------------------

if i_sigma_M == True:
    ct += [sum([Gamma[w_M[z][b]][w_sigma_M[y0][y1][zz][bb]] for b in range(nB) for bb in range(nB)]) == Gamma[w_id[0]][w_sigma[y0][y1]] for z in range(nZ) for zz in range(nZ) for y0 in range(nY0) for y1 in range(nY1)]
    ct += [sum([Gamma[w_M[z][b]][w_sigma_M[y0][y1][zz][bb]] for b in range(nB)]) == Gamma[w_id[0]][w_sigma_M[y0][y1][zz][bb]] for bb in range(nB) for z in range(nZ) for zz in range(nZ) for y0 in range(nY0) for y1 in range(nY1)]
    ct += [sum([Gamma[w_M[z][b]][w_sigma_M[y0][y1][zz][bb]] for bb in range(nB)]) == Gamma[w_M[z][b]][w_sigma[y0][y1]] for b in range(nB) for z in range(nZ) for zz in range(nZ) for y0 in range(nY0) for y1 in range(nY1)]
    #Orthogonal projector contraints!! -------
    ct += [Gamma[w_M[z][b]][w_sigma_M[y0][y1][z][bb]] == deltaF(b,bb)*Gamma[w_M[z][b]][w_sigma[y0][y1]] for b in range(nB) for bb in range(nB) for z in range(nZ) for y0 in range(nY0) for y1 in range(nY1)]
    #-----------------------------------------

#***************************
#lvl 2 first column
#***************************

if i_rho_sigma == True:
    ct += [Gamma[w_rho_sigma[x0][x1][y0][y1]][w_rho_sigma[x0][x1][y0][y1]] == 1.0 for x0 in range(nX0) for x1 in range(nX1) for y0 in range(nY0) for y1 in range(nY1)]
    
    if i_rho_M == True:
        ct += [sum([Gamma[w_rho_sigma[x0][x1][y0][y1]][w_rho_M[xx0][xx1][z][b]] for b in range(nB)]) == Gamma[w_rho_sigma[x0][x1][y0][y1]][w_rho[xx0][xx1]] for z in range(nZ) for x0 in range(nX0) for x1 in range(nX1) for y0 in range(nY0) for y1 in range(nY1) for xx0 in range(nX0) for xx1 in range(nX1)]
    if i_sigma_M == True:
        ct += [sum([Gamma[w_rho_sigma[x0][x1][y0][y1]][w_sigma_M[yy0][yy1][z][b]] for b in range(nB)]) == Gamma[w_rho_sigma[x0][x1][y0][y1]][w_sigma[yy0][yy1]] for z in range(nZ) for x0 in range(nX0) for x1 in range(nX1) for y0 in range(nY0) for y1 in range(nY1) for yy0 in range(nY0) for yy1 in range(nY1)]

#***************************
#lvl 2 second column
#***************************

if i_rho_M == True:
    #ct += [sum([Gamma[w_rho_M[x0][x1][z][b]][w_rho_M[x0][x1][zz][bb]] for b in range(nB) for bb in range(nB)]) == Gamma[w_rho[x0][x1]][w_rho[x0][x1]] for z in range(nZ) for zz in range(nZ) for x0 in range(nX0) for x1 in range(nX1)]
    #ct += [sum([Gamma[w_rho_M[x0][x1][z][b]][w_rho_M[xx0][xx1][zz][bb]] for b in range(nB)]) == Gamma[w_rho[x0][x1]][w_rho_M[xx0][xx1][zz][bb]] for bb in range(nB) for z in range(nZ) for zz in range(nZ) for x0 in range(nX0) for x1 in range(nX1) for xx0 in range(nX0) for xx1 in range(nX1)]
    #ct += [sum([Gamma[w_rho_M[x0][x1][z][b]][w_rho_M[xx0][xx1][zz][bb]] for bb in range(nB)]) == Gamma[w_rho_M[x0][x1][z][b]][w_rho[xx0][xx1]] for b in range(nB) for z in range(nZ) for zz in range(nZ) for x0 in range(nX0) for x1 in range(nX1) for xx0 in range(nX0) for xx1 in range(nX1)]
    #Orthogonal projector contraints!! -------
    ct += [Gamma[w_rho_M[x0][x1][z][b]][w_rho_M[xx0][xx1][z][bb]] == deltaF(b,bb)*Gamma[w_rho_M[x0][x1][z][b]][w_rho[xx0][xx1]] for b in range(nB) for bb in range(nB) for z in range(nZ) for x0 in range(nX0) for x1 in range(nX1) for xx0 in range(nX0) for xx1 in range(nX1)]
    #ct += [Gamma[w_rho_M[x0][x1][z][b]][w_rho_M[xx0][xx1][z][b]] == Gamma[w_rho_M[x0][x1][z][b]][w_rho[xx0][xx1]] for b in range(nB) for z in range(nZ) for x0 in range(nX0) for x1 in range(nX1) for xx0 in range(nX0) for xx1 in range(nX1)]
    #ct += [Gamma[w_rho_M[x0][x1][z][b]][w_rho_M[xx0][xx1][z][b]] == Gamma[w_rho[x0][x1]][w_rho_M[xx0][xx1][z][b]] for b in range(nB) for z in range(nZ) for x0 in range(nX0) for x1 in range(nX1) for xx0 in range(nX0) for xx1 in range(nX1)]
    #-----------------------------------------

    if i_sigma_M == True:    
        ct += [sum([Gamma[w_rho_M[x0][x1][z][b]][w_sigma_M[y0][y1][zz][bb]] for b in range(nB) for bb in range(nB)]) == 1.0 for z in range(nZ) for zz in range(nZ) for x0 in range(nX0) for x1 in range(nX1) for y0 in range(nY0) for y1 in range(nY1)]
        ct += [sum([Gamma[w_rho_M[x0][x1][z][b]][w_sigma_M[y0][y1][zz][bb]] for b in range(nB)]) == Gamma[w_rho[x0][x1]][w_sigma_M[y0][y1][zz][bb]] for bb in range(nB) for z in range(nZ) for zz in range(nZ) for x0 in range(nX0) for x1 in range(nX1) for y0 in range(nY0) for y1 in range(nY1)]
        ct += [sum([Gamma[w_rho_M[x0][x1][z][b]][w_sigma_M[y0][y1][zz][bb]] for bb in range(nB)]) == Gamma[w_rho_M[x0][x1][z][b]][w_sigma[y0][y1]] for b in range(nB) for z in range(nZ) for zz in range(nZ) for x0 in range(nX0) for x1 in range(nX1) for y0 in range(nY0) for y1 in range(nY1)]
        #Orthogonal projector contraints!! -------
        ct += [Gamma[w_rho_M[x0][x1][z][b]][w_sigma_M[y0][y1][z][bb]] == 0.0 for b in range(nB) for bb in range(nB) if b != bb for z in range(nZ) for x0 in range(nX0) for x1 in range(nX1) for y0 in range(nY0) for y1 in range(nY1)]
        ct += [Gamma[w_rho_M[x0][x1][z][b]][w_sigma_M[y0][y1][z][b]] == Gamma[w_rho_M[x0][x1][z][b]][w_sigma[y0][y1]] for b in range(nB) for z in range(nZ) for x0 in range(nX0) for x1 in range(nX1) for y0 in range(nY0) for y1 in range(nY1)]
        ct += [Gamma[w_rho_M[x0][x1][z][b]][w_sigma_M[y0][y1][z][b]] == Gamma[w_rho[x0][x1]][w_sigma_M[y0][y1][z][b]] for b in range(nB) for z in range(nZ) for x0 in range(nX0) for x1 in range(nX1) for y0 in range(nY0) for y1 in range(nY1)]
        #-----------------------------------------

#***************************
#lvl 2 third column
#***************************

if i_sigma_M == True:
    ct += [sum([Gamma[w_sigma_M[y0][y1][z][b]][w_sigma_M[y0][y1][zz][bb]] for b in range(nB) for bb in range(nB)]) == Gamma[w_sigma[y0][y1]][w_sigma[y0][y1]] for z in range(nZ) for zz in range(nZ) for y0 in range(nY0) for y1 in range(nY1)]
    ct += [sum([Gamma[w_sigma_M[y0][y1][z][b]][w_sigma_M[yy0][yy1][zz][bb]] for b in range(nB)]) == Gamma[w_sigma[y0][y1]][w_sigma_M[yy0][yy1][zz][bb]] for bb in range(nB) for z in range(nZ) for zz in range(nZ) for y0 in range(nY0) for y1 in range(nY1) for yy0 in range(nY0) for yy1 in range(nY1)]
    ct += [sum([Gamma[w_sigma_M[y0][y1][z][b]][w_sigma_M[yy0][yy1][zz][bb]] for bb in range(nB)]) == Gamma[w_sigma_M[y0][y1][z][b]][w_sigma[yy0][yy1]] for b in range(nB) for z in range(nZ) for zz in range(nZ) for y0 in range(nY0) for y1 in range(nY1) for yy0 in range(nY0) for yy1 in range(nY1)]
    #Orthogonal projector contraints!! -------
    ct += [Gamma[w_sigma_M[y0][y1][z][b]][w_sigma_M[yy0][yy1][z][bb]] == 0.0 for b in range(nB) for bb in range(nB) if b != bb for z in range(nZ) for y0 in range(nY0) for y1 in range(nY1) for yy0 in range(nY0) for yy1 in range(nY1)]
    ct += [Gamma[w_sigma_M[y0][y1][z][b]][w_sigma_M[yy0][yy1][z][b]] == Gamma[w_sigma_M[y0][y1][z][b]][w_sigma[yy0][yy1]] for b in range(nB) for z in range(nZ) for y0 in range(nY0) for y1 in range(nY1) for yy0 in range(nY0) for yy1 in range(nY1)]
    ct += [Gamma[w_sigma_M[y0][y1][z][b]][w_sigma_M[yy0][yy1][z][b]] == Gamma[w_sigma[y0][y1]][w_sigma_M[yy0][yy1][z][b]] for b in range(nB) for z in range(nZ) for y0 in range(nY0) for y1 in range(nY1) for yy0 in range(nY0) for yy1 in range(nY1)]
    #-----------------------------------------
                
#Relations between elements in Gamma

if i_rho_sigma == True:
    ct += [Gamma[w_rho_sigma[x0][x1][y0][y1]][w_id[0]] == Gamma[w_rho[x0][x1]][w_sigma[y0][y1]] for x0 in range(nX0) for x1 in range(nX1) for y0 in range(nY0) for y1 in range(nY1)]

if i_rho_M == True:
    ct += [Gamma[w_rho_M[x0][x1][z][b]][w_id[0]] == Gamma[w_rho[x0][x1]][w_M[z][b]] for b in range(nB) for z in range(nZ) for x0 in range(nX0) for x1 in range(nX1)]

if i_sigma_M == True:
    ct += [Gamma[w_sigma_M[y0][y1][z][b]][w_id[0]] == Gamma[w_sigma[y0][y1]][w_M[z][b]] for b in range(nB) for z in range(nZ) for y0 in range(nY0) for y1 in range(nY1)]

if i_rho_sigma == True and i_rho_M == True:
    ct += [Gamma[w_rho_sigma[x0][x1][y0][y1]][w_M[z][b]] == Gamma[w_rho_M[x0][x1][z][b]][w_sigma[y0][y1]] for x0 in range(nX0) for x1 in range(nX1) for y0 in range(nY0) for y1 in range(nY1) for b in range(nB) for z in range(nZ)]

if i_rho_sigma == True and i_sigma_M == True:
    ct += [Gamma[w_rho_sigma[x0][x1][y0][y1]][w_M[z][b]] == Gamma[w_sigma_M[y0][y1][z][b]][w_rho[x0][x1]] for x0 in range(nX0) for x1 in range(nX1) for y0 in range(nY0) for y1 in range(nY1) for b in range(nB) for z in range(nZ)]

Sd = 0.0
for x0 in range(nX0):
    for x1 in range(nX1):
        for y0 in range(nY0):
            for y1 in range(nY1):
                for z in range(nZ):
                            
                    #Define the game win condition -------------------
                    if d > 2:
                        if z < d:
                            w = int(np.mod(x1 + y1 - 2.0*z*(x0-y0),d)) 
                        else:
                            w = int(np.mod(x0-y0,d))
                    else:
                        if z == 0:
                            w = int(np.mod(x0+y0,d))
                        elif z == 1:
                            w = int(np.mod(x1+y1,d))
                        elif z == 2:
                            w = int(np.mod(x0+y0+x1+y1,d))
                    #-------------------------------------------------
                    
                    Sd = Sd + pbxyz[w][x0][x1][y0][y1][z]/(d**4*(1.0+d))
                    
obj = cp.Maximize(Sd)
prob = cp.Problem(obj,ct)

try:
    mosek_params = {
            "MSK_DPAR_INTPNT_CO_TOL_REL_GAP": 1e-1
        }
    prob.solve(solver='MOSEK',verbose=True, mosek_params=mosek_params)

except SolverError:
    something = 10
    
print(Sd.value,2.0/(1.0+d))
