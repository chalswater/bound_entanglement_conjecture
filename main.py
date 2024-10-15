#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 12:55:10 2024

@author: carles
"""

import numpy as np
import mosek
import cvxpy as cp
import matplotlib.pyplot as plt
#from toqito.random import random_state_vector

from qutip import *

from cvxpy import *

import time

# Moment matrix generators
from MoMPy.MoM import *


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


#------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------#
#                                        Functions                                         #
#------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------#

def Game(nX0,nX1,nY0,nY1,nZ,nB,monomials,gamma_matrix_els,d):
    
    """
    Compute the maximum success probability of a quantum communication game
    
    Inputs:
        nX0,nX1:
            Number of state preparations in Alice side
        nY0,nY1:
            Number of state preparations in Bob's side
        nB:
            Number of measurement outcomes in Charlies side
        nZ:
            Number of measurement settings in Charlie's side
        monomials:
            List of monomials used to build the SDP relaxation
        gamma_matrix_els:
            Set of identities and full matrix from the SDP relaxation after applying the relaxation rules
    """    
    [w_R,w_S,w_M] = monomials
    [G_new,map_table,S,list_of_eq_indices,Mexp] = gamma_matrix_els

    #----------------------------------------------------#
    #                  CREATE VARIABLES                  #
    #----------------------------------------------------#
    
    G_var_vec = {}
    for element in list_of_eq_indices:
        if element == map_table[-1][-1]:
            G_var_vec[element] = 0.0 # Zeros form orthogonal projectors
        else:
            G_var_vec[element] = cp.Variable()
               
    #--------------------------------------------------#
    #                  BUILD MATRICES                  #
    #--------------------------------------------------#
    
    G = {}
    lis = []
    for r in range(len(G_new)):
        lis += [[]]
        for c in range(len(G_new)):
            lis[r] += [G_var_vec[G_new[r][c]]]
    G = cp.bmat(lis)
    
    #------------------------------------------------------#
    #                  CREATE CONSTRAINTS                  #
    #------------------------------------------------------#
    
    ct = []
                        
    # Normalisation constraints --------------------------------------------------------------
    for z in range(nZ):
        map_table_copy = map_table[:]
        
        identities = [ term[0] for term in map_table_copy]
        norm_cts = normalisation_contraints(w_M[z],identities)
        
        for gg in range(len(norm_cts)):
            the_elements = [fmap(map_table,norm_cts[gg][jj]) for jj in range(nB+1) ]
            an_element_is_not_in_the_list = False
            for hhh in range(len(the_elements)):
                if the_elements[hhh] == 'ERROR: The value does not appear in the mapping rule':
                    an_element_is_not_in_the_list = True
            if an_element_is_not_in_the_list == False:
                ct += [ sum([ G_var_vec[fmap(map_table,norm_cts[gg][jj])] for jj in range(nB) ]) == G_var_vec[fmap(map_table,norm_cts[gg][nB])] ]
    # ----------------------------------------------------------------------------------------
    
    # Positivity of tracial matrices and localising matrices
    ct += [G >> 0.0]
            
    # Some specific constraints in each corr matrix  -- G
   
    # Rank-1 projectors
    ct += [ G_var_vec[fmap(map_table,[0])] == d**2 ]
    ct += [ G_var_vec[fmap(map_table,[w_R[x0][x1]])] == d for x0 in range(nX0) for x1 in range(nX1)]
    ct += [ G_var_vec[fmap(map_table,[w_S[y0][y1]])] == d for y0 in range(nY0) for y1 in range(nY1)]
    #ct += [ G_var_vec[fmap(map_table,[w_M[z][b]])] == 1.0 for b in range(nB) for z in range(nZ)]
    
    ct += [ G_var_vec[fmap(map_table,[w_R[x0][x1],w_S[y0][y1]])] == 1.0 for x0 in range(nX0) for x1 in range(nX1) for y0 in range(nY0) for y1 in range(nY1)]
   
    #----------------------------------------------------------------#
    #                  RUN THE SDP and WRITE OUTPUT                  #
    #----------------------------------------------------------------#
    
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
                        
                        Sd = Sd + G_var_vec[fmap(map_table,[w_S[y0][y1],w_R[x0][x1],w_M[z][w]])]/(d**4*(1.0+d))
                        

    obj = cp.Maximize(Sd)
    prob = cp.Problem(obj,ct)

    output = []

    try:
        mosek_params = {
                "MSK_DPAR_INTPNT_CO_TOL_REL_GAP": 1e-1
            }
        prob.solve(solver='MOSEK',verbose=True, mosek_params=mosek_params)

    except SolverError:
        something = 10
        
    return Sd.value


#------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------#
#                                        Main code                                         #
#------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------#

d = 5
level = 1
[nX0,nX1,nY0,nY1] = [d,d,d,d]
[nZ,nB] = [d+1,d]

#Locate the variables from the matrix of correlations
w_id = np.zeros(1,dtype=int)

cc = 0
w_id[0] = cc

#First level monomials
w_rho = np.zeros((nX0,nX1),dtype=int)
w_sigma = np.zeros((nY0,nY1),dtype=int)
w_M = np.zeros((nZ,nB),dtype=int)

# Track operators in the tracial matrix
w_R = [] # Rho
w_S = [] # Sigma
w_M = [] # Measurement

S_1 = [] # List of first order elements
cc = 1


for x0 in range(nX0):
    w_R += [[]]
    for x1 in range(nX1):
        S_1 += [cc]
        w_R[x0] += [cc]
        cc += 1

for y0 in range(nY0):
    w_S += [[]]
    for y1 in range(nY1):
        S_1 += [cc]
        w_S[y0] += [cc]
        cc += 1
        
for z in range(nZ):
    w_M += [[]]
    for b in range(nB):
        S_1 += [cc]
        w_M[z] += [cc]
        cc += 1
        
S_high = [] # Uncomment if we only allow up to some 2nd order elements in the hierarchy  

# Second order elements
some_second = True
if some_second == True:
    
    for x0 in range(nX0):
        for x1 in range(nX1):
            for z in range(nZ):
                for b in range(nB):
                    S_high += [[w_R[x0][x1],w_M[z][b]]]
                    
    #for y0 in range(nY0):
    #    for y1 in range(nY1):
    #        for z in range(nZ):
    #            for b in range(nB):
    #                S_high += [[w_S[y0][y1],w_M[z][b]]]
    
    
    #for y0 in range(nY0):
    #    for y1 in range(nY1):
    #        for x0 in range(nX0):
    #            for x1 in range(nX1):
    #                S_high += [[w_S[y0][y1],w_R[x0][x1]]]
            
# Set the operational rules within the SDP relaxation
list_states = [] # operators that do not commute with anything 
#list_states += [w_M[z][b] for z in range(nZ) for b in range(nB)]

rank_1_projectors = []
rank_1_projectors += [w_R[x0][x1] for x0 in range(nX0) for x1 in range(nX1)]
rank_1_projectors += [w_S[y0][y1] for y0 in range(nY0) for y1 in range(nY1)]
rank_1_projectors += [w_M[z][b] for z in range(nZ) for b in range(nB)]

orthogonal_projectors = []
orthogonal_projectors += [w_M[z] for z in range(nZ)]

commuting_variables = [] # commuting elements (wxcept with elements in "list_states"
#commuting_variables += [w_R[x0][x1] for x0 in range(nX0) for x1 in range(nX1)]
#commuting_variables += [w_S[y0][y1] for y0 in range(nY0) for y1 in range(nY1)]

print('Rank-1 projectors',rank_1_projectors)
print('Orthogonal projectors',orthogonal_projectors)
print('commuting elements',commuting_variables)

# Collect rules and generate SDP relaxation matrix
start = time.process_time()
[G_new,map_table,S,list_of_eq_indices,Mexp] = MomentMatrix(S_1,[],S_high,rank_1_projectors,orthogonal_projectors,commuting_variables,list_states)
end = time.process_time()
print(G_new)
print('Gamma matrix generated in',end-start,'s')
print('Matrix size:',np.shape(G_new))

monomials = [w_R,w_S,w_M]
gamma_matrix_els = [G_new,map_table,S,list_of_eq_indices,Mexp]   
             
out_game = Game(nX0,nX1,nY0,nY1,nZ,nB,monomials,gamma_matrix_els,d)
              
print(out_game,2.0/(1.0+d))          
              
              
                            
              
              
              
                            
              
              
              
                            
              
              
              
                            
              
              
              
                            
              
              
              
                            
              
              
              
                            
              
              
              