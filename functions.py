#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 12:52:48 2024

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

#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------

def deltaF(x,y):
    if x == y:
        return 1.0
    else:
        return 0.0
    
#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------

def random_unitary_matrix(n):
    """Generates a random n x n unitary matrix"""
    XX = np.random.randn(n, n) + 1j*np.random.randn(n, n)  # Generate a random n x n complex matrix
    Q, R = np.linalg.qr(XX)  # Perform QR decomposition
    D = np.diagonal(R)  # Extract the diagonal elements of R
    P = D / np.abs(D)  # Compute the phase of the diagonal elements
    U = Q @ np.diag(P)  # Compute the unitary matrix
    return U

#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------

#maximally entangled state in dimension d in each part
def max_ent(d):
    
    state = np.zeros(d**2)
    
    for i in range(d):
        for j in range(d):
            index = int(j*d**0.0 + i*d**1.0)
            if i==j:
                state[index] = 1.0
            
    return np.kron([state],np.transpose([state]))/d

#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------

def permute(d):
    #n: number of hilbert spaces: 4
    #d: dimension of Hilbert spaces
    n=4
    
    #vec of indices
    in_vec = []
    i=0
    for num in range(n):
        in_vec.insert(0,d**(i))
        i += 1
    
    O = np.zeros((d**n,d**n))
    for i in range(d):
        for j in range(d):
            for k in range(d):
                for l in range(d):
                    index_ijkl = np.sum(in_vec * np.array([i,j,k,l]))
                    index_ikjl = np.sum(in_vec * np.array([i,k,j,l]))
                    O[index_ijkl][index_ikjl] = 1.0

        
    return O












































