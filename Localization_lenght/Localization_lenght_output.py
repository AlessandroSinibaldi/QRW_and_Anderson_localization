import math as m
import matplotlib.pyplot as plt
import numpy as np
import random as rnd
from numpy.linalg import matrix_power
from numpy import linalg as LA
import os
import sys 

g = -1
#W = 1
#W = 1.5
#W = 2
#W = 2.5
W = 3
dim = 100
seed = 1234

repetitions = 100

eigenvalues_mean = np.zeros(dim)
log_mean = np.zeros(dim)
Hamiltonian = np.zeros((dim, dim))    
eigenvectors_ = np.zeros((dim, dim))

def create_Hamiltonian(Hamiltonian, seed, W):
    seed += 1
    rnd.seed(seed)
    for l in range(0, dim):
        Hamiltonian[l][l] = rnd.uniform(-W/2, W/2)
        if(l-1 < 0):
            Hamiltonian[l][l+1] = -g
            Hamiltonian[l][dim-1] = -g
        if(l+1 > dim-1):
            Hamiltonian[l][l-1] = -g
            Hamiltonian[l][0] = -g
        else:
            Hamiltonian[l][l+1] = -g
            Hamiltonian[l][l-1] = -g   
    return Hamiltonian, seed

def resolve_equation(Hamiltonian):
    eigenvalues, eigenvectors = LA.eig(Hamiltonian)
    eigenvalues_ = np.sort(eigenvalues)
    indexes = eigenvalues.argsort()
    counter = 0 
    for idx in indexes:
        eigenvectors_[:, counter] = eigenvectors[:, idx]
        counter+=1
    return eigenvalues_, eigenvectors_

for n in range(0, repetitions):
    Hamiltonian, seed = create_Hamiltonian(Hamiltonian, seed, W)
    eigenvalues, eigenvectors = resolve_equation(Hamiltonian)
    eigenvalues_mean = np.add(eigenvalues_mean, [x/repetitions for x in eigenvalues])
    log_mean = np.add(log_mean, [m.log(abs(x))/repetitions for x in eigenvectors[dim-1, :]])
localization_lenght = np.array([-dim/x for x in log_mean])

# Scrivere su file
f = open(os.path.join(sys.path[0], "localization_lenght_W3.txt"), 'w')     # Quando si cambia W, bisogna cambiare il nome del file su cui scrivere
for j in range(0, dim):
    f.write ("{:.16f}".format(float(localization_lenght[j])))
    f.write("\n")
    f.write ("{:.16f}".format(float(eigenvalues_mean[j])))
    f.write("\n")

