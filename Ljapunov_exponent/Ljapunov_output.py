import math as m
import matplotlib.pyplot as plt
import numpy as np
import random as rnd
from numpy.linalg import matrix_power
from numpy import linalg as LA
import os
import sys

g = -1
W = 3
seed = 1234

N_energies = 500
N = 1000    #con N arrivo fino a n+1 = N-1 e n = N-2
repetitions = 1000
initial_psi = np.zeros(N)
initial_psi[0] = 0.01
initial_psi[1] = 0
psi = initial_psi
log_mean = np.zeros(N_energies)
energies = np.linspace(-2, 2, N_energies)

def transfer(psi, energy, seed):
    for i in range(2, N):
        seed += 1
        rnd.seed(seed)
        epsilon = rnd.uniform(-W/2, W/2)
        psi[i] = ((epsilon - energy)/g)*psi[i-1] - psi[i-2]
    return psi, seed 

for j in range(0, N_energies):
    for i in range(0, repetitions):
        psi, seed = transfer(psi, energies[j], seed)
        log_mean[j] += m.log(m.sqrt(psi[N-1]**2 + psi[N-2]**2))/repetitions
        psi = initial_psi

localization_lenght = [(N-2)/x for x in log_mean]

f = open(os.path.join(sys.path[0], "Ljapunov_3.txt"), 'w')
for j in range(0, N_energies):
    f.write("{:.16f}".format(float(localization_lenght[j])))
    f.write("\n")
    f.write("{:.16f}".format(float(energies[j])))
    f.write("\n")

