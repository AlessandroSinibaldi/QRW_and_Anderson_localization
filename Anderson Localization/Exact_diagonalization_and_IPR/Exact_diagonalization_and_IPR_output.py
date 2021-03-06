import math as m
import matplotlib.pyplot as plt
import numpy as np
import random as rnd
from numpy import linalg as LA
import os 
import sys


#dim = 50
#dim = 60
#dim = 75
#dim = 65
#dim = 85
#dim = 4
#dim = 100
#dim = 150
#dim = 125
dim = 200
#dim = 250
#dim = 300
#dim = 350
#dim = 400
#dim = 500
g = -1
W = 0
seed = 1234
rnd.seed(seed)

repetitions = 2000

control = False         #False: autostati, True: IPR output

def create_Hamiltonian(Hamiltonian):
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
    return Hamiltonian

def resolve_equation(Hamiltonian):
    eigenvalues, eigenvectors = LA.eig(Hamiltonian)
    eigenvalues_ = np.sort(eigenvalues)
    indexes = eigenvalues.argsort()
    counter = 0 
    for idx in indexes:
        eigenvectors_[:, counter] = eigenvectors[:, idx]
        counter+=1
    probabilities = np.array([x**2 for x in eigenvectors_])
    return eigenvalues_, eigenvectors_, probabilities

def compute_IPR(eigenvectors):
    IPR = np.zeros(dim)
    for i in range(0, dim):
        for l in range(0, dim):
            IPR[i] += eigenvectors[l, i]**4
    return IPR

if(control == False):
    Hamiltonian = np.zeros((dim, dim))    
    eigenvectors_ = np.zeros((dim, dim))
    positions = np.array([x for x in range(0, dim)])
    create_Hamiltonian(Hamiltonian)
    eigenvalues, eigenvectors, probabilities = resolve_equation(Hamiltonian)

    plt.figure()
    plt.style.use('default')
    ax = plt.gca()
    ax.set_facecolor('#F2F3F4')
    title = "Localizzazione degli autostati ($\mathit{W}$ = 0)"
    plt.xlabel("$\mathit{l}$")
    plt.ylabel("Probabilit√†")

    plt.plot(positions, probabilities[:, 1], label = '$\mathit{E_{1}}$')
    plt.plot(positions, probabilities[:, 2], label = '$\mathit{E_{1}}$')
    plt.plot(positions, probabilities[:, 3], label = '$\mathit{E_{2}}$')
    plt.plot(positions, probabilities[:, 4], label = '$\mathit{E_{2}}$')
    plt.plot(positions, probabilities[:, 5], label = '$\mathit{E_{3}}$')
    plt.plot(positions, probabilities[:, 6], label = '$\mathit{E_{3}}$')
    plt.legend(loc = 'upper right', ncol = 2)
    print(eigenvalues)

    plt.grid(color='white', linestyle = 'dashed')
    hfont = {'fontname':'monospace'}
    plt.title(title)
    plt.show()

if(control == True):
    IPR_mean= np.zeros(dim)
    eigenvalues_mean = np.zeros(dim)
    Hamiltonian = np.zeros((dim, dim))
    eigenvectors_ = np.zeros((dim, dim))

    for k in range (0, repetitions):
        create_Hamiltonian(Hamiltonian)
        eigenvalues, eigenvectors, probabilities = resolve_equation(Hamiltonian)
        eigenvalues_mean = np.add(eigenvalues_mean, [x/repetitions for x in eigenvalues])
        IPR = compute_IPR(eigenvectors)
        IPR_mean = np.add(IPR_mean, [x/repetitions for x in IPR])
        seed += 1
        rnd.seed(seed)

    IPR_mean_ = [1/x for x in IPR_mean]

    f = open(os.path.join(sys.path[0], "IPR_600.txt"), 'w')     # Quando si cambia dim, √® necessario cambiare 
    f.write(str(dim))                                           # anche il nome del file di output
    f.write("\n")                                               
    for j in range(0, dim):
        f.write ("{:.16f}".format(float(IPR_mean_[j])))
        f.write("\n")
        f.write ("{:.16f}".format(float(eigenvalues_mean[j])))
        f.write("\n")

