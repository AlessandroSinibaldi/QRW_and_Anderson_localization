import cmath as cmt
import math as m
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import random as rnd
from numpy import linalg as LA
from scipy import linalg as LAs
import os 
import sys
from mpl_toolkits.mplot3d import Axes3D

dim = 200
phi = m.pi/2
chi = 0
theta = m.pi/2
i = complex(0, 1)
seed = 1234
rnd.seed(seed)

#W = 0
#W = m.pi/24
#W = m.pi/6
#W = 2*m.pi/3
W = 2*m.pi

control = False

a = cmt.exp(i*chi)*m.cos(theta/2)
b = cmt.exp(i*phi)*m.sin(theta/2)
c = -cmt.exp(-i*phi)*m.sin(theta/2)
d = cmt.exp(-i*chi)*m.cos(theta/2)
a_ = 0
b_ = 0 
c_ = 0
d_ = 0

N = 50
Ws = np.linspace(0, 2*m.pi, N)
thetas = np.linspace(0, m.pi, N)
IPR = np.zeros((N, N)) 
repetitions = 10

def create_Hamiltonian(Hamiltonian):
    for l in range(0, 2*dim):
        if(l%2 == 0):
            phi_up = rnd.uniform(-W/2, W/2)
            a_ = a*cmt.exp(i*phi_up)
            b_ = b*cmt.exp(i*phi_up)
            if(l==0):
                Hamiltonian[l][2*dim-2] = a_
                Hamiltonian[l][2*dim-1] = b_ 
            else:
                Hamiltonian[l][l-2] = a_
                Hamiltonian[l][l-1] = b_
        if(l%2 != 0):
            phi_down = rnd.uniform(-W/2, W/2)
            c_ = c*cmt.exp(i*phi_down)
            d_ = d*cmt.exp(i*phi_down)
            if(l==2*dim-1):
                Hamiltonian[l][0] = c_
                Hamiltonian[l][1] = d_
            else:
                Hamiltonian[l][l+1] = c_
                Hamiltonian[l][l+2] = d_
    return Hamiltonian

def create_Hamiltonian_(Hamiltonian, Ws, thetas):
    phi = m.pi/2
    chi = 0 
    a_i = cmt.exp(i*chi)*m.cos(thetas/2)
    b_i = cmt.exp(i*phi)*m.sin(thetas/2)
    c_i = -cmt.exp(-i*phi)*m.sin(thetas/2)
    d_i = cmt.exp(-i*chi)*m.cos(thetas/2)

    for l in range(0, 2*dim):
        if(l%2 == 0):
            phi_up = rnd.uniform(-Ws/2, Ws/2)
            a_i_ = a_i*cmt.exp(i*phi_up)
            b_i_ = b_i*cmt.exp(i*phi_up)
            if(l==0):
                Hamiltonian[l][2*dim-2] = a_i_
                Hamiltonian[l][2*dim-1] = b_i_
            else:
                Hamiltonian[l][l-2] = a_i_
                Hamiltonian[l][l-1] = b_i_
        if(l%2 != 0):
            phi_down = rnd.uniform(-Ws/2, Ws/2)
            c_i_ = c_i*cmt.exp(i*phi_down)
            d_i_ = d_i*cmt.exp(i*phi_down)
            if(l==2*dim-1):
                Hamiltonian[l][0] = c_i_
                Hamiltonian[l][1] = d_i_

            else:
                Hamiltonian[l][l+1] = c_i_
                Hamiltonian[l][l+2] = d_i_
    return Hamiltonian

def resolve_equation(Hamiltonian): 
    eigenvalues, eigenvectors = LA.eig(Hamiltonian)
    eigenvalues_ = np.sort(eigenvalues)
    indexes = eigenvalues.argsort()
    counter = 0 
    for idx in indexes:
        eigenvectors_[:, counter] = eigenvectors[:, idx]
        counter+=1
    probabilities = np.array([abs(x)**2 for x in eigenvectors_])
    return eigenvalues_, eigenvectors_, probabilities

def resolve_equation_(Hamiltonian):
    eigenvalues, eigenvectors = LAs.eig(Hamiltonian, left=False, right=True)
    eigenvalues_ = np.sort(eigenvalues)
    indexes = eigenvalues.argsort()
    counter = 0 
    for idx in indexes:
        eigenvectors_[:, counter] = eigenvectors[:, idx]
        counter+=1
    probabilities = np.array([abs(x)**2 for x in eigenvectors_])
    return eigenvalues_, eigenvectors_, probabilities


def compute_norm(state):
    norm = 0
    for s in range(0, len(state)):
        norm += state[s]
    return norm

def compute_norm_squared(state):
    norm = 0
    for s in range(0, len(state)):
        norm += abs(state[s])**2
    return norm

def compute_IPR(probabilities):
    IPR = np.zeros(2*dim)
    for i in range(0, 2*dim):
        for l in range(0, dim):
            IPR[i] += probabilities[l, i]**2
    return IPR

if(control == False):
    Hamiltonian = np.zeros((2*dim, 2*dim), dtype=complex)
    eigenvectors_ = np.zeros((2*dim, 2*dim), dtype=complex)
    positions = np.array([x for x in range(0, dim)])
    Hamiltonian = create_Hamiltonian(Hamiltonian)
    eigenvalues, eigenvectors, probabilities = resolve_equation(Hamiltonian)
    probabilities_tot = np.zeros((dim, 2*dim))
    for k in range(0, dim):
        probabilities_tot[k, :] = np.add(probabilities[2*k,:], probabilities[2*k+1,:])
    
    plt.figure()
    plt.style.use('default')
    ax = plt.gca()
    ax.set_facecolor('#F2F3F4')
    title = "Localizzazione degli autostati ($\mathit{W}$ = 0)"
    plt.title(title)
    plt.xlabel("$\mathit{n}$")
    plt.ylabel("Probabilità")

    plt.plot(positions, probabilities_tot[:, 26])
    plt.plot(positions, probabilities_tot[:, 24])
    plt.plot(positions, probabilities_tot[:, 6])
    plt.plot(positions, probabilities_tot[:, 16])
    plt.plot(positions, probabilities_tot[:, 12])
    plt.plot(positions, probabilities_tot[:, 18])
    plt.show()
    
    plt.figure()
    plt.style.use('default')
    ax = plt.gca()
    ax.set_facecolor('#F2F3F4')
    title = "Autovalori ($\mathit{W}$ = 2$\pi$/3)"
    plt.xlabel("$\mathit{Re(z)}$")
    plt.ylabel("$\mathit{Im(z)}$")
    plt.scatter(eigenvalues.real, eigenvalues.imag, facecolors = 'none', edgecolors='r')
    hfont = {'fontname':'monospace'}
    plt.title(title)
    plt.xlim([-1.1, 1.1])
    plt.ylim([-1.1, 1.1]) 
    
    plt.show()

if(control == True):
    for e in range(0, N):
        for j in range(0, N):
            Hamiltonian = np.zeros((2*dim, 2*dim), dtype=complex)
            eigenvectors_ = np.zeros((2*dim, 2*dim), dtype=complex)
            IPR_min = 0
            for t in range (0, repetitions):
                Hamiltonian = create_Hamiltonian_(Hamiltonian, Ws[e], thetas[j])
                eigenvalues, eigenvectors, probabilities = resolve_equation(Hamiltonian)
                probabilities_tot = np.zeros((dim, 2*dim))
                for k in range(0, dim):
                    probabilities_tot[k, :] = np.add(probabilities[2*k,:], probabilities[2*k+1,:])
                IPR_min += np.min(compute_IPR(probabilities_tot))/(repetitions)
                seed += 1
                rnd.seed(seed)
            IPR[e][j] = IPR_min
    IPR_ = np.array([1/(x*dim) for x in IPR])

    f = open(os.path.join(sys.path[0], "IPR_QRW.txt"), 'w')     # Quando si cambia dim, è necessario cambiare
    f.write(str(N))                                             # anche il nome del file di output 
    f.write("\n")                                              
    for j in range(0, N):
        f.write ("{:.16f}".format(float(Ws[j])))
        f.write("\n")
        f.write ("{:.16f}".format(float(thetas[j])))
        f.write("\n")
    for l in range(0, N):
        for k in range(0, N):
            f.write ("{:.16f}".format(float(IPR_[l][k])))
            f.write("\n")
