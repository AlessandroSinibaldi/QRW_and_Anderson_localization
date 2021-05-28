import math as m
import matplotlib.pyplot as plt
import numpy as np
import random as rnd
from numpy.linalg import matrix_power
from numpy import linalg as LA
from scipy import optimize
from pylab import *
import os
import sys

IPR_vector = []
eigenvalues_vector = []
N = []
IPR_point = []

control = True      #True: IPR vs energia, False: IPR vs N con fit

filenames = ["IPR_50.txt", "IPR_60.txt", "IPR_65.txt", "IPR_75.txt", "IPR_85.txt", "IPR_95.txt", 
"IPR_100.txt", "IPR_125.txt", "IPR_150.txt", "IPR_200.txt", "IPR_250.txt", "IPR_300.txt", "IPR_350.txt",
"IPR_400.txt", "IPR_500.txt", "IPR_600.txt"]

for k in range(0, len(filenames)):
    f = open(os.path.join(sys.path[0], filenames[k]), 'r')
    dim = int(next(f))
    N.append(dim)
    IPR = np.zeros(dim)
    eigenvalues = np.zeros(dim)
    for j in range(0, dim):
        IPR[j] = float(next(f))
        eigenvalues[j] = float(next(f))
    IPR_vector.append(IPR)
    eigenvalues_vector.append(eigenvalues)
    f.close()

plt.figure()
plt.style.use('default')
ax = plt.gca()
ax.set_facecolor('#F2F3F4')
title = "IPR"
plt.title(title)

if(control == True):
    plt.xlabel("Energia")
    plt.ylabel("1/<IPR>")
    p1, = plt.plot(eigenvalues_vector[0], IPR_vector[0], label = '$\mathit{N} = 50$')
    p2, = plt.plot(eigenvalues_vector[1], IPR_vector[1], label = '$\mathit{N} = 60$')
    p3, = plt.plot(eigenvalues_vector[2], IPR_vector[2], label = '$\mathit{N} = 65$')
    p4, = plt.plot(eigenvalues_vector[3], IPR_vector[3], label = '$\mathit{N} = 75$')
    p5, = plt.plot(eigenvalues_vector[4], IPR_vector[4], label = '$\mathit{N} = 85$')
    p7, = plt.plot(eigenvalues_vector[5], IPR_vector[5], label = '$\mathit{N} = 100$')
    l1 = legend((p1, p2, p3, p4, p5, p7), ['$\mathit{N} = 50$', '$\mathit{N} = 60$', '$\mathit{N} = 65$', 
    '$\mathit{N} = 75$', '$\mathit{N} = 85$', '$\mathit{N} = 100$'], loc = 'upper left')
    plt.gca().add_artist(l1)

    p8, = plt.plot(eigenvalues_vector[6], IPR_vector[6], label = '$\mathit{N} = 125$')
    p9, = plt.plot(eigenvalues_vector[7], IPR_vector[7], label = '$\mathit{N} = 150$')
    p10, = plt.plot(eigenvalues_vector[8], IPR_vector[8], label = '$\mathit{N} = 200$')
    p11, = plt.plot(eigenvalues_vector[10], IPR_vector[10], label = '$\mathit{N} = 300$')
    p12, = plt.plot(eigenvalues_vector[12], IPR_vector[12], label = '$\mathit{N} = 400$')
    p13, = plt.plot(eigenvalues_vector[13], IPR_vector[13], label = '$\mathit{N} = 500$')
    plt.legend((p8, p9, p10, p11, p12, p13), ['$\mathit{N} = 125$', '$\mathit{N} = 150$', 
    '$\mathit{N} = 200$', '$\mathit{N} = 300$', '$\mathit{N} = 400$', '$\mathit{N} = 500$'], loc = 'upper right')    
    plt.grid(color='white', linestyle = 'dashed')
    hfont = {'fontname':'monospace'}
    #plt.savefig("IPR.jpeg")

if(control == False):
    for i in range(0, len(N)):
        IPR_point.append(IPR_vector[i][int(N[i]/2)])

    def fit_func(x, a, b):
        return a-b/x

    params, params_covariance = optimize.curve_fit(fit_func, N, IPR_point)
    print(params)

    plt.xlabel("N")
    plt.ylabel("1/<IPR>")
    plt.plot(N, IPR_point, 'ro', label = "Dati")
    N_ = []
    N_.append(45)
    N_ += N
    plt.plot(N_, fit_func(N_, params[0], params[1]), label = "Funzione di fit: a-b/N")
    plt.grid(color='white', linestyle = 'dashed')
    hfont = {'fontname':'monospace'}
    plt.title(title)
    plt.legend(loc = 'right')
    plt.show()
    out = open('IPR_vs_N.txt', 'w')
    for j in range(0, len(N)):
        out.write(str(N[j]))
        out.write("\t")
        out.write(str(IPR_point[j]))
        out.write("\n")        
plt.show()