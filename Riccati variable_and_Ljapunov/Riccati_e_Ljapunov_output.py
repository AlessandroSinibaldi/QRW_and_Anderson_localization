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
from scipy.stats.morestats import Variance


im = complex(0, 1)
seed = 1234

chi = 0

control = 3 #funzioni differenti: 0. calcolare lambda con diverso disordine, 1. distribuzione di Riccati (file non caricato nel repository perché troppo pesante),
            #2. xi disordine completo, 3. lambda(z) disordine debole

if(control == 0):
    W = 2*m.pi
    n_energies = 100
    beta = np.linspace(0, 2*m.pi, n_energies)
    z = np.array([cmt.exp(-im*x) for x in beta]) 
    Ljapunov = np.zeros(n_energies)
    theta = m.pi/6
    r0 = 1
    N = 10000000

    def iterate(z, Ljapunov, theta, n, seed):
        rnd.seed(seed)
        r = r0
        for l in range(0, N):
            chi = rnd.uniform(-W/2, W/2)
            r = ((1/z)+z*cmt.exp(-im*chi)-(m.cos(theta/2)*cmt.exp(-im*chi))/r)/(m.cos(theta/2))
            Ljapunov[n] += m.log(abs(r))/N


    for j in range(0, len(z)):
        iterate(z[j], Ljapunov, theta, j, seed)
        seed+=1
    print(100*(np.max(Ljapunov)-np.min(Ljapunov))/(np.max(Ljapunov)+np.min(Ljapunov)))

if(control == 1):
    W = 2*m.pi
    N = 10000000
    r_histo = np.zeros(N, dtype=complex)
    r0 = 1
    theta = m.pi/3

    def iterate(theta):
        r = r0
        for l in range(0, N):
            chi = rnd.uniform(-W/2, W/2)
            r = (1+cmt.exp(-im*chi)-(m.cos(theta/2)*cmt.exp(-im*chi))/r)/(m.cos(theta/2))
            r_histo[l] = r
        return r_histo
    
    r_histo = iterate(theta)

    fig=plt.figure()
    ax = fig.gca(projection = '3d')
    x = r_histo.real
    y = r_histo.imag

    # Scrivere su file

    f = open(os.path.join(sys.path[0], "Riccati_pi3_N10^7.txt"), 'w')   
    for j in range(int(N/10), N):
        f.write ("{:.5f}".format(float(x[j])))
        f.write("\n")
        f.write ("{:.5f}".format(float(y[j])))
        f.write("\n")

if(control == 2):
    W = 2*m.pi
    n_thetas = 100
    thetas = np.linspace(0.05, m.pi/2-0.1, n_thetas)
    Ljapunov = np.zeros(n_thetas)
    r0 = 1
    N = 10000000

    def iterate(Ljapunov, theta, n, seed):
        rnd.seed(seed)
        r = r0
        for l in range(0, N):
            chi = rnd.uniform(-W/2, W/2)
            r = (1+cmt.exp(-im*chi)-(m.cos(theta)*cmt.exp(-im*chi))/r)/(m.cos(theta))
            if(l >= int(N/10)):
                Ljapunov[n] += m.log(abs(r))/(N-N/10)

    for j in range(0, len(thetas)):
        iterate(Ljapunov, thetas[j], j, seed)
        seed+=1
    localization_lenght = [1/x for x in Ljapunov]

    f = open(os.path.join(sys.path[0], "Ljapunov_QRW_10^6.txt"), 'w')
    f.write(str(n_thetas))
    f.write("\n")
    for j in range(0, n_thetas):
        f.write ("{:.16f}".format(float(thetas[j])))
        f.write("\n")
        f.write ("{:.16f}".format(float(localization_lenght[j])))
        f.write("\n")

if(control == 3):
    W = 0.3
    variance = (W**2)/3
    n_betas = 100
    thetas = np.array([m.pi/7, m.pi/4, m.pi/3])
    n_thetas = len(thetas)
    Ljapunov = np.zeros((n_thetas, n_betas))
    betas = np.zeros((n_thetas, n_betas))
    z = np.zeros((n_thetas, n_betas), dtype=complex)
    for j in range(0, n_thetas):
        betas[j] = np.linspace(thetas[j]+0.01, m.pi - thetas[j] - 0.01, n_betas)
        z[j] = np.array([cmt.exp(-im*x) for x in betas[j]]) 
    r0 = 1
    N = 10000000

    def iterate(Ljapunov, theta, a, z, b, seed):
        rnd.seed(seed)
        r = r0
        for l in range(0, N):
            phi_up = rnd.uniform(-W, W)
            phi_down = rnd.uniform(-W, W)
            chi = phi_up + phi_down
            r = ((1/z)+z*cmt.exp(-im*chi)-(m.cos(theta)*cmt.exp(-im*chi))/r)/(m.cos(theta))
            if(l >= int(N/10)):
                Ljapunov[a][b] += m.log(abs(r))/(N-N/10)

    for j in range(0, n_thetas):
        for k in range(0, n_betas):
            iterate(Ljapunov, thetas[j], j, z[j][k], k, seed)
            seed+=1

    Ljapunov =  np.array([x/variance for x in Ljapunov])
    
    f = open(os.path.join(sys.path[0], "Ljapunov_vs_beta.txt"), 'w')
    f.write(str(n_betas))
    f.write("\n")
    f.write(str(n_thetas))
    f.write("\n")
    for j in range(0, n_thetas):
        f.write ("{:.16f}".format(float(thetas[j])))
        f.write("\n")
        for k in range(0, n_betas):
            f.write ("{:.16f}".format(float(betas[j][k])))
            f.write("\n")
            f.write ("{:.16f}".format(float(Ljapunov[j][k])))
            f.write("\n")
