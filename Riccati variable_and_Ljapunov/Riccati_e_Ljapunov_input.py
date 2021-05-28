import math as m
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import random as rnd
from numpy.linalg import matrix_power
from numpy import linalg as LA
from scipy import optimize
from pylab import *
import os 
import sys
from scipy.stats import chisquare

control = 3 #funzioni differenti: 1. graficare distribuzione di Riccati (ma è stato fatto su ROOT),
            #2. graficare xi disordine completo, 3. graficare lambda(z) disordine debole

if(control == 1):
    f = open(os.path.join(sys.path[0], "Riccati_pi_3_N1000_entries1000.txt"), 'r')

    n_entries = int(next(f))
    x = np.zeros(2*n_entries, dtype=complex)
    y = np.zeros(2*n_entries, dtype=complex)

    for j in range(0, n_entries):
        x[j] = float(next(f))
        y[j] = float(next(f))
    f.close()  

    f = open(os.path.join(sys.path[0], "Riccati_2pi_3_N1000_entries1000.txt"), 'r')

    n_entries = int(next(f))

    for j in range(n_entries, 2*n_entries):
        x[j] = float(next(f))
        y[j] = float(next(f))
    f.close() 

    n_bins = 100

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    hist, xedges, yedges = np.histogram2d(x, y, bins=n_bins, range=[[0, 4], [-2, 2]])

    # Construct arrays for the anchor positions of the 16 bars.
    xpos, ypos = np.meshgrid(xedges[:-1], yedges[:-1], indexing="ij")
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = 0

    # Construct arrays with the dimensions for the 16 bars.
    dx = dy = (4/n_bins) * np.ones_like(zpos)
    dz = hist.ravel()

    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average', color='blue')
    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average', color='red')

    plt.show()

if(control == 2):
    f = open(os.path.join(sys.path[0], "Ljapunov_QRW.txt"), 'r')

    n_thetas = int(next(f))
    thetas = np.zeros(n_thetas)
    localization_lenght = np.zeros(n_thetas)

    for j in range(0, n_thetas):
        thetas[j] = float(next(f))
        localization_lenght[j] = float(next(f))
    f.close()  
    localization_lenght_theory = np.array([1/abs(m.log(m.cos(x))) for x in thetas])

    plt.figure()
    plt.style.use('default')
    ax = plt.gca()
    ax.set_facecolor('#F2F3F4')
    title = "Lunghezza di localizzazione"
    plt.title(title)
    plt.xlabel(r'$\theta$')
    plt.xlim([0, m.pi/2])
    ax.set_xticks([0, m.pi/8, m.pi/4, 3*m.pi/8, m.pi/2]) 
    ax.set_xticklabels(['0', '$\pi/8$', '$\pi/4$', '$3\pi/8$', '$\pi/2$'])
    plt.ylabel(r'$\xi$')
    ax.set_yscale('log')
    plt.plot(thetas, localization_lenght_theory, label = "Teoria")
    plt.plot(thetas, localization_lenght, 's', marker = 'o', fillstyle = 'none', label = "Simulazioni")
    plt.legend(loc = 'upper right')
    plt.show()
    plt.figure()
    print(chisquare(localization_lenght_theory, localization_lenght))

if(control == 3):
    f = open(os.path.join(sys.path[0], "Ljapunov_vs_beta.txt"), 'r')

    n_betas = int(next(f))
    n_thetas = int(next(f))
    thetas = np.zeros(n_thetas)
    betas = np.zeros((n_thetas, n_betas))
    Ljapunov = np.zeros((n_thetas, n_betas))
    Ljapunov_theory = np.zeros((n_thetas, n_betas))
    for j in range(0, n_thetas):
        thetas[j] = float(next(f))
        for k in range(0, n_betas):
            betas[j][k] = float(next(f))
            Ljapunov[j][k] = float(next(f))

    f.close()

    for j in range(0, n_thetas):
        for k in range(0, n_betas):
            Ljapunov_theory[j][k] = (m.sin(thetas[j])**2)/(4*(m.cos(thetas[j])**2 - m.cos(betas[j][k])**2))

    plt.figure()
    plt.style.use('default')
    ax = plt.gca()
    ax.set_facecolor('#F2F3F4')
    title = "Dipendenza spettrale dell'esponente di Ljapunov"
    plt.title(title)
    plt.xlabel(r'$\beta$')
    plt.xlim([0, m.pi])
    ax.set_xticks([0, m.pi/8, m.pi/4, 3*m.pi/8, m.pi/2, 5*m.pi/8, 3*m.pi/4, 7*m.pi/8, m.pi]) 
    ax.set_xticklabels(['0', '$\pi/8$', '$\pi/4$', '$3\pi/8$', '$\pi/2$', '$5\pi/8$', '$3\pi/4$', '$7\pi/8$', '$\pi$'])
    plt.ylabel(r'$\lambda / <\phi^{2}>$')
    ax.set_yscale('log')
    p1, = plt.plot(betas[2][7:93], Ljapunov_theory[2][7:93], label = r"$\theta = \pi/3$")
    p2, = plt.plot(betas[1][5:95], Ljapunov_theory[1][5:95], label = r"$\theta = \pi/4$")
    p3, = plt.plot(betas[0][3:97], Ljapunov_theory[0][3:97], label = r"$\theta = \pi/7$")
    l1 = plt.legend((p1, p2, p3), [r"$\theta = \pi/3$", r"$\theta = \pi/4$", r"$\theta = \pi/7$"], title = "Teoria", loc ='upper left')
    plt.gca().add_artist(l1)

    p4, = plt.plot(betas[2][7:93], Ljapunov[2][7:93], 's', marker = 'o', fillstyle = 'none', label = r"$\theta = \pi/3$")
    p5, = plt.plot(betas[1][5:95], Ljapunov[1][5:95], 's', marker = 'o', fillstyle = 'none', label = r"$\theta = \pi/4$")
    p6, = plt.plot(betas[0][3:97], Ljapunov[0][3:97], 's', marker = 'o', fillstyle = 'none', label = r"$\theta = \pi/7$")
    plt.legend((p4, p5, p6), [r"$\theta = \pi/3$", r"$\theta = \pi/4$", r"$\theta = \pi/7$"], title = "Simulazioni", loc = "upper right")
    plt.show()
    plt.figure()