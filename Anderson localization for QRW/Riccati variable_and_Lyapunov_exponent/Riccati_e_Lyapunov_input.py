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

control = True                         # True: xi disordine completo, False: lambda(z) disordine debole

if(control == True):
    f = open(os.path.join(sys.path[0], "Lyapunov_QRW.txt"), 'r')

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
    print(chisquare(localization_lenght, localization_lenght_theory))
    
if(control == False):
    f = open(os.path.join(sys.path[0], "Lyapunov_vs_beta.txt"), 'r')

    n_betas = int(next(f))
    n_thetas = int(next(f))
    thetas = np.zeros(n_thetas)
    betas = np.zeros((n_thetas, n_betas))
    Lyapunov = np.zeros((n_thetas, n_betas))
    Lyapunov_theory = np.zeros((n_thetas, n_betas))
    for j in range(0, n_thetas):
        thetas[j] = float(next(f))
        for k in range(0, n_betas):
            betas[j][k] = float(next(f))
            Lyapunov[j][k] = float(next(f))

    f.close()

    for j in range(0, n_thetas):
        for k in range(0, n_betas):
            Lyapunov_theory[j][k] = (m.sin(thetas[j])**2)/(4*(m.cos(thetas[j])**2 - m.cos(betas[j][k])**2))

    plt.figure()
    plt.style.use('default')
    ax = plt.gca()
    ax.set_facecolor('#F2F3F4')
    title = "Dipendenza spettrale dell'esponente di Lyapunov"
    plt.title(title)
    plt.xlabel(r'$\beta$')
    plt.xlim([0, m.pi])
    ax.set_xticks([0, m.pi/8, m.pi/4, 3*m.pi/8, m.pi/2, 5*m.pi/8, 3*m.pi/4, 7*m.pi/8, m.pi]) 
    ax.set_xticklabels(['0', '$\pi/8$', '$\pi/4$', '$3\pi/8$', '$\pi/2$', '$5\pi/8$', '$3\pi/4$', '$7\pi/8$', '$\pi$'])
    plt.ylabel(r'$\lambda / <\phi^{2}>$')
    ax.set_yscale('log')
    p1, = plt.plot(betas[2][7:93], Lyapunov_theory[2][7:93], label = r"$\theta = \pi/3$")
    p2, = plt.plot(betas[1][5:95], Lyapunov_theory[1][5:95], label = r"$\theta = \pi/4$")
    p3, = plt.plot(betas[0][3:97], Lyapunov_theory[0][3:97], label = r"$\theta = \pi/7$")
    l1 = plt.legend((p1, p2, p3), [r"$\theta = \pi/3$", r"$\theta = \pi/4$", r"$\theta = \pi/7$"], title = "Teoria", loc ='upper left')
    plt.gca().add_artist(l1)

    p4, = plt.plot(betas[2][7:93], Lyapunov[2][7:93], 's', marker = 'o', fillstyle = 'none', label = r"$\theta = \pi/3$")
    p5, = plt.plot(betas[1][5:95], Lyapunov[1][5:95], 's', marker = 'o', fillstyle = 'none', label = r"$\theta = \pi/4$")
    p6, = plt.plot(betas[0][3:97], Lyapunov[0][3:97], 's', marker = 'o', fillstyle = 'none', label = r"$\theta = \pi/7$")
    plt.legend((p4, p5, p6), [r"$\theta = \pi/3$", r"$\theta = \pi/4$", r"$\theta = \pi/7$"], title = "Simulazioni", loc = "upper right")
    plt.show()
    plt.figure()
    print(chisquare(Lyapunov[2][7:93], Lyapunov_theory[2][7:93]))
    print(chisquare(Lyapunov[1][5:95], Lyapunov_theory[1][5:95]))
    print(chisquare(Lyapunov[0][3:97], Lyapunov_theory[0][3:97]))


