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

f = open(os.path.join(sys.path[0], "IPR_QRW_N50_dim100_rep10.txt"), 'r')

N = int(next(f))
Ws = np.zeros(N)
thetas = np.zeros(N)
IPR_ = np.zeros((N, N)) 

for j in range(0, N):
    Ws[j] = float(next(f))
    thetas[j] = float(next(f))
for l in range(0, N):
    for k in range(0, N):
        IPR_[k][l] = float(next(f))
f.close()  

Ws, thetas = np.meshgrid(Ws, thetas)
fig = plt.figure()
plt.rcParams['grid.color'] = "black"

ax = fig.gca(projection = '3d')
title = "IPR"

ax.set_xlabel("$W$", rotation = 0)
ax.set_xticks([0, m.pi/2, m.pi, 3*m.pi/2, 2*m.pi]) 
ax.set_xticklabels(['0', '$\pi/2$', '$\pi$', '$3\pi/2$', '$2\pi$'])
ax.xaxis._axinfo["grid"]['linestyle'] = ":"

ax.yaxis.set_rotate_label(False) 
ax.set_ylabel(r'$\theta$', rotation = 0)
ax.yaxis._axinfo["grid"]['linestyle'] = ":"
ax.set_yticks([0, m.pi/4, m.pi/2, 3*m.pi/4, m.pi]) 
ax.set_yticklabels(['0', '$\pi/8$', '$\pi/4$', '$3\pi/8$', '$\pi/2$'])

ax.zaxis.set_rotate_label(False) 
ax.zaxis._axinfo["grid"]['linestyle'] = ":"
ax.set_zlabel(r'$\dfrac{1}{\mathrm{IPR} \cdot N}$', rotation = 0, labelpad = 20)
ax.grid(color = 'white')
surf = ax.plot_surface(Ws, thetas, IPR_, cmap=cm.coolwarm, )
hfont = {'fontname':'monospace'}
plt.title(title)
plt.show()