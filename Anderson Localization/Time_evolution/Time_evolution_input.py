import math as m
import matplotlib.pyplot as plt
import numpy as np
import random as rnd
from numpy import linalg as LA
from matplotlib.animation import FuncAnimation
import os 
import sys

f = open(os.path.join(sys.path[0], "time_evolution_animation.txt"), 'r')
dim = int(next(f))
len_t = int(next(f))
len_W = int(next(f))
max = float(next(f))

probability_tot = np.zeros((len_W, len_t, dim))
positions = np.array([x for x in range(0, dim)])
y = np.zeros((len_W, dim))

for k in range(0, len_W):
    for i in range(0, len_t):
        for j in range(0 ,dim):
            probability_tot[k][i][j] = float(next(f))

def animate(i):
        plt.cla()
        x = positions
        for j in range(0, len_t):
            if(i == j):
                for k in range(0, len_W):
                    y[k] = probability_tot[k][j]
        plt.ylim(0, max)
        for k in range(0, len_W):   # max 5 disorders 
            if(k == 0):
                axes.plot(x, y[k], color = "green")
            if(k == 1):
                axes.plot(x, y[k], color = "blue")
            if(k == 2):
                axes.plot(x, y[k], color = "red")
            if(k == 3):
                axes.plot(x, y[k], color = "black")
            if(k == 4):
                axes.plot(x, y[k], color = "purple")
       
fig = plt.figure()
axes = fig.add_subplot(111)
plt.style.use('default')
ax = plt.gca()
ax.set_facecolor('#F2F3F4')
graphs = axes.plot([], [])
title = "Localizzazione degli autostati"
plt.xlabel("$\mathit{l}$")
plt.ylabel("Probabilità")
plt.grid(color='white', linestyle = 'dashed')
hfont = {'fontname':'monospace'}
plt.title(title)
animation = FuncAnimation(fig=fig, func=animate, frames=range(0, len_t), interval=250, repeat=False)
plt.show()
