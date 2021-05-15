import math as m
import matplotlib.pyplot as plt
import numpy as np
import random as rnd
from numpy.linalg import matrix_power
from numpy import linalg as LA
import math as m

f = open("localization_lenght_W1.txt", 'r')
dim = 1000  # Se viene cambiata dim in Localization_lenght_output.py, va cambiata anche qua

localization_lenght = np.zeros((5, dim))
eigenvalues_mean = np.zeros((5, dim))

for j in range(0, dim):
    localization_lenght[0][j] = float(next(f))
    eigenvalues_mean[0][j] = float(next(f))

g = open("localization_lenght_W1_5.txt", 'r')

for j in range(0, dim):
    localization_lenght[1][j] = float(next(g))
    eigenvalues_mean[1][j] = float(next(g))

h = open("localization_lenght_W2.txt", 'r')

for j in range(0, dim):
    localization_lenght[2][j] = float(next(h))
    eigenvalues_mean[2][j] = float(next(h))

i = open("localization_lenght_W2_5.txt", 'r')

for j in range(0, dim):
    localization_lenght[3][j] = float(next(i))
    eigenvalues_mean[3][j] = float(next(i))

k = open("localization_lenght_W3.txt", 'r')

for j in range(0, dim):
    localization_lenght[4][j] = float(next(k))
    eigenvalues_mean[4][j] = float(next(k))

plt.figure()
plt.style.use('default')
ax = plt.gca()
ax.set_facecolor('#F2F3F4')
title = "Lunghezza di localizzazione"
plt.xlabel("Energia")
plt.ylabel(r'$\xi$')
plt.grid(color='white', linestyle = 'dashed')
hfont = {'fontname':'monospace'}
plt.title(title)
plt.plot(eigenvalues_mean[0], localization_lenght[0], label = "$\mathit{W}$ = 1")
plt.plot(eigenvalues_mean[1], localization_lenght[1], label = "$\mathit{W}$ = 1.5")
plt.plot(eigenvalues_mean[2], localization_lenght[2], label = "$\mathit{W}$ = 2")
plt.plot(eigenvalues_mean[3], localization_lenght[3], label = "$\mathit{W}$ = 2.5")
plt.plot(eigenvalues_mean[4], localization_lenght[4], label = "$\mathit{W}$ = 3")
plt.legend(loc = 'upper right')
plt.show()

