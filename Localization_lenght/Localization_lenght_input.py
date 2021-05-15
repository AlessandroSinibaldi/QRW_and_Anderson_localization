import math as m
import matplotlib.pyplot as plt
import numpy as np
import random as rnd
from numpy.linalg import matrix_power
from numpy import linalg as LA
import os
import sys

#f = open("localization_lenght_W1.txt", 'r')
dim = 1000  # Se viene cambiata dim in Localization_lenght_output.py, va cambiata anche qua

filenames = ["localization_lenght_W1.txt", "localization_lenght_W1_5.txt", "localization_lenght_W2.txt" ,
"localization_lenght_W2_5.txt", "localization_lenght_W3.txt"]

localization_lenght = np.zeros((len(filenames), dim))
eigenvalues_mean = np.zeros((len(filenames), dim))

for k in range(0, 5):
    f = open (os.path.join(sys.path[0], filenames[k]), 'r')
    for j in range(0, dim): 
        localization_lenght[k][j] = float(next(f))
        eigenvalues_mean[k][j] = float(next(f))
    f.close()

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

