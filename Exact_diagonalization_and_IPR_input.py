import math as m
import matplotlib.pyplot as plt
import numpy as np
import random as rnd
from numpy.linalg import matrix_power
from numpy import linalg as LA
import math as m
from scipy import optimize
from pylab import *


IPR_vector = []
eigenvalues_vector = []
N = []
IPR_point = []

control = False      #True: IPR vs energia, False: IPR vs N con fit

f = open("IPR_50.txt", 'r')
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

u = open("IPR_60.txt", 'r')
dim = int(next(u))
N.append(dim)
IPR = np.zeros(dim)
eigenvalues = np.zeros(dim)
for j in range(0, dim):
    IPR[j] = float(next(u))
    eigenvalues[j] = float(next(u))
IPR_vector.append(IPR)
eigenvalues_vector.append(eigenvalues)
u.close()

v = open("IPR_65.txt", 'r')
dim = int(next(v))
N.append(dim)
IPR = np.zeros(dim)
eigenvalues = np.zeros(dim)
for j in range(0, dim):
    IPR[j] = float(next(v))
    eigenvalues[j] = float(next(v))
IPR_vector.append(IPR)
eigenvalues_vector.append(eigenvalues)
v.close()

o = open("IPR_75.txt", 'r')
dim = int(next(o))
N.append(dim)
IPR = np.zeros(dim)
eigenvalues = np.zeros(dim)
for j in range(0, dim):
    IPR[j] = float(next(o))
    eigenvalues[j] = float(next(o))
IPR_vector.append(IPR)
eigenvalues_vector.append(eigenvalues)
o.close()

t = open("IPR_85.txt", 'r')
dim = int(next(t))
N.append(dim)
IPR = np.zeros(dim)
eigenvalues = np.zeros(dim)
for j in range(0, dim):
    IPR[j] = float(next(t))
    eigenvalues[j] = float(next(t))
IPR_vector.append(IPR)
eigenvalues_vector.append(eigenvalues)
t.close()

g = open("IPR_100.txt", 'r')
dim = int(next(g))
N.append(dim)
IPR = np.zeros(dim)
eigenvalues = np.zeros(dim)
for j in range(0, dim):
    IPR[j] = float(next(g))
    eigenvalues[j] = float(next(g))
IPR_vector.append(IPR)
eigenvalues_vector.append(eigenvalues)
g.close()

w = open("IPR_125.txt", 'r')
dim = int(next(w))
N.append(dim)
IPR = np.zeros(dim)
eigenvalues = np.zeros(dim)
for j in range(0, dim):
    IPR[j] = float(next(w))
    eigenvalues[j] = float(next(w))
IPR_vector.append(IPR)
eigenvalues_vector.append(eigenvalues)
w.close()

p = open("IPR_150.txt", 'r')
dim = int(next(p))
N.append(dim)
IPR = np.zeros(dim)
eigenvalues = np.zeros(dim)
for j in range(0, dim):
    IPR[j] = float(next(p))
    eigenvalues[j] = float(next(p))
IPR_vector.append(IPR)
eigenvalues_vector.append(eigenvalues)
p.close()

h = open("IPR_200.txt", 'r')
dim = int(next(h))
N.append(dim)
IPR = np.zeros(dim)
eigenvalues = np.zeros(dim)
for j in range(0, dim):
    IPR[j] = float(next(h))
    eigenvalues[j] = float(next(h))
IPR_vector.append(IPR)
eigenvalues_vector.append(eigenvalues)
h.close()

q = open("IPR_250.txt", 'r')
dim = int(next(q))
N.append(dim)
IPR = np.zeros(dim)
eigenvalues = np.zeros(dim)
for j in range(0, dim):
    IPR[j] = float(next(q))
    eigenvalues[j] = float(next(q))
IPR_vector.append(IPR)
eigenvalues_vector.append(eigenvalues)
q.close()

i = open("IPR_300.txt", 'r')
dim = int(next(i))
N.append(dim)
IPR = np.zeros(dim)
eigenvalues = np.zeros(dim)
for j in range(0, dim):
    IPR[j] = float(next(i))
    eigenvalues[j] = float(next(i))
IPR_vector.append(IPR)
eigenvalues_vector.append(eigenvalues)
i.close()

s = open("IPR_350.txt", 'r')
dim = int(next(s))
N.append(dim)
IPR = np.zeros(dim)
eigenvalues = np.zeros(dim)
for j in range(0, dim):
    IPR[j] = float(next(s))
    eigenvalues[j] = float(next(s))
IPR_vector.append(IPR)
eigenvalues_vector.append(eigenvalues)
s.close()

l = open("IPR_400.txt", 'r')
dim = int(next(l))
N.append(dim)
IPR = np.zeros(dim)
eigenvalues = np.zeros(dim)
for j in range(0, dim):
    IPR[j] = float(next(l))
    eigenvalues[j] = float(next(l))
IPR_vector.append(IPR)
eigenvalues_vector.append(eigenvalues)
l.close()

m = open("IPR_500.txt", 'r')
dim = int(next(m))
N.append(dim)
IPR = np.zeros(dim)
eigenvalues = np.zeros(dim)
for j in range(0, dim):
    IPR[j] = float(next(m))
    eigenvalues[j] = float(next(m))
IPR_vector.append(IPR)
eigenvalues_vector.append(eigenvalues)
m.close()

n = open("IPR_600.txt", 'r')
dim = int(next(n))
N.append(dim)
IPR = np.zeros(dim)
eigenvalues = np.zeros(dim)
for j in range(0, dim):
    IPR[j] = float(next(n))
    eigenvalues[j] = float(next(n))
IPR_vector.append(IPR)
eigenvalues_vector.append(eigenvalues)
n.close()


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