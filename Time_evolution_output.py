import math as m
import matplotlib.pyplot as plt
import numpy as np
import random as rnd
from numpy import linalg as LA
from matplotlib.animation import FuncAnimation

dim = 200
g = -1
W = np.array([0, 1, 2])
seed = 1234
rnd.seed(seed)
t = np.linspace(0, 75, 10)
delta = False
gauss = not delta
output = False

def create_Hamiltonian(Hamiltonian, W):
    for l in range(0, dim):
        Hamiltonian[l][l] = rnd.uniform(-W/2, W/2)
        if(l-1 < 0):
            Hamiltonian[l][l+1] = -g
            Hamiltonian[l][dim-1] = -g
        if(l+1 > dim-1):
            Hamiltonian[l][l-1] = -g
            Hamiltonian[l][0] = -g
        else:
            Hamiltonian[l][l+1] = -g
            Hamiltonian[l][l-1] = -g   
    return Hamiltonian

def resolve_equation(Hamiltonian):
    eigenvalues, eigenvectors = LA.eig(Hamiltonian)
    eigenvalues_ = np.sort(eigenvalues)
    indexes = eigenvalues.argsort()
    counter = 0 
    for idx in indexes:
        eigenvectors_[:, counter] = eigenvectors[:, idx]
        counter+=1
    return eigenvalues_, eigenvectors_

def normalize(state):
    norm = 0
    for i in range(0, dim):
        norm += state[i]**2
    state = np.array([x/m.sqrt(norm) for x in state])
    return state

def compute_norm_squared(state):
    norm = 0
    for i in range(0, dim):
        norm += state[i]**2
    print(m.sqrt(norm))

def compute_norm(state):
    norm = 0
    for i in range(0, dim):
        norm += state[i]
    print(norm)

def create_initial_state(eigenvectors, coefficients):
    eigenvectors_ = np.multiply(eigenvectors, coefficients)
    initial_state = np.zeros(dim)
    for i in range(0, dim):
        for j in range(0, dim):
            initial_state[i] += eigenvectors_[i][j] 
    return eigenvectors_, initial_state

def time_evolution(eigenvectors_, eigenvalues, t):
    state_Re = np.zeros(dim)
    state_Im = np.zeros(dim)
    for i in range(0, dim):
        for j in range(0, dim):
            state_Re[i] += eigenvectors_[i][j]*m.cos(eigenvalues[j]*t)
            state_Im[i] += eigenvectors_[i][j]*m.sin(eigenvalues[j]*t)
    state_Re_ = np.array([x**2 for x in state_Re])
    state_Im_ = np.array([x**2 for x in state_Im])
    probabilities = np.add(state_Re_, state_Im_)
    return probabilities

def gaussian_wf(x, mu, sigma):
    return (m.exp(-(x-mu)**2/(4*sigma**2)))/((2*m.pi*sigma**2)**(1/4))

def create_coefficients(eigenvectors):
    if(delta == True):
        coefficients = eigenvectors[int(dim/2)]

    if(gauss == True):
        coefficients = np.zeros(dim)
        sigma = 5
        for i in range(0, dim):
            for j in range(0, dim):
                coefficients[i] += eigenvectors[j][i]*gaussian_wf(j, int(dim/2), sigma)
    return coefficients

probability_tot = []

for i in range(0, len(W)):
    Hamiltonian = np.zeros((dim, dim))    
    eigenvectors_ = np.zeros((dim, dim))
    positions = np.array([x for x in range(0, dim)])
    create_Hamiltonian(Hamiltonian, W[i])
    eigenvalues, eigenvectors = resolve_equation(Hamiltonian)
    coefficients = create_coefficients(eigenvectors)
    eigenvectors_, initial_state = create_initial_state(eigenvectors, coefficients)
    
    # Evoluzione temporale
    prob = []
    for i in range(0, len(t)):
        prob.append(time_evolution(eigenvectors_, eigenvalues, t[i]))
    probability_tot.append(prob)
    
max = np.max(probability_tot)

# Scrivere su file
f = open("time_evolution_animation.txt", 'w')
f.write(str(dim))
f.write("\n")
f.write(str(len(t)))
f.write("\n")
f.write(str(len(W)))
f.write("\n")
f.write(str(max))
f.write("\n")
for k in range(0, len(W)):
    for i in range(0, len(t)):
        for j in range(0, dim):
            f.write ("{:.16f}".format(float(probability_tot[k][i][j])))
            f.write("\n")

if(output == True):
    def animate(i):
        plt.cla()
        x = positions
        for j in range(0, len(t)):
            if(i == j):
                y_1 = probability_tot[0][j]
                y_2 = probability_tot[1][j]
                y_3 = probability_tot[2][j]
        plt.ylim(0, max)
        axes.plot(x, y_1, color = 'green', label = '$\mathit{W} = 0$')
        axes.plot(x, y_2, color = 'blue', label = '$\mathit{W} = 1$')
        axes.plot(x, y_3, color = 'red', label = '$\mathit{W} = 2$')

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
    animation = FuncAnimation(fig=fig, func=animate, frames=range(0, len(t)), interval=150, repeat=True)
    plt.show()
