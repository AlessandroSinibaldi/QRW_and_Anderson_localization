import math as m
import random as rnd
import matplotlib.pyplot as plt
import numpy as np

# Numero di walker
n = 10000000

# Numero di passi T
n_steps = 200

# Stato iniziale nell'origine
initial_state = []
for i in range (0, n):
    initial_state.append(0)

# Probabilità p e q
p = 0.5
q = 1 - p

# Movimento del walker
def translation(state):
    for i in range(0, n): 
        y = rnd.uniform(0, 1)
        if(y <= p):
            state[i]+=1
        else:
            state[i]-=1
    return state

# RW
for i in range(0, n_steps):
    initial_state = translation(initial_state)

# Creazione dei dati
maximum = max(initial_state)
minimum = min(initial_state)
bins = maximum - minimum + 1

entries = np.zeros(bins)
for i in initial_state:
    entries[i-minimum] += 1

n = []
for i in range (minimum, maximum + 1):
    n.append(i)

# Normalizzare la distribuzione numerica
total_entries = sum(entries)
entries = entries/total_entries

# Considerare solo le posizioni con n pari (se T è pari)
entries_ = entries[::2]
n_ = n[::2]

# Distribuzione limite
Gaussian = np.array([(2/m.sqrt(2*m.pi*n_steps))*m.exp(-x**2/(2*n_steps)) for x in n_])

# Graficare le distribuzioni
plt.figure()
plt.style.use('default')
ax = plt.gca()
ax.set_facecolor('#F2F3F4')
plt.grid(color='white', linestyle = 'dashed')
title = "RW con $\mathit{p}=\mathit{q}=1/2$ per $T = 200$"
plt.title(title)
plt.xlabel("$\mathit{n}$")
plt.ylabel("Probabilità")
plt.plot(n_, entries_, linewidth = 3, color = "green", label = "Distribuzione numerica" )
plt.plot(n_, Gaussian, color = "black", linestyle = "dashed", label = "Distribuzione limite")
plt.legend(loc='upper left')
plt.show()

# Calcolare e stampare i momenti della distribuzione numerica
print("Mean: ")
print(np.mean(initial_state))
print("Standard deviation: ")
print(np.std(initial_state))