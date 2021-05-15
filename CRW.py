import math as m
import random as rnd
import matplotlib.pyplot as plt

# Numero di walker
n = 800000

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

# CRW
for i in range(0, n_steps):
    initial_state = translation(initial_state)

# Creare dei dati
maximum = max(initial_state)
minimum = min(initial_state)
bins = maximum - minimum + 1
(entries, bins, patches) = plt.hist(initial_state, bins)
n = []
for i in range (minimum, maximum + 1):
    n.append(i)

# Considerare solo le posizioni con n pari
entries_ = entries[::2]
n_ = n[::2]

# Normalizzare la distribuzione
total_entries = sum(entries_)
entries_ = entries_/total_entries

# Graficare la distribuzione
plt.figure()
plt.style.use('default')
ax = plt.gca()
ax.set_facecolor('#F2F3F4')
plt.grid(color='white', linestyle = 'dashed')
title = "CRW con $\mathit{p}=\mathit{q}=1/2$ per $T = 200$"
plt.title(title)
plt.xlabel("$\mathit{n}$")
plt.ylabel("Probabilità")
if(maximum > abs(minimum)):
    limit = minimum
else:
    limit = maximum
plt.xlim([-limit, limit])
plt.plot(n_, entries_, linewidth = 3, color = "green")
plt.show()