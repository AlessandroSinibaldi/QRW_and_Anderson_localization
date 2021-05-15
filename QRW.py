import math as m
import matplotlib.pyplot as plt
from scipy.stats import chisquare

# Stato:
# [0]: coefficiente, [1]: posizione, 
# [2]: spin (1: up, -1: down, 2: i-up, -2: i-down)

# Stato iniziale asimmetrico nell'origine
#initial_state = [[1, 0, 1]]    

# Stato iniziale asimmetrico nell'origine
initial_state = [[1/m.sqrt(2), 0, 1], [1/m.sqrt(2), 0, -2]]

# Numero di passi T
T = 200

# Coin quantistico di Hadamard H
def Hadamard_coin(state):
    state_ = state[:]
    i = 0
    for x in state:
        if (x[2] == 1):
            x[0] = x[0]/m.sqrt(2)
            state_[i] = [x[0], x[1], -1]
        if (x[2] == -1):
            x[0] = -x[0]/m.sqrt(2)
            state_[i] = [-x[0], x[1], 1]
        if (x[2] == 2):
            x[0] = x[0]/m.sqrt(2)
            state_[i] = [x[0], x[1], -2]
        if (x[2] == -2):
            x[0] = -x[0]/m.sqrt(2)
            state_[i] = [-x[0], x[1], 2]
        i+=1
    return state + state_

# Coin quantistico Y (simmetrico)
def Y_coin(state):
    state_ = state[:]
    i = 0
    for x in state:
        if (x[2] == 1):
            x[0] = x[0]/m.sqrt(2)
            state_[i] = [x[0], x[1], -2]
        if (x[2] == -1):
            x[0] = x[0]/m.sqrt(2)
            state_[i] = [x[0], x[1], 2]
        if (x[2] == 2):
            x[0] = x[0]/m.sqrt(2)
            state_[i] = [-x[0], x[1], -1]
        if (x[2] == -2):
            x[0] = x[0]/m.sqrt(2)
            state_[i] = [-x[0], x[1], 1]
        i+=1
    return state + state_

# Operatore S di traslazione condizionata
def translation_operator(state):
    for x in state:
        if (x[2] == 1 or x[2] == 2):
            x[1] = x[1]+1
        if (x[2] == -1 or x[2] == -2):
            x[1] = x[1]-1
    return state

# Interferenza tra gli stati
def interference(state):
    for x in state:
        for y in state:
            if ((x is not y) and (x[1] == y[1]) and (x[2] == y[2])):
                x[0] += y[0]
                state.remove(y)
    for x in state:       
        if(abs(x[0]) < 0.0000000001):
            state.remove(x)  
    return state

# Misurazione (collasso dello stato)
def probabilities_and_positions(state, probabilities, positions):
    for i in range(-T, T+1):
        ampl_tot = 0
        for x in state:
            if(x[1] == i):
                ampl_tot += x[0]**2
        probabilities.append(ampl_tot)
        positions.append(i)

# Distribuzione p_slow
def p_slow(restricted_alpha, p_slow_probabilities):
    for a in restricted_alpha:
        p_slow_probabilities.append(2/(m.pi*T*m.sqrt(1 - 2*(a**2))*(1 - a)))

# Momenti principali della distribuzione
def moments(positions, probabilities):
    alpha = [n/T for n in positions]
    mean = 0
    mean_abs = 0
    mean_squared = 0
    for i in range(0, len(alpha)):
        mean += probabilities[i]*alpha[i]
        mean_abs += probabilities[i]*abs(alpha[i])
        mean_squared += probabilities[i]*alpha[i]**2
    std_deviation = m.sqrt(mean_squared - mean**2)
    return [mean, mean_abs, mean_squared, std_deviation]

# Approssimazione di p con il metodo della fase stazionaria
def stationary_phase_approx(restricted_alpha, restricted_probabilities_approx):
    t = T
    for a in restricted_alpha:
        k_alpha = m.acos(a/m.sqrt(1-a**2))
        omega_k_alpha = m.asin(m.sin(k_alpha)/m.sqrt(2))
        p_up_approx = ((2+2*a)/(m.pi*t*(1-a)*m.sqrt(1-2*a**2)))*(m.cos(-omega_k_alpha*t+a*t*k_alpha+m.pi/4))**2
        p_down_approx = (2/(m.pi*t*m.sqrt(1-2*a**2)))*(m.cos(-omega_k_alpha*t+(a*t+1)*k_alpha+m.pi/4))**2
        restricted_probabilities_approx.append(p_up_approx + p_down_approx)

# QRW
for i in range (0, T):
    initial_state = translation_operator(Hadamard_coin(initial_state))
    initial_state = interference(initial_state)
state_final = initial_state

# Probabilità e posizioni
probabilities = []
positions = []
p_slow_probabilities = []
restricted_positions = []
restricted_probabilities = []
restricted_probabilities_approx = []

probabilities_and_positions(state_final, probabilities, positions)
probabilities_even = probabilities[::2]
positions_even = positions[::2]

for i in range(0, len(positions_even)):
    if(abs(positions_even[i]) < T/m.sqrt(2)):
        restricted_positions.append(positions_even[i])
        restricted_probabilities.append(probabilities_even[i])

restricted_positions_approx = [n for n in restricted_positions]
del restricted_positions_approx[len(restricted_positions_approx)-1]
del restricted_positions_approx[len(restricted_positions_approx)-1]
del restricted_positions_approx[0]
del restricted_positions_approx[0]

restricted_probabilities_ = [n for n in restricted_probabilities]
del restricted_probabilities_[len(restricted_probabilities_)-1]
del restricted_probabilities_[len(restricted_probabilities_)-1]
del restricted_probabilities_[0]
del restricted_probabilities_[0]

alpha = [n/T for n in positions]
alpha_even = [n/T for n in positions_even]
restricted_alpha = [n/T for n in restricted_positions]

# Calcolare la distribuzione p_slow e p approssimata
p_slow(restricted_alpha, p_slow_probabilities)
stationary_phase_approx(restricted_alpha, restricted_probabilities_approx)
del restricted_probabilities_approx[len(restricted_probabilities_approx)-1]
del restricted_probabilities_approx[len(restricted_probabilities_approx)-1]
del restricted_probabilities_approx[0]
del restricted_probabilities_approx[0]

# Graficare le distribuzioni
plt.figure()
plt.style.use('default')
#plt.plot(positions_even, probabilities_even, color = "blue", label = '$\mathit{p}$')
plt.plot(positions_even, probabilities_even, color = "red", label = '$\mathit{p}$')
#plt.plot(restricted_positions_approx, restricted_probabilities_approx, color = "black", linestyle = "dashed", label = '$\mathit{p}$ approssimata')
ax = plt.gca()
ax.set_facecolor('#F2F3F4')
#title = "QRW con $\mathit{coin}$ di Hadamard per $t = 200$"
#title = "QRW simmetrico con $\mathit{coin}$ di Hadamard per $t = 200$"
#title = "Confronto tra $\mathit{p}$ e $\mathit{p'_{slow}}$ per $t = 200$"
title = ("Confronto tra $\mathit{p}$ e $\mathit{p}$ approssimata "
"per $t = 200$")
plt.grid(color='white', linestyle = 'dashed')
hfont = {'fontname':'monospace'}
plt.title(title)
plt.xlabel("$\mathit{n}$")
plt.ylabel("Probabilità")
#plt.plot(restricted_positions, p_slow_probabilities, color = "black", linestyle = 'dashed', label = "$\mathit{p'_{slow}}$")
plt.legend(loc='upper left')
plt.show()

# Calcolare e stampare i momenti
statistic = moments(positions_even, probabilities_even)
print('Mean: ', statistic[0])
print('Absolute value mean: ', statistic[1])
print('Mean squared: ', statistic[2])
print('Standard deviation: ', statistic[3])

# Calcolare e stampare il chi quadrato tra la p esatta e quella approssimata
print(chisquare(restricted_probabilities_approx, restricted_probabilities_))