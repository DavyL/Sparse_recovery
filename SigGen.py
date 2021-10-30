import numpy as np
import matplotlib.pyplot as plt
import random
import math

d = 1000     ##Dimension of the signal

dcos = lambda f,t : (2/np.sqrt(2*d))*math.cos(2*math.pi * f * t / d)          ##Discrete cosine of frequency f, evaluated at time t

def addIndex(cur_indices, index_range):
    count = 0
    while(count < index_range):
        count+=1
        new_index = random.randint(0, index_range-1)
        if new_index not in cur_indices:
            cur_indices.append(new_index) 
            return
    return print("In addIndex :" + str(count) + "indices have been trying without success.")

def createFourierVect(arr_size, freq):
    b = []
    cos_at_freq = lambda t: dcos(freq,t)
    for t_step in range(arr_size):
        b.append(cos_at_freq(t_step))
    return b

def addListsInFirst(l1, l2):    ##Adds (pointwise) elems from l2 to l1
    for i in range(len(l1)):
        l1[i] += l2[i]

def createFourierList(sparsity, list_length):
    indices_fourier = []
    output_list = np.zeros(list_length)
    for i in range(sparsity):
        addIndex(indices_fourier, list_length)
    
    for frequency in indices_fourier:
        f = createFourierVect(list_length, frequency)
        addListsInFirst(output_list, f)
    
    return output_list
    
def createDiracList(sparsity, list_length):
    indices_dirac = []
    output_list = np.zeros(list_length)
    for i in range(sparsity):
        addIndex(indices_dirac, list_length)

    for i in indices_dirac:
        output_list[i] += 1

    return output_list

def l2_norm(l):
    s = 0
    for elem in l:
        s+= elem*elem
    print(np.sqrt(s))


spars = int(np.sqrt(d) / 2) ##Sparsity constraint for the number of nonzero indices (here evenly split between both bases)
#spars = int(0.2*d/np.log(d))
#spars = 3
fourier = createFourierList(spars, d)
dirac = createDiracList(spars, d)
a = np.zeros(d)
addListsInFirst(a, dirac)
addListsInFirst(a, fourier)

plt.plot(a)
plt.ylabel('Dirac + Fourier')
plt.show()

plt.plot(dirac)
plt.ylabel('Dirac')
plt.show()

plt.plot(fourier)
plt.ylabel('Fourier')
plt.show()

