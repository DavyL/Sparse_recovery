import numpy as np
import matplotlib.pyplot as plt
import random
import math


def addIndex(cur_indices, index_range):
    for count in range(index_range):
        new_index = random.randint(0, index_range-1)
        if new_index not in cur_indices:
            cur_indices.append(new_index) 
            return
    print("In addIndex :" + str(count) + "indices have been tried without success.")
    return

def createFourierVect(arr_size, freq, func):
    b = np.zeros(arr_size)
    cos_at_freq = lambda t: func(freq,t)
    for t_step in range(arr_size):
        b[t_step] = cos_at_freq(t_step)
    normalize = np.sqrt(np.dot(b, b))
    for t_step in range(arr_size):
        b[t_step] *= 1/normalize
    return b

def addListsInFirst(l1, l2):    ##Adds (pointwise) elems from l2 to l1
    for i in range(len(l1)):
        l1[i] += l2[i]

def createFourierList(sparsity, list_length, func):
    indices_fourier = []
    output_list = np.zeros(list_length)
    for i in range(sparsity):
        addIndex(indices_fourier, list_length)
    
    for frequency in indices_fourier:
        f = createFourierVect(list_length, frequency, func)
        addListsInFirst(output_list, f)
    
    return (output_list, indices_fourier)
    
def createDiracList(sparsity, list_length):
    indices_dirac = []
    output_list = np.zeros(list_length)
    for i in range(sparsity):
        addIndex(indices_dirac, list_length)

    for i in indices_dirac:
        output_list[i] += 1

    return (output_list, indices_dirac)


def GenSparseSignal(dimension, dirac_spars, fourier_spars):
    d = dimension
    dcos = lambda f,t : (np.sqrt(2)/np.sqrt(2*d))*np.cos(np.pi * f * (2*t+1) / (2*d))          ##Discrete cosine of frequency f, evaluated at time t


    #spars = int(np.sqrt(d) / 2) ##Sparsity constraint for the number of nonzero indices (here evenly split between both bases)
    #spars = int(0.2*d/np.log(d))
    #spars = 1
    (fourier, indices_fourier) = createFourierList(fourier_spars, d, dcos)
    (dirac, indices_dirac) = createDiracList(dirac_spars, d)
    for i in range(len(indices_fourier)):
        indices_dirac.append(d + indices_fourier[i])
    a = np.zeros(d)
    addListsInFirst(a, dirac)
    addListsInFirst(a, fourier)

    return (a, indices_dirac)
    #PlotSignals(a, dirac, fourier)


def PlotSignals(s1, s2, s3):
    plt.plot(s1)
    plt.ylabel('Dirac + Fourier')
    plt.show()

    plt.plot(s2)
    plt.ylabel('Dirac')
    plt.show()

    plt.plot(s3)
    plt.ylabel('Fourier')
    plt.show()

    

