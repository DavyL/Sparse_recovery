import numpy as np
import matplotlib.pyplot as plt
import random


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
    b*=1/normalize
    
    return b


def createFourierList(sparsity, list_length, func):
    indices_fourier = []
    output_list = np.zeros(list_length)
    for i in range(sparsity):
        addIndex(indices_fourier, list_length)
    
    indices_tuple = []
    for frequency in indices_fourier:
        scal = np.random.normal(0,1,1)
        output_list += scal*createFourierVect(list_length, frequency, func)
        indices_tuple.append((frequency+list_length, scal))

    return (output_list, indices_tuple)
    
def createDiracList(sparsity, list_length):
    indices_dirac = []
    output_list = np.zeros(list_length)
    for i in range(sparsity):
        addIndex(indices_dirac, list_length)

    indices_tuple = []
    for i in indices_dirac:
        scal = np.random.normal(0,1,1)
        output_list[i] += scal
        indices_tuple.append((i, scal))
    
    return (output_list, indices_tuple)

##GenSignal() : generates a signal with atoms in dict and support and indices are given by index_tuple
# index_tuple := (atom_index, scalar)
def GenSignal(dimension, dictionary, index_tuple):
    ret_signal = np.zeros(dimension)
    for i in index_tuple:
        ret_signal += i[1] * dictionary[i[0]]

    return ret_signal

def GenSparseSignal(dimension, dirac_spars, fourier_spars):
    d = dimension
    dcos = lambda f,t : (np.sqrt(2)/np.sqrt(2*d))*np.cos(np.pi * f * (2*t+1) / (2*d))          ##Discrete cosine of frequency f, evaluated at time t


    #spars = int(np.sqrt(d) / 2) ##Sparsity constraint for the number of nonzero indices (here evenly split between both bases)
    #spars = int(0.2*d/np.log(d))
    #spars = 1
    (fourier, indices_fourier) = createFourierList(fourier_spars, d, dcos)
    (dirac, indices_dirac) = createDiracList(dirac_spars, d)
    
     #Offset indices of the fourier basis so that it gets assigned in position in the Dirac-Fourier dictionary (D,F)

    
    indices = indices_dirac + indices_fourier #Concatenation of lists (of tuples)
    ret_signal = np.zeros(d)
    ret_signal += dirac
    ret_signal += fourier

    return (ret_signal, indices, indices_dirac, indices_fourier)
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

    

