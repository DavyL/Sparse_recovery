import numpy as np
import matplotlib.pyplot as plt

import math
from numpy import rec

from numpy.lib.index_tricks import index_exp

import SigGen as sg

def GetDiracBasis(dim):
    B = []
    for i in range(dim):
        temp = np.zeros(dim)
        temp[i] = 1
        B.append(temp)
    return B

def GetFourierBasis(dim, func):
    B = []
    for freq in range(dim):
        temp = sg.createFourierVect(dim, freq, func)
        B.append(temp)
    return B

def GetDotProducts(input_signal, dict):
    l=[]
    for f in dict:
        l.append(np.dot(input_signal, f))
    return l

def GetMaxIndex(l):
    max = 0
    max_index = 0
    for i in range(len(l)):
        if (abs(l[i]) > max):
            max = abs(l[i])
            max_index = i
    return max_index

def update(dict, signal_to_recover, recovered_signal):
    dot_prod = GetDotProducts(signal_to_recover, dict)

    max_index = GetMaxIndex(dot_prod)
    #print("max index is " + str(max_index) + "which evaluates at " + str(dot_prod[max_index]))    
    for i in range(len(signal_to_recover)):
        recovered_signal[i] += dot_prod[max_index]*dict[max_index][i]
        signal_to_recover[i] -= dot_prod[max_index]*dict[max_index][i]

    return (signal_to_recover, recovered_signal, max_index)

def normalize_dictionary(dic):
    for vect in dic:
        norm = np.sqrt(np.dot(vect, vect))
        for i in range(len(vect)):
            vect[i] = vect[i]/norm

def recover_signal(dict, orig_sig):
    recovered_indices = []
    modified_signal = orig_sig.copy()
    sig = np.zeros(len(orig_sig))
    for i in range(2*len(orig_sig)):
        (modified_signal, sig, new_index) = update(dict, modified_signal, sig)
        recovered_indices.append(new_index)
        if(np.dot(sig-orig_sig, sig-orig_sig) < 0.1):
            print("Number of iterations to recover : " + str(i +1))
            return (sig, recovered_indices)
    print("Recovery didn't succeed after " + str(i+1) + "iterations")
    return (sig, recovered_indices)



signal_dimension = 1000
#sparsity_dirac = 50
#sparsity_fourier = 50

sparsity =  int(np.floor(0.5*signal_dimension/np.log(signal_dimension)))
print("sparsity : " + str(sparsity))
sparsity_dirac = sparsity
sparsity_fourier = sparsity 

dcos = lambda f,t : (np.sqrt(2)/np.sqrt(signal_dimension))* np.cos( (np.pi * f * (2*t + 1)) / (2*signal_dimension))          ##Discrete cosine of frequency f, evaluated at time t

(signal, original_indices) = sg.GenSparseSignal(signal_dimension,sparsity_dirac, sparsity_fourier)
orig_signal = signal.copy()

dirac_basis = GetDiracBasis(signal_dimension)
fourier_basis = GetFourierBasis(signal_dimension, dcos)
#normalize_dictionary(fourier_basis)

dictionary = dirac_basis.copy()
for vect in fourier_basis:
    dictionary.append(vect)


(recovered_signal, recovered_indices) = recover_signal(dictionary, signal)

original_indices.sort()
recovered_indices.sort()
original_indices = list(dict.fromkeys(original_indices))
recovered_indices = list(dict.fromkeys(recovered_indices))
plt.plot(orig_signal, 'g-')
plt.plot(recovered_signal, 'b-')
plt.show()

#print(original_indices)
#print(recovered_indices)

excess_indices = []
for i in recovered_indices:
    if i not in original_indices:
        excess_indices.append(i)
print("indices in excess : " + str(excess_indices))

missing_indices = []
for i in original_indices:
    if i not in recovered_indices:
        missing_indices.append(i)
print("indices missing : " + str(missing_indices))


