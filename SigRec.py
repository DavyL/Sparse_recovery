import numpy as np
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

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
    #for i in range(len(signal_to_recover)):
    #    recovered_signal[i] += dot_prod[max_index]*dict[max_index][i]
    #    signal_to_recover[i] -= dot_prod[max_index]*dict[max_index][i]
    recovered_signal += (dot_prod[max_index]*dict[max_index])
    signal_to_recover -= (dot_prod[max_index]*dict[max_index])
    
    return (signal_to_recover, recovered_signal, (max_index, dot_prod[max_index]))

def normalize_dictionary(dic):
    for vect in dic:
        norm = np.sqrt(np.dot(vect, vect))
        vect *= 1/norm


##recover_signal(): Performs MatchingPursuit on orig_sig using the dictionary dict
#Returns the recovered signal associated with a list of tuple containing the index of the atom in the dictionary and the associated scalar
#Note that each index can appear many times (call clean_indices() to solve this)
def recover_signal(dict, orig_sig):
    recovered_indices = []
    modified_signal = orig_sig.copy()
    sig = np.zeros(len(orig_sig))
    for i in range(2*len(orig_sig)):
        (modified_signal, sig, new_index) = update(dict, modified_signal, sig)
        recovered_indices.append(new_index)
        if(np.dot(sig-orig_sig, sig-orig_sig) < 0.01):
            print("Number of iterations to recover : " + str(i +1))
            return (sig, recovered_indices)
    print("Recovery didn't succeed after " + str(i+1) + "iterations")
    return (sig, recovered_indices)

##clean_indices(): returns from a list of tuple (index, scal) a list of tuple where each index appears only once 
# and the associated scalar is the sum of scalars with associated index
def clean_indices(indices_tuple):
    ret_list_indices = []
    ret_list_scalar = []
    for i in indices_tuple:
        if i[0] not in ret_list_indices:
            ret_list_indices.append(i[0])
            ret_list_scalar.append(i[1])
        else:
            ind = ret_list_indices.index(i[0])
            ret_list_scalar[ind] += i[1]

    ret_tuple = [(ret_list_indices[i], ret_list_scalar[i]) for i in range(len(ret_list_indices))]
    return ret_tuple


signal_dimension = 200

sparsity =  int(np.floor(0.5*signal_dimension/np.log(signal_dimension)))
print("sparsity : " + str(sparsity))
sparsity_dirac = sparsity
sparsity_fourier = sparsity 

dcos = lambda f,t : (np.sqrt(2)/np.sqrt(signal_dimension))* np.cos( (np.pi * f * (2*t + 1)) / (2*signal_dimension))          ##Discrete cosine of frequency f, evaluated at time t

(signal, original_indices, dirac_indices, fourier_indices) = sg.GenSparseSignal(signal_dimension, sparsity_dirac, sparsity_fourier)
orig_signal = signal.copy()

dirac_basis = GetDiracBasis(signal_dimension)
fourier_basis = GetFourierBasis(signal_dimension, dcos)

dictionary = dirac_basis + fourier_basis


(recovered_signal, recovered_indices_tuple) = recover_signal(dictionary, signal)

cleaned_indices = clean_indices(recovered_indices_tuple)

orig_dirac_signal = sg.GenSignal(signal_dimension, dictionary, dirac_indices)
orig_fourier_signal = sg.GenSignal(signal_dimension, dictionary, fourier_indices)

recovered_indices_tuple_dirac = [x for x in cleaned_indices if x[0] <signal_dimension]
dirac_signal_recovered = sg.GenSignal(signal_dimension, dictionary, recovered_indices_tuple_dirac) 
recovered_indices_tuple_fourier = [x for x in cleaned_indices if x[0]>=signal_dimension]
fourier_signal_recovered = sg.GenSignal(signal_dimension, dictionary, recovered_indices_tuple_fourier)


fig = plt.figure()
plt.subplot(231)
plt.plot(orig_signal, 'g-')
plt.xlabel('t')
plt.ylabel('f')
plt.title('Original signal $f = f_D + f_F$')
plt.subplot(232)
plt.plot(orig_dirac_signal, 'r-')
plt.title('$f_D$ (unknown) Dirac part of the signal (s = ' + str(sparsity_dirac) + ')')
plt.subplot(233)
plt.title('$f_F$ (unknown) Fourier part of the signal (s = ' + str(sparsity_fourier) + ')')
plt.plot(orig_fourier_signal, 'c-')
plt.subplot(234)
plt.title('recovered signal $\hat{f} = \hat{f_D} + \hat{f_F}$')
plt.plot(recovered_signal, 'b-')
plt.subplot(235)
plt.title('recovered $\hat{f_D}$')
plt.plot(dirac_signal_recovered, 'r-')
plt.subplot(236)
plt.title('recovered signal $\hat{f_F}$')
plt.plot(fourier_signal_recovered,'c-')
plt.show()

