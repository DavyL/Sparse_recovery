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
    
    for i in range(2*len(orig_sig)):        ##Change this constant to try longer   
        (modified_signal, sig, new_index) = update(dict, modified_signal, sig)
        recovered_indices.append(new_index)
        if(np.dot(sig-orig_sig, sig-orig_sig) < 0.001):            ##Change this value to increase or decrease reconstruction precision
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

##test_recovery(): generates a random signal with given sparsity and dictionary and tries recovery
#sparsity_tup = (s_1, s_2)
def test_recovery(signal_dimension, sparsity_tup, dictionary):
    sparsity_dirac = sparsity_tup[0]
    sparsity_fourier = sparsity_tup[1]
    (signal, original_indices, orig_dirac_indices, orig_fourier_indices) = sg.GenSparseSignal(signal_dimension, sparsity_dirac, sparsity_fourier)

    (recovered_signal, recovered_indices_tuple) = recover_signal(dictionary, signal)

    cleaned_indices = clean_indices(recovered_indices_tuple)

    return (recovered_signal, cleaned_indices, dictionary, orig_dirac_indices, orig_fourier_indices)

def test_recovery_plot(signal_dimension, sparsity_tup, dictionary):
    #Recover signal
    (recovered_signal, cleaned_indices, dictionary, orig_dirac_indices, orig_fourier_indices) = test_recovery(signal_dimension, sparsity_tup, dictionary)
   
    #Recompute original signal
    orig_dirac_signal = sg.GenSignal(signal_dimension, dictionary, orig_dirac_indices)
    orig_fourier_signal = sg.GenSignal(signal_dimension, dictionary, orig_fourier_indices)
    orig_signal = orig_dirac_signal + orig_fourier_signal

    #Separate the recovered signal based on support
    recovered_indices_tuple_dirac = [x for x in cleaned_indices if x[0] <signal_dimension]
    dirac_signal_recovered = sg.GenSignal(signal_dimension, dictionary, recovered_indices_tuple_dirac) 
    recovered_indices_tuple_fourier = [x for x in cleaned_indices if x[0]>=signal_dimension]
    fourier_signal_recovered = sg.GenSignal(signal_dimension, dictionary, recovered_indices_tuple_fourier)

    #Display
    plt.figure()

    plt.subplot(331)
    plt.plot(orig_signal, 'g-')
    plt.xlabel('t')
    plt.ylabel('f')
    plt.title('Original signal $f = f_D + f_F$')
    plt.subplot(332)
    plt.plot(orig_dirac_signal, 'g-')
    plt.title('$f_D$ (unknown) Dirac part of the signal (s = ' + str(sparsity_tup[0]) + ')')
    plt.subplot(333)
    plt.title('$f_F$ (unknown) Fourier part of the signal (s = ' + str(sparsity_tup[1]) + ')')
    plt.plot(orig_fourier_signal, 'g-')
    plt.subplot(334)
    plt.title('recovered signal $\hat{f} = \hat{f_D} + \hat{f_F}$')
    plt.plot(recovered_signal, 'b-')
    plt.subplot(335)
    plt.title('recovered $\hat{f_D}$')
    plt.plot(dirac_signal_recovered, 'b-')

    plt.subplot(336)
    plt.title('recovered signal $\hat{f_F}$')
    plt.plot(fourier_signal_recovered,'b-')
    
    plt.subplot(337)
    plt.title('original(g) and recovered(b) signal')
    plt.plot(orig_signal, 'g-')
    plt.plot(recovered_signal, 'b-')

    plt.subplot(338)
    plt.title('original(g) and recovered(b) dirac signal')
    plt.plot(orig_dirac_signal, 'g-')
    plt.plot(dirac_signal_recovered, 'b-')

    plt.subplot(339)
    plt.title('original(g) and recovered(b) fourier signal')
    plt.plot(orig_fourier_signal, 'g-')
    plt.plot(fourier_signal_recovered, 'b-')

    plt.show()

    return (recovered_signal, cleaned_indices, dictionary, orig_dirac_indices, orig_fourier_indices)


##batch_test(): tries to compute sparse recovery on batch_size signals
def batch_test(batch_size, signal_dimension, sparsity_tup, dictionary):
    for i in range(batch_size):
        test_recovery_plot(signal_dimension, sparsity_tup, dictionary)


 


