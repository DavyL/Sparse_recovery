import numpy as np
import matplotlib.pyplot as plt


import SigRec as sr
import SigGen as sg



def elementary_test_1():
    signal_dimension = 200

    sparsity =  int(np.floor(0.5*signal_dimension/np.log(signal_dimension)))
    print("sparsity : " + str(sparsity))
    sparsity_dirac = 2*sparsity
    sparsity_fourier = 2*sparsity 

    dcos = lambda f,t : (np.sqrt(2)/np.sqrt(signal_dimension))* np.cos( (np.pi * f * (2*t + 1)) / (2*signal_dimension))          ##Discrete cosine of frequency f, evaluated at time t
    dirac_basis = sr.GetDiracBasis(signal_dimension)
    fourier_basis = sr.GetFourierBasis(signal_dimension, dcos)

    dictionary = dirac_basis + fourier_basis

    (recovered_signal, cleaned_indices, dictionary, orig_dirac_indices, orig_fourier_indices) = sr.test_recovery_plot(signal_dimension, (sparsity_dirac, sparsity_fourier), dictionary)

def elementary_test_2():
    signal_dimension = 400

    binary=0##Set to 1 for 0-1 valued indices, other indices values are distributed according to standard gaussian

    sparsity =  int(np.floor(0.5*signal_dimension/np.log(signal_dimension)))
    print("sparsity : " + str(sparsity))
    sparsity_dirac = sparsity
    sparsity_fourier = sparsity 

    dcos = lambda f,t : (np.sqrt(2)/np.sqrt(signal_dimension))* np.cos( (np.pi * f * (2*t + 1)) / (2*signal_dimension))          ##Discrete cosine of frequency f, evaluated at time t
    dirac_basis = sr.GetDiracBasis(signal_dimension)
    fourier_basis = sr.GetFourierBasis(signal_dimension, dcos)

    dictionary = dirac_basis + fourier_basis

    (observed_signal, indices, indices_dirac, indices_fourier) = sg.GenSparseSignal(signal_dimension, sparsity_dirac, sparsity_fourier)

    if(binary==1):
        indices = [(i[0], 1.0) for i in indices]
        
        observed_signal = sg.GenSignal(signal_dimension, dictionary, indices)

    (recovered_signal, recovered_indices) = sr.recover_signal(dictionary, observed_signal)
    clean_recovered_indices = sr.clean_indices(recovered_indices)
    plt.figure()
    plt.subplot(121)
    plt.title("observed(g) and recovered(r) signal")
    plt.plot(observed_signal, 'g-')
    plt.plot(recovered_signal, 'r-')
    plt.subplot(122)
    plt.title("coefficients original (g) and recovered(r) (dirac and fourier)")
    plt.plot([x[0] for x in clean_recovered_indices], [x[1] for x in clean_recovered_indices], 'r+')
    plt.plot([x[0] for x in indices], [x[1] for x in indices], 'g+')
    plt.show()

elementary_test_1()