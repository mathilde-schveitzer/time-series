# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 16:42:02 2021

@author: mathi
"""

import csv
import numpy as np
import random as rd
import matplotlib.pyplot as plt

def generate_signal(length_seconds, sampling_rate, frequencies_list, func=[], trend=[0,1], alea=False, bornes=[-10,10], add_noise=0, plot=True):
    """
    Generate a `length_seconds` seconds signal at `sampling_rate` sampling rate. See torchsignal (https://github.com/jinglescode/torchsignal) for more info.
    
    Args:
        length_seconds : int
            Duration of signal in seconds (i.e. `10` for a 10-seconds signal)
        sampling_rate : int
            The sampling rate of the signal.
        frequencies_list : 1 dimension python list a floats
            An array of floats, where each float is the desired frequencies to generate (i.e. [5, 12, 15] to generate a signal containing a 5-Hz, 12-Hz and 15-Hz)
        alea : boolean, default : false
            When alea==true, an and bn are randomly picked
        func : list, default: []
            Contains the periodic functions to generate signal, either `sin` or `cos`
            Must match frequencies_list size. If not, will be automatically completed by "sin"
        bornes: list, 
            Specifies the max and min values for the rand function that will then pick the value of the "an"/"bn" coefficient of the Fourier series
        trend : list, default : [0,1]
            Coefficient of a polynomial functions that will give a generall trend to the signal
        add_noise : float, default: 0
            Add random noise to the signal, where `0` has no noise
        plot : boolean
            Plot the generated signal
    Returns:
        signal : 1d ndarray
            Generated signal, a numpy array of length `sampling_rate*length_seconds`
    """
    
    
    frequencies_list = np.array(frequencies_list, dtype=object)
    nf=frequencies_list.shape[0]
    
    #On complete func pour matcher avec les fréquences fournises
    assert len(func)<=nf, "unuseful functions have been precised"
    while nf>len(func) :
        func.append('sin')
        
    npnts = sampling_rate*length_seconds  # number of time samples 
    
    
    time = np.arange(0, npnts)/sampling_rate
    signal = np.zeros(npnts)
    
    
    assert bornes[0]<bornes[1], "on doit pouvoir extraire entre les bornes fournises"
    
    if alea==True : 
        "Première idée : on tire aléatoirement les coefs selon une loi gaussienne dont les caractéristiques sont tirés aléatiorement"
        for k in range(nf) :
            rd.seed=1024
            mu=rd.uniform(bornes[0],bornes[1])
            sigma=(rd.uniform(bornes[0],bornes[1]))**2
            theta=rd.gauss(mu,sigma)**2 #i wanted a positiv amplitude
        
            if func[k] == "cos":
                signal = signal + theta*np.cos(2*np.pi*frequencies_list[k]*time)
            else:
                signal = signal + theta*np.sin(2*np.pi*frequencies_list[k]*time)
    else : 
        for k in range(nf) :
            if func[k] == "cos":
                signal = signal + np.cos(2*np.pi*frequencies_list[k]*time)
            else:
                signal = signal + np.sin(2*np.pi*frequencies_list[k]*time)
    
    for t in range(len(signal)) :       
        signal[t]=sum(trend[k]*signal[t]**k for k in range(len(trend)))
               
    if add_noise:        
        noise = np.random.uniform(low=0, high=add_noise, size=(frequencies_list.shape[0],npnts))
        signal = signal + noise

    if plot:
        plt.plot(time, signal.T)  
        plt.show()
       
    return signal,time,length_seconds


def perturbation(signal,time,length_seconds):
    '''creates (and shows) a linear piecewise perturbation that matchs with the signal
    Args : 
        signal ; array, the signal you want to disturb
        time : an array, the temporal values that correspond to the numpy signal
        length_seconds : int, duration of signal'''
    rd.seed=1024
    u=rd.uniform(0,length_seconds)
    print(u)
    
    def fun(n,x):
            if x <= n :
                return -x
            else :
                return x
        
    vfun=np.vectorize(fun)
    
    perturbation=vfun(u,time)
    print(np.size(perturbation))
    print(np.size(signal))
    
    plt.plot(time, perturbation.T)
    plt.show()
    
    return(perturbation)
    
def register_signal(signal,identifiant) :
    '''Stores the signal in a csv file
    Args :
        - signal : an array which cointains signal(s) values (eventually multidimensionnel)
        - time : an array which contains the time values that match with signal
        - identifiant : will be the name of your csv file'''
    name='{}.csv'.format(identifiant)
    with open(name, "w", newline='') as csvfile :
        writer=csv.writer(csvfile, delimiter=',')
        if np.size(signal.shape)>1 : #signal eventuellement multidimensionnel
            i=signal.shape[1] #it
            for k in range(i):
                writer.writerow(signal) # generalisation par boucles iteratives
        else :
            writer.writerow(signal)
    return(1)  

def read_signal(filename):
    name='{}.csv'.format(filename)
    x_tl=[]
    with open(name, "r") as file:
        reader = csv.reader(file, delimiter=',')
        #un peu space mais l'idée est de se débarasser des en-têtes
        for line in reader:
            x_tl.append(line)
    print(x_tl)
    return(1)
