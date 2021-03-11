import sys
import os
sys.path.append(os.getcwd())
import argparse

import generate_signal as gs
from data import get_data
import numpy as np

def main(name,iterations=1000):
    
    ''' First programm to execute. It sets datas format, therefor the following parameters shouldn't be modified then :
    - backcast and forecast length 
    - iteration : determines the number of samples
    - signal : choose the caracteristics of the signal that will be analyzed.

    Datas are stored in file.txt easely exploitable, following the format : xtrain_name.txt'''
    
    # we generate the signal which will be analyzed
    length_seconds,sampling_rate=1000, 150 #that makes 15000 pts
    freq_list=[0.5,0.8,0.4,2] #pour 4 frequences, loss devient importante
    print('----creating the signal, plz wait------')
    sig=gs.generate_signal(length_seconds, sampling_rate, freq_list)
    print('finish : we start storing it in a csv file')
    gs.register_signal(sig[0],name)
    print('----we got it : time to create the ndarray-----')

    #hyperparameters of the model go there
    backcast_length=100
    forecast_length = 100
    limit=int(length_seconds*sampling_rate*0.9) #we keep 10% appart to form the testing set                                                       
    xtrain,ytrain, xtest, ytest=get_data(backcast_length, forecast_length,limit, '{}.csv'.format(name),copy=iterations)
    np.savetxt('xtrain_{}.txt'.format(name),xtrain)
    np.savetxt('ytrain_{}.txt'.format(name),ytrain)
    np.savetxt('xtest_{}.txt'.format(name), xtest)
    np.savetxt('ytest_{}.txt'.format(name), ytest)
   
if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('name', help='Name of the storing file')
    parser.add_argument('-iterations', help='Number of line your ndarray data will contain')
    args=parser.parse_args()
    if not args.iterations :
        main(args.name)
    else :
        main(args.name,int(args.iterations))
