import sys
import os
sys.path.append(os.getcwd())
import argparse

from dummy_main import trainandsave
import generate_signal as gs
from data import get_data
import numpy as np
import random as rd
import matplotlib.pyplot as plt

def main(name,nb,device='cpu',samples=5000):
    
    ''' First programm to execute. It sets datas format, therefor the following parameters shouldn't be modified then :
    - backcast and forecast length 
    - iteration : determines the number of samples
    - signal : choose the caracteristics of the signal that will be analyzed.

    Datas are stored in file.txt easely exploitable, following the format : xtrain_name.txt'''
    #we create the directories that will be usefull afterwards
    datapath='./data/{}/datas'.format(name)
    os.makedirs(datapath)
    os.makedirs('./data/{}/predictions'.format(name))
    os.makedirs('./data/{}/out'.format(name))
    for k in range(1,nb+1) :
        os.makedirs('./data/{}/predictions/nblocks_{}'.format(name,k))
    # we generate the signal which will be analyzed
    length_seconds, sampling_rate=1000, 150 #that makes 15000 pts
    freq_list=[0.5,0.3,0.2,4,5,6]
    print('----creating the signal, plz wait------')
    sig=gs.generate_signal(length_seconds, sampling_rate, freq_list)
    print('finish : we start storing it in a csv file')
    gs.register_signal(sig[0],'./data/{}/signal'.format(name))
    plt.plot(sig[0])
    plt.title('Analyzed signal')
    plt.savefig('./data/{}/out/signal.png'.format(name))
    print('----we got it : time to create the ndarray-----')

    #hyperparameters of the model go there
    backcast_length=100
    forecast_length = 100
    limit=int(length_seconds*sampling_rate*0.9) #we keep 10% appart to form the testing set                                                       
    xtrain,ytrain, xtest, ytest=get_data(backcast_length, forecast_length,limit, './data/{}/signal.csv'.format(name),copy=samples)
    np.savetxt(datapath+'/xtrain.txt',xtrain)
    np.savetxt(datapath+'/ytrain.txt',ytrain)
    np.savetxt(datapath+'/xtest.txt', xtest)
    np.savetxt(datapath+'/ytest.txt', ytest)
    print('--------- name of the file you used : {} ---------'.format(name))

    trainandsave(name,device,nb)
    
if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('name', help='Name of the storing file')
    parser.add_argument('nb', help='Maximum of blocks that will be tested')
    parser.add_argument('-samples', help='Number of line your ndarray data will contain')
    parser.add_argument('-device', help='Processor used for torch tensor')
    args=parser.parse_args()
    if not args.samples :
        if not args.device :
            main(args.name,int(args.nb))
        else :
            main(args.name,int(args.nb),device=args.device)
    else :
        if not args.device :
            main(args.name,int(args.nb),samples=int(args.samples))
        else :    
            main(args.name,int(args.nb), args.device, int(args.samples))
