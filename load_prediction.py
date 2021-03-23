import matplotlib.pyplot as plt
import numpy as np
import generate_signal as gs
import argparse
import random as rd
from tqdm import tqdm

def main(signal,nblock,hlu=False):

    def merge_line(a,b,k):
        merge=np.zeros(a.shape[1]+b.shape[1])
        merge[:a.shape[1]]=a[k,:]
        merge[a.shape[1]:]=b[k,:]
        return(merge)

    main_path='./data/{}/predictions/'.format(signal)
    xtrain=np.loadtxt('./data/{}/datas/xtrain.txt'.format(signal))
    ytrain=np.loadtxt('./data/{}/datas/ytrain.txt'.format(signal))

    #construction des predictions par blocks
    storage=[]
    if not hlu :
        for nb in range(0,nblock):
            storage.append(np.loadtxt(main_path+'nblocks_{}/seasonnality_per_block_{}.txt'.format(nblock,nb)))
    else : #n=3 fixe de block
        for block in range(0,3):
            storage.append(np.loadtxt(main_path+'nblocks_{}/seasonnality_per_block_{}.txt'.format(nblock,block)))
        
    plt.figure(figsize=(10,10))
    k=rd.randint(0,xtrain.shape[0])


    plt.plot(merge_line(xtrain,ytrain,k),label="original signal")

    for id_block in range(len(storage)):
        plt.plot(merge_line(xtrain,storage[id_block],k), label='out of block num={}'.format(id_block))
    #construction prediction totale :
    prediction=np.loadtxt(main_path+'seasonnalitytotale_nb{}.txt'.format(nblock))
    plt.plot(merge_line(xtrain,prediction,k), label='signal predicted by sum(blocks)')
    
    plt.legend()
    plt.show()

           
if __name__ == '__main__' :

    parser=argparse.ArgumentParser()
    parser.add_argument('signal', help='Name of the signal analyzed : signal')
    parser.add_argument('nblocks', help='Precise the number of blocks of the NN')
    parser.add_argument('-hlu', help='Are you testing on hlu ?')

    args=parser.parse_args()
    if args.hlu==None :
        main(args.signal,int(args.nblocks))
    else :
        main(args.signal,int(args.nblocks), hlu=True)
