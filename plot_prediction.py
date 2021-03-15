import matplotlib.pyplot as plt
import numpy as np
import generate_signal as gs
import argparse
import random as rd
from tqdm import tqdm


def main(signal, epochs, nb, predictions, nblock=0):

    def merge_line(a,b,k):
        merge=np.zeros(a.shape[1]+b.shape[1])
        merge[:a.shape[1]]=a[k,:]
        merge[a.shape[1]:]=b[k,:]
        return(merge)
    
    
    xtrain=np.loadtxt('./data/{}/datas/xtrain.txt'.format(signal))
    ytrain=np.loadtxt('./data/{}/datas/ytrain.txt'.format(signal))

    predictionpath='./data/{}/predictions/'.format(signal)

    rep=[0,0,0]
    if not predictions[0]==None:
        prediction_to_plot=np.loadtxt(predictionpath+'generic_{}.txt'.format(epochs))
        rep[0]=1
    if not predictions[1]==None :
        prediction2_to_plot=np.loadtxt(predictionpath+'seasonnality_{}.txt'.format(epochs))
        rep[1]=1
    if not predictions[2]==None :
        prediction3_to_plot=np.loadtxt(predictionpath+'trend_{}.txt'.format(epochs))
        rep[2]=1      
    for i in tqdm(range(nb)) :
        plt.figure(figsize=(10,10))
   
        k=rd.randint(0,xtrain.shape[0])

        plt.plot(merge_line(xtrain,ytrain,k),label="original signal")
    
        if rep[0]==1 :
            plt.plot(merge_line(xtrain,prediction_to_plot,k),label='Predicted with Generic Block')
        if rep[1]==1:
            plt.plot(merge_line(xtrain,prediction2_to_plot,k), label='Predicted with Seasonnability Block')
        if rep[2]==1:
            plt.plot(merge_line(xtrain,prediction3_to_plot,k), label='Predicted with Trendy Block')
       
        for id_block in range(nblock):
            prediction_per_block=np.loadtxt(predictionpath+"seasonality_per_block_{}_{}.txt".format(id_block,epochs))
            plt.plot(merge_line(xtrain,prediction_per_block,k),label="Predicted with block num = {}".format(id_block))
        plt.title('predictions num={}'.format(i))
        plt.savefig('./data/{}/out/predictions{}.png'.format(signal,i))
        plt.legend(loc='best')

         
            
if __name__ == '__main__' :

    parser=argparse.ArgumentParser()
    parser.add_argument('signal', help='Name of the signal analyzed : signal')
    parser.add_argument('epochs', help='indicate the number of epochs (used to find out the predictions files)')
    parser.add_argument('nb', help='The number of predictions that will be ploted')
    parser.add_argument('-p1', help='Prediction : Generic is expected (just say yes) ')
    parser.add_argument('-p2', help='Prediction : Seasonnality is expected (just say yes)')
    parser.add_argument('-p3', help='Prediction : Trend is expected (just say yes)')
    parser.add_argument('-f', help='Say true if you want to plot what each block is predicting')
    parser.add_argument('-nblock', help='precise the number of block (TODO : integrate it in the txt file)')
    args=parser.parse_args()
    if args.f :
        main(args.signal, int(args.epochs),int(args.nb),[args.p1,args.p2,args.p3], True, int(args.nblock))
    else :
        main(args.signal,int(args.epochs), int(args.nb),[args.p1,args.p2,args.p3])
