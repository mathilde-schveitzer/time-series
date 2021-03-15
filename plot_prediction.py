import matplotlib.pyplot as plt
import numpy as np
import generate_signal as gs
import argparse
import random as rd
from tqdm import tqdm

def main(signal,nb,predictions, plot_forecast=False):

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
        prediction_to_plot=np.loadtxt(predictionpath+'generic.txt')
        rep[0]=1
    if not predictions[1]==None :
        prediction2_to_plot=np.loadtxt(predictionpath+'seasonnality.txt')
        rep[1]=1
    if not predictions[2]==None :
        prediction3_to_plot=np.loadtxt(predictionpath+'trend.txt')
        rep[2]=1

        
    for i in tqdm(range(nb)) :
        plt.figure(figsize=(10,10))
   
        k=rd.randint(0,xtrain.shape[0])

        plt.plot(merge_line(xtrain,ytrain,k),label='original signal')
    
        if rep[0]==1 :
            plt.plot(merge_line(xtrain,prediction_to_plot,k),label='Predicted with Generic Block')
        if rep[1]==1:
            plt.plot(merge_line(xtrain,prediction2_to_plot,k), label='Predicted with Seasonnability Block')
        if rep[2]==1:
            plt.plot(merge_line(xtrain,prediction3_to_plot,k), label='Predicted with Trendy Block')
        plt.title('predictions num={}'.format(i))
        plt.savefig('./data/{}/out/predictions{}.png'.format(signal,i))
        plt.legend(loc='best')
        if plot_forecast :
            
if __name__ == '__main__' :

    parser=argparse.ArgumentParser()
    parser.add_argument('signal', help='Name of the signal analyzed : signal')
    parser.add_argument('-idd', help='In order to store prediction without destroying other files')
    parser.add_argument('-p1', help='Prediction : Generic is expected (just say yes) ')
    parser.add_argument('-p2', help='Prediction : Seasonnality is expected (just say yes)')
    parser.add_argument('-p3', help='Prediction : Trend is expected (just say yes)')
    parser.add_argument('-f', help='Say true if you want to plot what each block is predicting')
    args=parser.parse_args()
    if args.idd :
        if ars.f :
            main(args.signal,int(args.idd),[args.p1,args.p2,args.p3],True)
        else :
            main(args.signal,int(args.idd),[args.p1,args.p2,args.p3],True)
      
    else :
        if args.f :
            main(args.signal,10,[args.p1,args.p2,args.p3], True)
        else :
            main(args.signal,10,[args.p1,args.p2,args.p3])
      
