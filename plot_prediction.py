import matplotlib.pyplot as plt
import numpy as np
import generate_signal as gs
import argparse
import random as rd

def main(signal,predictions):

    def merge_line(a,b,k):
        merge=np.zeros(a.shape[1]+b.shape[1])
        merge[:a.shape[1]]=a[k,:]
        merge[a.shape[1]:]=b[k,:]
        return(merge)
    
    plt.figure(figsize=(10,10))
    
    xtrain=np.loadtxt('./data/{}/datas/xtrain.txt'.format(signal))
    ytrain=np.loadtxt('./data/{}/datas/ytrain.txt'.format(signal))

    k=rd.randint(0,xtrain.shape[0])
    
    plt.plot(merge_line(xtrain,ytrain,k),label='original signal')

    predictionpath='./data/{}/predictions/'.format(signal)
    
    if not predictions[0]==None:
        prediction_to_plot=np.loadtxt(predictionpath+'generic.txt')
        plt.plot(merge_line(xtrain,prediction_to_plot,k),label='Predicted with Generic Block')
    if not predictions[1]==None :
        prediction2_to_plot=np.loadtxt(predictionpath+'seasonnality.txt')
        plt.plot(merge_line(xtrain,prediction2_to_plot,k), label='Predicted with Seasonnability Block')
    if not predictions[2]==None :
        prediction3_to_plot=np.loadtxt(predictionpath+'trend.txt')
        plt.plot(merge_line(xtrain,prediction3_to_plot,k), label='Predicted with Trendy Block')

    print(signal) #aide memoire
    plt.legend(loc='best')
    plt.title('predictions')
    plt.savefig('./data/{}/out/predictions.png'.format(signal))
    
    
if __name__ == '__main__' :

    parser=argparse.ArgumentParser()
    parser.add_argument('signal', help='Name of the signal analyzed : signal')
    parser.add_argument('-p1', help='Prediction : Generic is expected (just say yes) ')
    parser.add_argument('-p2', help='Prediction : Seasonnality is expected (just say yes)')
    parser.add_argument('-p3', help='Prediction : Trend is expected (just say yes)')
    args=parser.parse_args()
    main(args.signal,[args.p1,args.p2,args.p3])
  
