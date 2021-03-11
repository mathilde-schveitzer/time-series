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
    
    xtrain=np.loadtxt('xtrain_{}.txt'.format(signal))
    ytrain=np.loadtxt('ytrain_{}.txt'.format(signal))

    k=rd.randint(0,xtrain.shape[0])
    
    plt.plot(merge_line(xtrain,ytrain,k),label='original signal')

    if not predictions[0]=='':
        prediction_to_plot=np.loadtxt(predictions[0])
        plt.plot(merge_line(xtrain,prediction_to_plot,k),label='Predicted with Generic Block')
    if not predictions[1]=='' :
        prediction2_to_plot=np.loadtxt(predictions[1])
        plt.plot(merge_line(xtrain,prediction2_to_plot,k), label='Predicted with Seasonnability Block')
    if not predictions[2]=='' :
        prediction3_to_plot=np.loadtxt(predictions[2])
        plt.plot(merge_line(xtrain,prediction3_to_plot,k), label='Predicted with Trendy Block')

    print(signal) #aide memoire
    plt.legend(loc='best')
    plt.show()
    
    
if __name__ == '__main__' :

    parser=argparse.ArgumentParser()
    parser.add_argument('signal', help='Name of the signal analyzed : signal')
    parser.add_argument('-p1', help='Prediction : Generic is expected ')
    parser.add_argument('-p2', help='Prediction : Seasonnality is expected')
    parser.add_argument('-p3', help='Prediction : Trend is expected')
    args=parser.parse_args()
    n=len(vars(args))
    assert n>1, 'plz provide at least one prediction'
    if n==4:
        main(args.signal,[args.p1,args.p2,args.p3])
    elif n==3 :
        if not args.p1 :
            main(args.signal,['',args.p2,args.p3])
        elif not args.p2 :
            main(args.signal,[args.p1,'',args.p3])
        else :
            main(args.signal,[args.p1,args.p2,''])
    else :
        if args.p1 :
            main(args.signal,[args.p1,'',''])
        elif args.p2 :
            main(args.signal,['',args.p2,''])
        else :
            main(args.signal,['','',args.p3])                         
