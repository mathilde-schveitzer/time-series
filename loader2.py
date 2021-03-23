import argparse
import numpy as np
import matplotlib.pyplot as plt

def main(filename):

    plt.figure(figsize=(10,10))

    for k in range(1,6) :
        filename1='./data/{}/out/testloss_{}.txt'.format(filename,k)
        tab1=np.loadtxt(filename1)

        filename2='./data/{}/out/trainloss_{}.txt'.format(filename,k)
        tab2=np.loadtxt(filename2)

        liste1=filename1.split('/')
        name1=liste1[-1]

        liste2=filename2.split('/')
        name2=liste2[-1]
   

        plt.plot(tab1, label=name1)
        plt.plot(tab2, label=name2)

    plt.legend()
    plt.title('Test sur nb block')
    plt.savefig('./data/{}/out/loss_'.format(filename))

if __name__ == '__main__' :

    main('nblock_test')
