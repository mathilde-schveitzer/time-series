import sys
import os
sys.path.append(os.getcwd())
import argparse

import numpy as np
from data import get_data
from nbeats_pytorch.model import NBeatsNet
import torch
import matplotlib.pyplot as plt

def trainandsave(name,device,nb):

    ''' This creates and trains the NN, from the datas generated by save_data.py :
    - name should be the same as save_data's args.name
    Be carefull you choose the same set of hyperparameters
    It shows the loss. In order to show the signal and its prediction as a function of the NN, the predictions are stored in a txt file'''
    predictionpath='./data/{}/predictions'.format(name)
    datapath='./data/{}/datas'.format(name)
    xtrain=np.loadtxt(datapath+'/xtrain.txt'.format(name))
    ytrain=np.loadtxt(datapath+'/ytrain.txt'.format(name))
    xtest=np.loadtxt(datapath+'/xtest.txt'.format(name))
    ytest=np.loadtxt(datapath+'/ytest.txt'.format(name))
    print('-------we got the DATA dude : it works ------------')
    #Reminder of the hyperparameter
    backcast_length = 100
    forecast_length = 100
    epochs=5000

    
    #Definition of the seasonality  model :
    for k in range(1,nb+1):
        thetas_dim=tuple(4*np.ones(k,int))
        stack_types=NBeatsNet.SEASONALITY_BLOCK,
        model= NBeatsNet(device=torch.device(device),backcast_length=backcast_length, forecast_length=forecast_length, stack_types=stack_types, nb_blocks_per_stack=k, thetas_dim=thetas_dim, share_weights_in_stack=True, hidden_layer_units=32)

        model.compile_model(loss='mae', learning_rate=1e-5)
        model.fit(xtrain, ytrain, name, k, validation_data=(xtest,ytest), epochs=epochs, batch_size=150)

        predictions,elt=model.predict(xtrain,return_prediction=True)
        
        for block in range(elt.shape[0]) :
            np.savetxt(predictionpath+'/nblocks_{}/seasonnality_per_block_{}.txt'.format(k,block), elt[block,:,:])
        np.savetxt(predictionpath+'/seasonnalitytotale_nb{}.txt'.format(k),predictions)
