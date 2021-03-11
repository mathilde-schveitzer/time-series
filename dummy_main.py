import sys
import os
sys.path.append(os.getcwd())
import argparse

import numpy as np
from data import get_data
from nbeats_pytorch.model import NBeatsNet
import torch
import matplotlib.pyplot as plt

def main(name,copy,device='cpu'):                                          
    xtrain=np.loadtxt('xtrain_{}.txt'.format(name))
    ytrain=np.loadtxt('ytrain_{}.txt'.format(name))
    xtest=np.loadtxt('xtest_{}.txt'.format(name))
    ytest=np.loadtxt('ytest_{}.txt'.format(name))
    print('-------we got the DATA dude : it works ------------')

    #Reminder of the hyperparameter
    backcast_length=100
    forecast_length = 100
  
    #Definition of the seasonality  model :
    thetas_dim1=4,
    stack_types1=NBeatsNet.SEASONALITY_BLOCK,
    model1= NBeatsNet(device=torch.device(device),backcast_length=backcast_length, forecast_length=forecast_length, stack_types=stack_types1, nb_blocks_per_stack=2, thetas_dim=thetas_dim1, share_weights_in_stack=True, hidden_layer_units=64)

    model1.compile_model(loss='mae', learning_rate=1e-5)
    plt.figure(figsize=(10,10))
    #model training
    model1.fit(xtrain, ytrain, validation_data=(xtest,ytest), epochs=100, batch_size=10)

    model1.save('nbeats_test_seasonality.h5')

    predictions1=model1.predict(xtrain)

    np.savetxt('predictions_seasonality_{}.txt'.format(name),predictions1)
   # print(predictions.shape)

    #Definition of the generic  model :
    thetas_dim2=4,
    stack_types2=NBeatsNet.GENERIC_BLOCK,
    model2= NBeatsNet(device=torch.device(device),backcast_length=backcast_length, forecast_length=forecast_length, stack_types=stack_types2, nb_blocks_per_stack=2, thetas_dim=thetas_dim2, share_weights_in_stack=True, hidden_layer_units=64)

    model2.compile_model(loss='mae', learning_rate=1e-5)

    #model training
    model2.fit(xtrain, ytrain, validation_data=(xtest,ytest), epochs=100, batch_size=10)

    model2.save('nbeats_test_generic.h5')

    predictions2=model2.predict(xtrain)
    np.savetxt('predictions_generic_{}.txt'.format(name),predictions2)
   # print(predictions.shape)

     #Definition of the trend model :
    thetas_dim3=4,
    stack_types3=NBeatsNet.TREND_BLOCK,
    model3= NBeatsNet(device=torch.device(device),backcast_length=backcast_length, forecast_length=forecast_length, stack_types=stack_types3, nb_blocks_per_stack=2, thetas_dim=thetas_dim3, share_weights_in_stack=True, hidden_layer_units=64)

    model3.compile_model(loss='mae', learning_rate=1e-5)

    #model training
    
    model3.fit(xtrain, ytrain, validation_data=(xtest,ytest), epochs=100, batch_size=10)

    model3.save('nbeats_test_trend.h5')

    predictions3=model3.predict(xtrain)
    np.savetxt('predictions_trend_{}.txt'.format(name),predictions3)
    # print(predictions.shape)
    plt.show()
if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('name', help='Name of the storing file')
    parser.add_argument('nb_of_samples', help='Precise the number of samples that will be picked')
    parser.add_argument('-device', help='The device used to execute the algo : cpu or cuda:k')
    args=parser.parse_args()
    if not args.device :
        main(args.name,int(args.nb_of_samples))
    else :
        main(args.name,int(args.nb_of_samples),str(args.device))
