import sys
import os
sys.path.append(os.getcwd())
import argparse

import numpy as np
from data import get_data
from nbeats_pytorch.model import NBeatsNet
import torch
import matplotlib.pyplot as plt

def main(name,epochs=10,device='cpu'):

    ''' This creates and trains the NN, from the datas generated by save_data.py :
    - name should be the same as save_data's args.name
    Be carefull you choose the same set of hyperparameters
    It shows the loss. In order to show the signal and its prediction as a function of the NN, the predictions are stored in a txt file'''
    predictionpath='./data/{}/predictions/'.format(name)
    datapath='./data/{}/datas'.format(name)
    xtrain=np.loadtxt(datapath+'/xtrain.txt'.format(name))
    ytrain=np.loadtxt(datapath+'/ytrain.txt'.format(name))
    xtest=np.loadtxt(datapath+'/xtest.txt'.format(name))
    ytest=np.loadtxt(datapath+'/ytest.txt'.format(name))
    print('-------we got the DATA dude : it works ------------')
    #Reminder of the hyperparameter
    backcast_length=100
    forecast_length = 100
    epochs=epochs

    
    #Definition of the seasonality  model :
    thetas_dim1=(4,4,4)
    stack_types1=(NBeatsNet.SEASONALITY_BLOCK)
    model1= NBeatsNet(device=torch.device(device),backcast_length=backcast_length, forecast_length=forecast_length, stack_types=stack_types1, nb_blocks_per_stack=3, thetas_dim=thetas_dim1, share_weights_in_stack=True, hidden_layer_units=16)

    model1.compile_model(loss='mae', learning_rate=1e-5)
    plt.figure(figsize=(10,10))
    #model training
    model1.fit(xtrain, ytrain, name, validation_data=(xtest,ytest), epochs=epochs, batch_size=150)

    model1.save('nbeats_test_seasonality.h5')

    predictions1,elt=model1.predict(xtrain,return_prediction=True)
    print('------------ ELT.SHAPE----------- {}'.format(elt.shape))
    for k in range(elt.shape[0]) :
          np.savetxt(predictionpath+'seasonality_per_block_{}_{}.txt'.format(k,epochs), elt[k,:,:])
    np.savetxt(predictionpath+'seasonnality_{}.txt'.format(epochs),predictions1)


    #Definition of the generic  model :
  # thetas_dim2=4,
 #  stack_types2=NBeatsNet.GENERIC_BLOCK,
  # model2= NBeatsNet(device=torch.device(device),backcast_length=backcast_length, forecast_length=forecast_length, stack_types=stack_types2, nb_blocks_per_stack=2, thetas_dim=thetas_dim2, share_weights_in_stack=True, hidden_layer_units=64)

  #  model2.compile_model(loss='mae', learning_rate=1e-5)

    #model training
  #  model2.fit(xtrain, ytrain, name, validation_data=(xtest,ytest), epochs=epochs, batch_size=150)

  #  model2.save('nbeats_test_generic.h5')

  #  predictions2=model2.predict(xtrain)
   # np.savetxt(predictionpath+generic),predictions2)

     #Definition of the trend model :
  #  thetas_dim3=4,
  #  stack_types3=NBeatsNet.TREND_BLOCK,
  #  model3= NBeatsNet(device=torch.device(device),backcast_length=backcast_length, forecast_length=forecast_length, stack_types=stack_types3, nb_blocks_per_stack=2, thetas_dim=thetas_dim3, share_weights_in_stack=True, hidden_layer_units=64)

  #  model3.compile_model(loss='mae', learning_rate=1e-5)

    #model training
    
  #  model3.fit(xtrain, ytrain, name, validation_data=(xtest,ytest), epochs=epochs, batch_size=150)

  #  model3.save('nbeats_test_trend.h5')

 #   predictions3=model3.predict(xtrain)
   # np.savetxt(predictionpath+trend),predictions3)

    plt.savefig('./data/{}/out/loss{}.png'.format(name,epochs))

    print('---------- your signal name is {}, and the number of epochs was {} --------'.format(name,epochs))

    
if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('name', help='Name of the storing file')
    parser.add_argument('-epochs', help='Number of epochs')
    parser.add_argument('-device', help='The device used to execute the algo : cpu or cuda:k')
    args=parser.parse_args()
    if not args.device :
        if not args.epochs :
            main(args.name)
        else :
            main(args.name,epochs=int(args.epochs))
    else :
        if not args.epochs :
            main(args.name,str(args.device))
        else :
            main(args.name,int(args.epochs),str(args.device))
