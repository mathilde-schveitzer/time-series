import sys
import os
sys.path.append(os.getcwd())
import argparse

import generate_signal as gs
from data import get_data
from nbeats_pytorch.model import NBeatsNet
import torch

def main(name,copy,device='cpu'):
    # we generate the signal which will be analyzed
    length_seconds,sampling_rate=10000, 150 #that makes 15000 pts
    freq_list=[0.5] #pour 4 frequences, loss devient importante
    print('----creating the signal, plz wait------')
    sig=gs.generate_signal(length_seconds, sampling_rate, freq_list)
    print('finish : we start storing it in a csv file')
    gs.register_signal(sig[0],name)
    print('----we got it : DL session is starting-----')

    #hyperparameters of the model go there
    backcast_length=100
    forecast_length = 100
    limit=int(length_seconds*sampling_rate*0.9) #we keep 10% appart to form the testing set                                                       
    xtrain,ytrain, xtest, ytest=get_data(backcast_length, forecast_length,limit, '{}.csv'.format(name),copy)

    #Definition of the model :
    thetas_dim=4,
    stack_types=NBeatsNet.SEASONALITY_BLOCK,
    model= NBeatsNet(device=torch.device(device),backcast_length=backcast_length, forecast_length=forecast_length, stack_types=stack_types, nb_blocks_per_stack=2, thetas_dim=thetas_dim, share_weights_in_stack=True, hidden_layer_units=64)

    model.compile_model(loss='mae', learning_rate=1e-5)

    #model training
    model.fit(xtrain, ytrain, validation_data=(xtest,ytest), epochs=100, batch_size=10)

    model.save('nbeats_test.h5')

    predictions=model.predict(xtest)
    print(predictions.shape)

if __name__ == '__main__':
    
    parser=argparse.ArgumentParser()
    parser.add_argument('name', help='Name of the storing file')
    parser.add_argument('nb_of_samples', help='Precise the number of samples that will be picked')
    parser.add_argument('-device', help='The device used to execute the algo : cpu or cuda:k')
    args=parser.parse_args()
    if not args.device :
        main(args.name,int(args.nb_of_samples))
    else :
        main(args.name, int(args.nb_of_samples),str(args.device))
