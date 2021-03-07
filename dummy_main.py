import sys
import os
sys.path.append(os.getcwd())


import generate_signal as gs
from data import get_data
from nbeats_pytorch.model import NBeatsNet


def main():
    # we generate the signal which will be analyzed
    length_seconds,sampling_rate=10, 150 #that makes 1500 pts
    freq_list=[0.5,0.8,0.3,0.9] #pour 4 freauences, loss devient importante
    print('----creating the signal, plz wait------') 
    sig=gs.generate_signal(length_seconds, sampling_rate, freq_list)
    print('finish : we start sorting it in a csv file')
    gs.register_signal(sig[0],'first_test4',10)
    print('----we got it : DL session is starting-----')
    #hyperparameters of the model go there
    backcast_length=10
    forecast_length = 1
    limit=int(length_seconds*sampling_rate*0.9) #we keep 10% appart to form the testing set                                                       
    xtrain,ytrain, xtest, ytest=get_data(backcast_length, forecast_length,limit, 'first_test.csv')

    #Definition of the model :
    model= NBeatsNet(backcast_length=backcast_length, forecast_length=forecast_length, stack_types=(NBeatsNet.GENERIC_BLOCK, NBeatsNet.GENERIC_BLOCK), nb_blocks_per_stack=2, thetas_dim=(4,4), share_weights_in_stack=True, hidden_layer_units=64)
    model.compile_model(loss='mae', learning_rate=1e-5)

    #model training
    model.fit(xtrain, ytrain, validation_data=(xtest,ytest), epochs=3, batch_size=128)

    model.save('nbeats_test.h5')

    predictions=model.predict(xtest)
    print(predictions.shape)

if __name__ == '__main__':
    main()
