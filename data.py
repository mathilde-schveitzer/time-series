import csv
import numpy as np
from tqdm import tqdm

def get_data(backast_length, forecast_length, limit, filename,copy=1):
    """
    Create the set of datas that will be used to train the neural networks, from the file filename
    
    Args :
         - backast_length : int
               The number of points used for the prediction (x vector dimension)
         - forecast_length : int 
               The number of points predicted (y vector dimension)
         - limit : int
               Delimitate the training and the validation set (the validation set will be picked for k>limit only)
         - filename : string
               Name of the file that contains the signal, must be a .csv
         - copy : int
               After getting the signal, the algo copies it copy times in a numpy array. Each copy corresponds to a sample(x,y)
"""
    xtrain = np.array([]).reshape(0, backast_length)
    ytrain = np.array([]).reshape(0, forecast_length)

    xtest=np.array([]).reshape(0, backast_length)
    ytest=np.array([]).reshape(0, forecast_length)

    x_tl = []
    name='{}'.format(filename)
    with open(name, "r") as file:
        reader = csv.reader(file, delimiter=',')
        for line in reader:
           x_tl.append(line)
    x_tl_tl = np.zeros((copy,len(x_tl[0])))
    for i in tqdm(range(copy)):
        x_tl_tl[i,:]=x_tl[0]
        
    time_series = np.array(x_tl_tl[i])
    time_series = [float(s) for s in time_series if s != '']
    time_series_cleaned = np.array(time_series)
        
    time_series_cleaned_fortraining_x = np.zeros((1, backast_length))
    time_series_cleaned_fortraining_y = np.zeros((1, forecast_length))
        
    time_series_cleaned_fortesting_x=np.zeros((1,backast_length))
    time_series_cleaned_fortesting_y=np.zeros((1, forecast_length))
        
    for i in range(x_tl_tl.shape[0]):
        j = np.random.randint(backast_length, limit - forecast_length)#version beta : les echantillons sont selectionnes au hasard avec possibilite de redondance
        k= np.random.randint(limit+backast_length, time_series_cleaned.shape[0]-forecast_length)

        time_series_cleaned_fortraining_x[0, :] = time_series_cleaned[j - backast_length: j]
        time_series_cleaned_fortraining_y[0, :] = time_series_cleaned[j:j + forecast_length]
        
        time_series_cleaned_fortesting_x[0,:]=time_series_cleaned[k-backast_length:k]
        time_series_cleaned_fortesting_y[0,:]=time_series_cleaned[k:k+forecast_length]
        
        
        xtrain = np.vstack((xtrain, time_series_cleaned_fortraining_x))
        ytrain = np.vstack((ytrain, time_series_cleaned_fortraining_y))

        xtest=np.vstack((xtest, time_series_cleaned_fortesting_x))
        ytest=np.vstack((ytest, time_series_cleaned_fortesting_y))

    return xtrain, ytrain, xtest, ytest

