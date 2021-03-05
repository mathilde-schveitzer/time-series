import csv
import numpy as np

def get_data(backcast_length, forecast_length, limit, filename):

    xtrain = np.array([]).reshape(0, backcast_length)
    ytrain = np.array([]).reshape(0, forecast_length)

    xtest=np.array([]).reshape(0, backast_length)
    ytest=np.array([]).reshape(0, forecast_length)

    x_tl = []
    name='{}'.format(filename)
    with open(filename, "r") as file:
        reader = csv.reader(file, delimiter=',')
        for line in reader:
           x_tl.append(line)
    x_tl_tl = np.array(x_tl)
    for i in range(x_tl_tl.shape[0]): 
        time_series = np.array(x_tl_tl[i])
        time_series = [float(s) for s in time_series if s != '']
        time_series_cleaned = np.array(time_series)
        time_series_cleaned_forlearning_x = np.zeros((1, backcast_length))
        time_series_cleaned_forlearning_y = np.zeros((1, forecast_length))
        j = np.random.randint(backcast_length, limit - forecast_length)#version beta : les echantillons sont selectionnes au hasard avec possibilite de redondance
        k= np.random.randint(limit+backast_lenght, time_series_cleaned.shape[0]-forecast_lenght)

        time_series_cleaned_fortraining_x[0, :] = time_series_cleaned[j - backcast_length: j]
        time_series_cleaned_fortraining_y[0, :] = time_series_cleaned[j:j + forecast_length]
        time_series_cleaned_fortesting_x[0,:]=time_series_cleaned[k-backast_lenght:k]
        time_series_cleaned_fortesting_y[0,:]=time_series_cleaned[k:k+forecast_lenght]

        xtrain = np.vstack((xtrain, time_series_cleaned_fortraining_x))
        ytrain = np.vstack((ytrain, time_series_cleaned_fortraining_y))

        xtest=np.vstack((xtest, time_series_cleaned_fortesting_x))
        ytest=np.vstack((ytest, time_series_cleanes_fortesting_y))

    return xtrain, ytrain, xtest, ytest

