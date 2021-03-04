import csv
import numpy as np

def get_data(backcast_length, forecast_length, filename):
    x = np.array([]).reshape(0, backcast_length)
    y = np.array([]).reshape(0, forecast_length)
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
        j = np.random.randint(backcast_length, time_series_cleaned.shape[0] - forecast_length)#version beta : les echantillons sont selectionnes au hasard avec possibilite de redondance
        time_series_cleaned_forlearning_x[0, :] = time_series_cleaned[j - backcast_length: j]
        time_series_cleaned_forlearning_y[0, :] = time_series_cleaned[j:j + forecast_length]
        x = np.vstack((x, time_series_cleaned_forlearning_x))
        y = np.vstack((y, time_series_cleaned_forlearning_y))

    return x, y

