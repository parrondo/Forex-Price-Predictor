
import time
import warnings
import numpy as np
from numpy import newaxis
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
#import matplotlib.pyplot as plt

#have input shape be (49,50,5)
def load_data(filename, seq_len, normalise_window):
    f = open(filename, 'r').read()
    data = f.split('\n')
    for i,d in enumerate(data):
	data[i] = d.split(',')
    #data = np.array(data)
    #data = data.reshape(49,50,5)
    #print(data.shape)
    sequence_length = seq_len + 1
    result = []
    
    for i in range(len(data) - sequence_length):
        result.append(data[i: i + sequence_length])
    if normalise_window:
        result = normalise_windows(result)
   
    result = np.array(result)
    #stores each normalized window

    row = round(0.9 * result.shape[0])
    train = result[:int(row), :]
    np.random.shuffle(train)
    #now we have a list of 45 windows for training [[sequence of 50], [sequence of 50], etc], each i in sequence of 50 has [O,H,L,C,V]
    
    #x_train stores 45 sequences of 50
    x_train = train[:, :-1]
    #y_train stores 45 labels for each sequence in x_train  (ex: [O,H,L,C,V] of the next step in the sequnce)
    y_train = train[:, -1]
    
    x_test = result[int(row):, :-1]
    y_test = result[int(row):, -1]
    
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1],36))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1],36 ))  

    return [x_train, y_train, x_test, y_test]

def normalise_windows(window_data):
    normalised_data = []
    for window in window_data:
	close0 = float(window[0][3])
	volume0 = float(window[0][4])
	normalised_window = []
	for i,row in enumerate(window):
		normalised_row = []
		for j in range(0,4):
			normalised_row.append( float(window[i][j]) / close0 - 1) #normalize OHLC using closing value
		normalised_row.append( float(window[i][4]) / volume0 - 1) #normalize volume data
		normalised_window.append(normalised_row)
	normalised_data.append(normalised_window)
    
		
		
    return normalised_data

def build_model(layers, X_train, y_train, epochs):
    model = Sequential()
    numFeautres = X_train.shape
    model.add(LSTM(units=50, input_shape=(50,36), return_sequences=True))
    model.add(Dropout(0.25))
    model.add(LSTM(100, return_sequences=False))
    model.add(Dropout(0.25))
    model.add(Dense(units=5))
    model.add(Activation("linear"))

    start = time.time()
    model.compile(loss="mse", optimizer="rmsprop", metrics=['mse', 'mae', 'mape'])
    print "Compilation Time : ", time.time() - start
    model.fit(X_train, y_train, batch_size=512, epochs = epochs, validation_split=0.05) 

   
    return model

