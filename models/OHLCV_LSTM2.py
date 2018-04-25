
import time
import warnings
import numpy as np
from numpy import newaxis
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM 
from keras.layers import ConvLSTM2D
from keras.models import Sequential
from keras import callbacks
from keras import optimizers
#import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from sklearn.preprocessing import MinMaxScaler
from keras.utils import normalize
import keras.backend as K
from tensorflow import logical_and
config = tf.ConfigProto()
sess = tf.Session(config=config)
set_session(sess)


def load_data(filename, seq_len, normalise_window):
    df = pd.read_csv('indicators.csv',index_col=0)
    #print df
    df = df.dropna()
    #print df
     
    data = df.values.tolist()
   
    #print data
    print len(data)
    
    sequence_length = seq_len + 1
    result = []
    
    for i in range(len(data) - sequence_length):
        result.append(data[i: i + sequence_length])

    result = np.array(result)
    print result.shape
    if normalise_window:
        for window in result:
            for feature in window:
                dmax,dmin = feature.max(),feature.min()
                feature = (feature-dmax)/(dmax-dmin)
    print result[0].shape
   
        
   
    

    row = round(0.9 * result.shape[0])
    train = result[:int(row), :]
    np.random.shuffle(train)
    
    x_train = train[:, :-1]
    y_train = train[:, -1]
       
    label = []
        
    '''
    for i,candle in enumerate(y_train):
        if (candle[3] -  x_train[i][99][3] > 0.1):
            label.append(1)
        else:
            label.append(0)   
        print candle[3], x_train[i][99][3], label[i]
    '''
    
    for i,candle in enumerate(y_train):
        if float(candle[3]) > float(candle[0]):
            label.append(1)
        else:
            label.append(0)
            
        #print candle[3]
        #print candle
        #label.append(candle[3])
        #predict price
        ##label.append(float(candle[3]))
        
        
    
            
    
    x_test = result[int(row):, :-1]
    y_test = result[int(row):, -1]
    print x_train[0][0]
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 141))
    #x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 36))  

    y_train = label
    return [x_train, y_train, x_test, y_test]
    '''
    def label(y_train):
        label = []
        for i,candle in enumerate(y_train):
            if candle[3] > x_train[i][49][3]:
                label.append(1)
            else:
                label.append(0)
        return label    
    '''
'''
def normalise_windows(window_data):
    normalised_data = []
    for window in window_data:
        close0 = float(window[0][3])
        volume0 = float(window[0][4])
        normalised_window = []
        for i,row in enumerate(window):
            print i
            normalised_row = []
            for j in range(0,4):
                normalised_row.append( float(window[i][j]) / close0 - 1) #normalize OHLC using closing value
            normalised_row.append( float(window[i][4]) / volume0 - 1) #normalize volume data
            for j in range(5,141):  
                normalised_row.append(window[i][j])
            normalised_window.append(normalised_row)
        
        normalised_data.append(normalised_window)
    
        
        
    return normalised_data
'''
def build_model(layers, X_train, y_train, epochs, batch_size):
    model = Sequential()
    print X_train
    model.add(LSTM(units=1024, input_shape=(X_train.shape[1],X_train.shape[2]), return_sequences=True))
    model.add(Dropout(0.25))
    
    model.add(LSTM(1024, return_sequences=True))
    model.add(Dropout(.25))
    model.add(LSTM(1024,return_sequences=True))    
    model.add(LSTM(1024,return_sequences=False))
    
    model.add(Dense(units=3))
    #model.add(Activation("linear"))
    #model.add(Activation('hard_sigmoid'))
    model.add(Activation('softmax'))

    start = time.time()
    #model.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=['accuracy'])
    #reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.2)
    tensorboard = callbacks.TensorBoard(log_dir='./logs')
    checkpoint = callbacks.ModelCheckpoint('checkpoint6.h5', monitor='val_acc', verbose=0, save_weights_only=False, mode='auto', period=1)
    #model.compile(loss="mse", optimizer="rmsprop", metrics=['accuracy', 'mae'])
    #sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    
    model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['accuracy', classA,classB,classC, highConf])
    #i need to write my own mettric for this
    print "Compilation Time : ", time.time() - start
    history = model.fit(X_train, y_train, shuffle=True, batch_size=batch_size, epochs = epochs, validation_split=0.05, callbacks=[checkpoint, tensorboard])
    print "Sequences shape: ",sequences.shape

   
    return model
def acc_A(y_true,y_pred):
    class_id_true = K.argmax(y_true, axis=-1)
    class_id_preds = K.argmax(y_pred, axis=-1)
    # Replace class_id_preds with class_id_true for recall here
    accuracy_mask = K.cast(K.equal(class_id_preds, 0), 'int32')
    class_acc_tensor = K.cast(K.equal(class_id_true, class_id_preds), 'int32') * accuracy_mask
    class_acc = K.sum(class_acc_tensor) / K.maximum(K.sum(accuracy_mask), 1)
    return class_acc
    
def classA(y_true,y_pred):
    classID = K.argmax(y_pred, axis=-1)
    acc = K.cast(K.equal(classID, 0), 'int32')
    return K.sum(acc)
    
def classB(y_true,y_pred):
    classID = K.argmax(y_pred, axis=-1)
    acc = K.cast(K.equal(classID, 1), 'int32')
    return K.sum(acc)
    
def classC(y_true,y_pred):
    classID = K.argmax(y_pred, axis=-1)
    acc = K.cast(K.equal(classID, 2), 'int32')
    return K.sum(acc)

#create metric for confidence, number of predictions A or B with high confidence(55% assuming binary) that are correct
#for this use 35% conf
def highConf(y_true,y_pred):
    classID = K.argmax(y_pred,axis=-1)
    classID_true = K.argmax(y_true, axis = -1)
    #tensor of binary, 0 = less than .345 conf, 1 = greater
    
        

    maxVal = K.max(y_pred, axis=-1)
    highConf = K.greater(maxVal, 1/3 )
    
    A_pred = K.equal(classID, 0)
    B_pred = K.equal(classID,1)
    
    highConfA = logical_and(A_pred,highConf )
    
    highConfB = logical_and(B_pred,highConf )
    
        
    #highConfA_correct = K.cast(K.equal(highConfA, K.cast(K.equal(classID_true,0),'int32')),'int32')
    highConfA_correct = K.cast(logical_and(highConfA, K.equal(classID_true,0)),'int32')
    highConfB_correct = K.cast(logical_and(highConfB,K.equal(classID_true,1)),'int32')
    highConfA_wrong = K.cast(logical_and(highConfA, K.not_equal(classID_true,0)),'int32')
    highConfB_wrong = K.cast(logical_and(highConfB, K.not_equal(classID_true,1)),'int32')
    
    countHighConfAB_correct = K.sum(highConfA_correct) + K.sum(highConfB_correct)
    countHighConfAB_wrong = K.sum(highConfA_wrong) + K.sum(highConfB_wrong)
   
    return K.switch(K.equal(countHighConfAB_correct,0),tf.to_float(countHighConfAB_correct),tf.to_float(tf.divide(countHighConfAB_correct, (countHighConfAB_wrong + countHighConfAB_correct))))  

    #    return countHighConfAB_correct
    #else:        
    #    return countHighConfAB_correct / ( countHighConfAB_wrong + countHighConfAB_correct)
    
    
