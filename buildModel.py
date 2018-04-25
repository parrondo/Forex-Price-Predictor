import random
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import models.OHLCV_LSTM2, time
import numpy as np
import pandas as pd
from getIndicators import getMomentumIndicators, getCustomData
from sklearn import preprocessing

def main():
    pair = ""
    timeframe = ""
    dataFile = 'data/USD_CAD_H1_50000_OHLCV2.csv'
    epochs = 10000
    batch_size = 256
    sequenceLength = 24
    normalize = True
    
    #X_train1, y_train, X_test, y_test = load_data(dataFile, sequenceLength, True)
    a,e, X_test, y_test = load_data(dataFile, sequenceLength, True)
    
    b,f, X_test, y_test = load_data("data/EUR_USD_H1_50000_OHLCV2.csv", sequenceLength, True)
    
    c,g, X_test, y_test = load_data("data/NZD_USD_H1_50000_OHLCV2.csv", sequenceLength, True)
    d,h, X_test, y_test = load_data("data/USD_CHF_H1_50000_OHLCV2.csv", sequenceLength, True)
    j,k, X_test, y_test = load_data("data/AUD_JPY_H1_50000_OHLCV2.csv", sequenceLength, True)
    l,m, X_test, y_test = load_data("data/EUR_CHF_H1_50000_OHLCV2.csv", sequenceLength, True)
    n,o, X_test, y_test = load_data("data/EUR_JPY_H1_50000_OHLCV2.csv", sequenceLength, True)
    p,q, X_test, y_test = load_data("data/GBP_USD_H1_50000_OHLCV2.csv", sequenceLength, True) 
    labels = [e,f,g,h,k,m,o,q]
    train = [a,b,c,d,j,l,n,p]
    X_train,y_train = balanceClasses(train,labels)
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    #X_train = np.concatenate((a,b,c,d,j,l,n,p))
    #y_train = np.concatenate((e,f,g,h,k,m,o,q))
    '''
    seed = 12345
    state = np.random.RandomState(seed)
    state.shuffle(X_train)
    state.seed(seed)
    state.shuffle(y_train)
    '''
    print "Sequence Length: ", sequenceLength
    print "Training Samples: ", X_train.shape[0]
    print "Input Shape: ", X_train[0].shape
    print "Feature Size: ", X_train.shape[2]
    time.sleep(2)    
    layers = [1,50,100,1]   
    print "X_TRAIN SAMPLE 2",X_train[0][0]
    time.sleep(1)
    model = models.OHLCV_LSTM2.build_model(layers, X_train, y_train, epochs, batch_size)
    model.save('models/H1_LSTM.h5')


def load_data(dataFile, sequenceLength, normalize):
    df = pd.read_csv(dataFile, header=None)
    df = df.loc[:,1:5]
    df.columns = ["Open", "High", "Low", "Close", "Volume"]
    print df
    x = df[["Volume"]].values.astype(float)
    scaler = preprocessing.MinMaxScaler()
    df['Volume'] = scaler.fit_transform(x)
    print df
    getCustomData(df)
    df['RSI'] = df['RSI']/100
    df['ADX'] = df['ADX']/100
    df['RSI2'] = df['RSI2']/100
    df['RSI3'] = df['RSI3']/100
    df['ADX1'] = df['ADX1']/100
    df['ADX2'] = df['ADX2']/100
    df['SLOWK'] = df['SLOWK']/100
    df['SLOWD'] = df['SLOWD']/100
    df['FASTK'] = df['FASTK']/100
    df['FASTD'] = df['FASTD']/100
    df['ULTOSC'] = df['ULTOSC']/100
    df['WILLR'] = df['WILLR']/-100
    
    df = df.dropna()

    
    print df.head()
    time.sleep(1)
    data = df.values.tolist()
    print data[0]
    sequences = []
    for i in range(len(data) - (sequenceLength +1)):
        sequences.append(data[i: i+sequenceLength+1 ])
   
       
    sequences = np.array(sequences)
   
    #sequences = sequences[:int(.9*len(sequences)), :]
    
    #print sequences.shape[0]
    #print sequences.shape
    #print sequences[0]
    #sequences = np.reshape(sequences, (len(sequences), len(sequences[0])))
    
    #print sequences.shape
    print sequences[0]
    print sequences
    #sequences = np.reshape(sequences, (sequences.shape[0], len(sequences[0]), len(sequences[0][0])))
    #print sequences.shape
    #for window in sequences:
       # for feature in window:
            #dmax,dmin = feature.max(), feature.min()
            #feature = (feature-dmax)/(dmax-dmin)
    
    X_train = sequences[:,:-1]
    y_train = sequences[:,-1]
    print y_train.shape
    c1 = 0
    c2 = 0
    c3 = 0
    label = []
    for d in y_train:
        if d[3] - d[0] > d[0] * .001:
            label.append([1,0,0])
            c1 += 1
        elif d[0] - d[3] > d[0] * .001:
            label.append([0,1,0])
            c2 += 1
        else:
            label.append([0,0,1])
            c3 += 1
    c = min(c1,c2,c3)

   # seed = 111
    #state = np.random.RandomState(seed)
    #state.shuffle(X_train)
    #state.seed(seed)
    #state.shuffle(label)
    
    y_train = label
    


    #row = round(0.9 * sequences.shape[0])
    #train = sequences[:int(row), :]
    #test = sequence[int(row):, :]
    
    #X_train = np.reshape(
    #X_train = X_train[:,:,3:6]
        
    X_test = []
    y_test = []

    X_train = X_train[:,:,5:]
    print "X_TRAIN SAMPLE: ", X_train[0][0]
    time.sleep(1)
    return [X_train, y_train, X_test, y_test]
def balanceClasses(x_train,classes):
    a = []
    b = []
    overallMin = 50000
    for labels in classes:
       
        minCount = min(labels.count([1,0,0]),labels.count([0,1,0]), labels.count([0,0,1]))
        print minCount
        overallMin = min(minCount,overallMin)
    print "Overall Min: ", overallMin
    time.sleep(5)
    for i,labels in enumerate(classes):
        sequences = x_train[i]
        a_indices = [j for j, x in enumerate(labels) if x == [1,0,0]]
        b_indices = [j for j, x in enumerate(labels) if x == [0,1,0]]
        c_indices = [j for j, x in enumerate(labels) if x == [0,0,1]]
       
        print len(a_indices) 
        sampleA = random.sample(range(0,len(a_indices)), overallMin)
        for j in sampleA:
            a.append(sequences[a_indices[j]])
            b.append(labels[a_indices[j]])
        sampleB = random.sample(range(0,len(b_indices)), overallMin)
        for j in sampleB:
            a.append(sequences[b_indices[j]])
            b.append(labels[b_indices[j]])
        sampleC = random.sample(range(0,len(c_indices)), overallMin)
        for j in sampleC:
            a.append(sequences[c_indices[j]])
            b.append(labels[c_indices[j]])
    
    z = zip(a,b)
    random.shuffle(z)
    a,b = zip(*z)
    
    return a,b
        
         

if __name__ == "__main__":
    main()
