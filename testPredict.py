from keras.models import load_model
import numpy as np
#from getData import getData
import pandas as pd
import matplotlib.pyplot as plt
#loads an already trained  model, and uses a sequence of size 50 to predict the next closing price
def main():
    pair = "USD_JPY"
    timeframe = "H1"

    
    #ignore last datapoint since candle is incomplete

    modelPath = 'checkpoint.h5'
    model = load_model(modelPath)

    #data = getData(pair, timeframe, 60)


    df = pd.read_csv('indicators4.csv',index_col=0)
    df = df.dropna()
    print df
    df = df.tail(150)
    df = df.reset_index(drop=True)
    
    print df
    data = df.values.tolist()
   
    print data
    predictions = []
    for i in range(100):
        inputs = data[i:50+i]
        actualPrice = data[50+i][3]
        actual = data[50+i][3] - data[50+i][0]
        #print "Inputs:\n", inputs
        #normalizedInput, p0  = normalize(inputs)
        #reshapedInput = np.reshape(normalizedInput, (1,50,1))
        reshapedInput = np.reshape(inputs,(1,50,141))
        xmax,xmin = reshapedInput.max(),reshapedInput.min()
        reshapedInput = (reshapedInput - xmin)/(xmax-xmin)
        
        prediction = predict(model, reshapedInput)
        
        
        predictedDirection = 0
        confidence = 1 - float(prediction)
        if (float(prediction) > 0.5):
            predictedDirection = 1
            confidence = float(prediction)
        
        predictions.append([predictedDirection, confidence, actual, actualPrice])
    print predictions
    numCorrect = 0
    total = 0
    num0 = 0
    num1 = 0
    for prediction in predictions:
        confidence = predictions[1]
        if prediction[0] == 1 and prediction[2] > 0:
            numCorrect += 1
        elif prediction[0] == 0 and prediction[2] < 0:
            numCorrect += 1
        if prediction[0] == 0:
            num0 += 1
        else:
            num1 += 1
        total += 1
    print numCorrect
    print total
    
    print "0: ",num0
    print "1: ",num1
    #plot(predictions)
def plot(predictions):
    predictions = np.array(predictions)
    plt.plot(predictions[:,3],'b')
    plt.plot( predictions[:,3] + actual, 'r' )
    for x,i in enumerate(predictions):
        if i[0] == 1:
            plt.plot(x,i[3] - .2, marker=r'$\uparrow$')
        else:
            plt.plot(x,i[3] + .2, marker=r'$\downarrow$')
    
       
    
    plt.show()   
        
def predict( model, inputs ):
    return model.predict(inputs)
    

def normalize(data):
    p0 = float(data[0])
    data = [((float(p) / float(data[0])) - 1) for p in data]
    data = np.array(data)
    return data, p0

def denormalize(n, p0):
    return p0*(n + 1)   

def directionTrue(actual,previous,prediction):
    if actual > previous and prediction > previous:
        return True
    if actual < previous and prediction < previous:
        return True
    return False
    
    

if __name__ == "__main__":
    main()
