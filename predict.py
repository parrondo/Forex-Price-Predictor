from keras.models import load_model
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import models.lstm, time
import numpy as np
#import getData2
from getData import getData

#loads an already trained  model, and uses an sequence of size 50 to predict the next closing price
def main():
	pair = "USD_JPY"
	timeframe = "H1"
	inputs = getData(pair, timeframe, 50)
	print "inputs: ", inputs
	 
	modelPath = 'models/lstm.h5'
	model = load_model(modelPath)
	#print(model.predict(inputs ), p0

	normalizedInput, p0  = normalize(inputs)
	reshapedInput = np.reshape(normalizedInput, (1,50,1))
	normalizedPrediction = predict(model, reshapedInput)
	print "Normalized Prediction: ", normalizedPrediction
	denormalizedPrediction = denormalize(normalizedPrediction, p0)
	print "Denormalized Prediction: ", denormalizedPrediction
	
	
def predict( model, inputs ):
	return model.predict(inputs)
	

def normalize(data):
	p0 = float(data[0])
	data = [((float(p) / float(data[0])) - 1) for p in data]
	data = np.array(data)
	return data, p0

def denormalize(n, p0):
	return p0*(n + 1)	


def predictNext(pair, timeframe):
	#getData2.getData(pair, timeframe)
	predicted, p0 = main(pair, timeframe)
	print "predicted"
	print predicted
	print "De-Normalised prediction"  #pi = p0(ni + 1)
	denormalised = p0*(predicted[0][0] + 1)
	print denormalised
	return predicted[0][0], denormalised	

if __name__ == "__main__":
	main()
