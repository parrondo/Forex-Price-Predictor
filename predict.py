from keras.models import load_model
import numpy as np
from getData import getData

#loads an already trained  model, and uses a sequence of size 50 to predict the next closing price
def main():
	pair = "USD_JPY"
	timeframe = "H1"
	data = getData(pair, timeframe, 51)

	#ignore last datapoint since candle is incomplete
	inputs = data[0:50]
	print "Inputs:\n", inputs
	 
	modelPath = 'models/lstm.h5'
	model = load_model(modelPath)

	normalizedInput, p0  = normalize(inputs)
	#print normalizedInput,p0
	reshapedInput = np.reshape(normalizedInput, (1,50,1))

	normalizedPrediction = predict(model, reshapedInput)
	print "Normalized Prediction:\n", normalizedPrediction

	denormalizedPrediction = denormalize(normalizedPrediction, p0)
	print "Denormalized Prediction:\n", denormalizedPrediction
	
	
def predict( model, inputs ):
	return model.predict(inputs)
	

def normalize(data):
	p0 = float(data[0])
	data = [((float(p) / float(data[0])) - 1) for p in data]
	data = np.array(data)
	return data, p0

def denormalize(n, p0):
	return p0*(n + 1)	

if __name__ == "__main__":
	main()
