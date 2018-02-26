
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import models.lstm, time
import numpy as np
#import getData2


#in this file use it just to make a prediction, loads a model and calls it to predict
def main():
	pair = ""
	timeframe = ""
	modelPath = 'models/lstm.h5'
	model = keras.models.load_model(modelPath)
	return model.predict(inputs ), p0

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
