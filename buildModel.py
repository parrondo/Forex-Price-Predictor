
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import models.lstm, time
import numpy as np
#import getData2

def main():
	pair = ""
	timeframe = ""
	dataFile = 'data/USD_JPY_H1_history2.csv'
	epochs = 3
	X_train, y_train, X_test, y_test = models.lstm.load_data(dataFile, 50, True)
		
	layers = [1,50,100,1]	
	model = models.lstm.build_model(layers, X_train, y_train, epochs)
		
	prediction = models.lstm.predict_point_by_point(model, X_test)
#	print "X_test"
#	print X_test
#	print y_test
#	print prediction

	print "Predictions"
	print prediction
	print "prediction[0]"
	print prediction[0]
	
	
	inputs = []
	i = 0
	for line in reversed(open(dataFile).readlines()): 
		if i == 0:
			line.rstrip()  #ignoring since candle is incomplete
		else:
			inputs.append(float(line.rstrip()))
		i += 1
		if i == 51:
			break

	print "Inputs"
	print inputs
	inputs.reverse()
	#saving for denormalizing      
	p0 = float(inputs[0])
	inputs = [((float(p) / float(inputs[0])) - 1) for p in inputs]
	inputs = np.array(inputs)
	print "Normalized inputs"
	print inputs
	
	inputs = np.reshape(inputs, (1, 50, 1))
	print "Reshaped inputs"
	print inputs

	print model.predict(inputs ), p0
	#predictNext()
def predictNext():
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
