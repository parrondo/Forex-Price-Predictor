from keras.models import load_model
import numpy as np
from getData import getData


#train on [n,t - 1000],test on [t-1000, t]

#loads an already trained  model, and uses a sequence of size 50 to predict the next closing price
def main():
	pair = "EUR_USD"
	timeframe = "H1"

	
	#ignore last datapoint since candle is incomplete
	 
	modelPath = 'models/EUR_USD_H1_lstm.h5'
	model = load_model(modelPath)

	data = getData(pair, timeframe, 150)
	
	predictions = []
	for i in range(100):
		inputs = data[i:50+i]
			
		actual = data[50+i]
		print "Inputs:\n", inputs
		normalizedInput, p0  = normalize(inputs)
		reshapedInput = np.reshape(normalizedInput, (1,50,1))

		normalizedPrediction = predict(model, reshapedInput)
		#print "Normalized Prediction:\n", normalizedPrediction

		denormalizedPrediction = denormalize(normalizedPrediction, p0)
		#print "Denormalized Prediction:\n", denormalizedPrediction
		
		predictions.append((str(denormalizedPrediction), (inputs[49],actual)))
	for prediction in predictions:
		print "Prediction: ", prediction[0], "Actual: ",prediction[1][1], "Previous: ",prediction[1][0], "Direction: ", directionTrue(prediction[1][1],prediction[1][0],prediction[0])
	numDirectionTrue(predictions)	
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

def numDirectionTrue(predictions):
	numCorrect = 0
	total = 0
	for prediction in predictions:
		if directionTrue( prediction[1][1], prediction[1][0], prediction[0] ):
			numCorrect += 1
		total += 1
	print "numDirectionCorrect: ", numCorrect
	print "total: ", total

	
def possibleProfit(predictions):
	#pass bids and asks and calculate possible profit,
	pass 

if __name__ == "__main__":
	main()
