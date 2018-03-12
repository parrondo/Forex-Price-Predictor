import oandapy
import json
import csv
from datetime import datetime, timedelta
from dateutil import parser

account = ""
access_token = ""
with open('config.json', 'r') as f:
		config = json.load(f)
		
#environment = raw_input("Live or practice: ")
environment = "live"
if (environment == "live"):
	account = config['liveAccount']
	access_token = config['liveToken']

def getDataCSV(pair, granularity, numDataPoints, OHLCV, time):
	oanda = oandapy.API(environment=environment, access_token=access_token)

	end = datetime.utcnow()
	if OHLCV:
		file = open(pair+'_'+granularity+'_'+str(numDataPoints)+'_OHLCV.csv', 'w')
	else:
		file = open(pair+'_'+granularity+'_'+str(numDataPoints)+'.csv', 'w')
		
	#can get max 5k datapoints in 1 call to the API
	count = 5000
	data = []
	print "getting data"
	while numDataPoints > 0:

		if numDataPoints < 5000:
			count = numDataPoints
		print end.isoformat('T')
		response = oanda.get_history(instrument = pair, granularity=granularity, count = count, candleFormat="midpoint", end=end.isoformat('T'))['candles']
		end = parser.parse(response[0]["time"])
		data += reversed(response)
		numDataPoints -= count
	
	for candle in reversed(data):
		if not OHLCV:
			if not time:
				file.write(str(candle['closeMid']) + '\n')
			else:
				file.write(str(candle['time'])+','+str(candle['closeMid']) + '\n')
		else:
			if not time:
				file.write(str(candle['openMid']) +',' + str(candle['highMid'])+','+str(candle['lowMid'])+','+str(candle['closeMid'])+','+str(candle['volume'])+ '\n')
			else:
				file.write(str(candle['time']) +','+ str(candle['openMid']) +',' + str(candle['highMid'])+','+str(candle['lowMid'])+','+str(candle['closeMid'])+','+str(candle['volume'])+ '\n')
			
	
def main():
	pair = raw_input("Enter pair: ")
	granularity = raw_input("Enter granularity: ")
	numDataPoints = raw_input("Enter number of datapoints: ")
	OHLCV = raw_input("OHLCV? (y/n): ")
	if (OHLCV == 'y'):
		OHLCV = True
	else:
		OHLCV = False
	time = raw_input("Time? (y/n): ")
	if (time == 'y'):
		time = True
	else:
		time = False
	getDataCSV(pair, granularity, int(numDataPoints), OHLCV, time)

		
if __name__ == "__main__":
	main()


