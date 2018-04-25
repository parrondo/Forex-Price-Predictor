import oandapy
import json
import csv
from datetime import datetime, timedelta

account = ""
access_token = ""
environment = "live"
with open('config.json','r') as f:
	config = json.load(f)
account = config['liveAccount']
access_token = config['liveToken']


def getData(pair, granularity, count, OHLCV=False):
	oanda = oandapy.API(environment=environment, access_token=access_token)
	
	#OHLCV data
	data = oanda.get_history(instrument = pair, granularity = granularity, count = count, candleFormat = "midpoint")
	
	prices = []
	for candle in data['candles']:
		if not OHLCV:
			prices.append(candle['closeMid'])
		else:
			prices.append([candle['openMid'],candle['highMid'],candle['lowMid'],candle['closeMid'],candle['volume']])
	return prices

