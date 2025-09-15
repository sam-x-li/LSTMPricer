import numpy as np 
import keras
import matplotlib.pyplot as plt
import yfinance as yf
import pandas
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import setup

'''
	### LSTM Predictor ###

	Loads Keras models stored in the Keras Models folder. Select daily or hourly mode in main(). Predicts the changes in asset prices specified in setup.py
	Scrapes prices using yfinance (stored in Historical Data, can be used for retraining), and uses these to predict the next price. Produces a report, stored in 
	the Reports folders.
	
'''


stocks = setup.stocks
indices = setup.indices
commodities = setup.commodities
forex = setup.forex
           
def create_dataset(dataset, look_back=7):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return np.array(dataX), np.array(dataY)

def loadcsv(ticker, period, interval, mode):
	data = yf.download(ticker, period=period, interval=interval)
	if mode == 'daily':
		data.to_csv(f'Historical Data\{ticker}_historical_data.csv')
	elif mode == 'hourly':
		data.to_csv(f'Historical Data\{ticker}_hourly_historical_data.csv')

def getData(tickers, mode):
	if mode == 'daily':
		interval = '1d'
	elif mode == 'hourly':
		interval = '1h'
	data = yf.download(tickers, period='1mo', interval=interval)
	data.to_csv(f'{mode}_temp_data.csv') 

def readcsv(ticker, mode):
	if mode == 'daily':
		dataframe = pandas.read_csv(f'Historical Data\{ticker}_historical_data.csv', skiprows=3, usecols=[1], header=None, engine='python')
	elif mode == 'hourly':
		dataframe = pandas.read_csv(f'Historical Data\{ticker}_hourly_historical_data.csv', skiprows=3, usecols=[1], header=None, engine='python')
	dataset = dataframe.values
	dataset = dataset.astype('float32')
	return dataset

def sign(diff):
	if diff > 0:
		return 'INCREASE +'
	elif diff == 0 :
		return 'NO CHANGE '
	else:
		return 'DECREASE '

def getDifference(dataset, pred):
	prev = dataset[-1][0]
	diff = pred - prev
	percentage = 100 * (diff) / prev
	return percentage

def getPrediction(ticker, mode):
	print(ticker)

	if mode == 'daily':
		look_back = 5
		period = '3y'
		interval = '1d'
	elif mode == 'hourly':
		look_back = 24
		period = '2y'
		interval = '1h'

	loadcsv(ticker, period, interval, mode)
	dataset = readcsv(ticker, mode)
	scaler = MinMaxScaler(feature_range=(0, 1))
	scaleDataset = scaler.fit_transform(dataset)
	
	new, temp = create_dataset(scaleDataset, look_back) 
	new = np.reshape(new, (new.shape[0], 1, new.shape[1]))
	
	if mode == 'daily':
		model = keras.models.load_model(f"Keras Models\{ticker}1.keras")
	elif mode == 'hourly':
		model = keras.models.load_model(f"Keras Models\{ticker}2.keras")
	
	prediction = model.predict(np.array([new[-1]]))
	prediction = scaler.inverse_transform(prediction.reshape(1, 1)) 
	scaledPrediction = prediction[0][0]
	
	diff = getDifference(dataset, scaledPrediction) #float, signed percentage difference
	
	return scaledPrediction, diff

def writeLine(tup, f):
	noti = sign(tup[2])
	name = f'{tup[0]}:'
	change = f'{noti}{tup[2]:.2f}%'
	f.write(f'{name:<15}{change:<20}Closing:\t{tup[1]:.5f}\n')

def writeUp(stockPredicts, indexPredicts, CommodityPredicts, forexPredicts, mode):
	if mode == 'daily':
		date = datetime.today().strftime('%Y-%m-%d')
		folder = 'Daily'
	elif mode == 'hourly':
		date = datetime.today().strftime('%Y-%m-%d-%H')
		folder = 'Hourly'

	with open(f'{folder} Reports\{date}.txt', "w") as f:
		f.write(f'{date}\n\n\n\n')
		f.write('Stocks\n\n')
		for stock in stockPredicts:
			writeLine(stock, f)
		
		f.write('\n\nIndices\n\n')
		for i in indexPredicts:
			writeLine(i, f)

		f.write('\n\nCommodities\n\n')
		for c in CommodityPredicts:
			writeLine(c, f)

		f.write('\n\nForex\n\n')
		for fork in forexPredicts:
			writeLine(fork, f)
	

def main():
	mode = 'hourly'
	stockPredicts = []
	indexPredicts = []
	CommodityPredicts = []
	forexPredicts = []
	for stock in stocks:
		prediction, diff = getPrediction(stock, mode)
		stockPredicts.append((stock, prediction, diff))
	for i in indices:
		prediction, diff = getPrediction(i, mode)
		indexPredicts.append((i, prediction, diff))
	for c in commodities:
		prediction, diff = getPrediction(c, mode)
		CommodityPredicts.append((c, prediction, diff))
	for fork in forex:
		prediction, diff = getPrediction(fork, mode)
		forexPredicts.append((fork, prediction, diff))

	stockPredicts.sort(key=lambda tup: -tup[2]) 
	indexPredicts.sort(key=lambda tup: -tup[2])
	CommodityPredicts.sort(key=lambda tup: -tup[2])
	forexPredicts.sort(key=lambda tup: -tup[2])

	writeUp(stockPredicts, indexPredicts, CommodityPredicts, forexPredicts, mode)

if __name__ == '__main__':
	main()