import numpy as np 
import keras
import matplotlib.pyplot as plt
import pandas
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import setup

'''
	### LSTM Trainer ###

	Scrapes historical asset price data using yfinance and stores in the Historical Data folder. Can select between daily and hourly prices.
	Trains the relevant LSTM and stores in Keras Models folder. Can change the layers of the models in MakeModel().
	
'''


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
	if mode == 'hourly':
		data.to_csv(f'Historical Data\{ticker}_hourly_historical_data.csv')

def readcsv(ticker, mode):
	if mode == 'daily':
		dataframe = pandas.read_csv(f'Historical Data\{ticker}_historical_data.csv', skiprows=3, usecols=[1], header=None, engine='python')
	elif mode == 'hourly':
		dataframe = pandas.read_csv(f'Historical Data\{ticker}_hourly_historical_data.csv', skiprows=3, usecols=[1], header=None, engine='python')
	dataset = dataframe.values
	dataset = dataset.astype('float32')
	return dataset

def makeModel(ticker, look_back, trainX, trainY, mode, epochs=100, batch_size=1, save=True):
	optimizer = keras.optimizers.Adam(learning_rate=0.05)
	if mode == 'daily':
		model = keras.Sequential()
		model.add(keras.layers.Input(shape=(1, look_back)))
		model.add(keras.layers.LSTM(64, return_sequences=True))
		model.add(keras.layers.Dropout(0.1))
		model.add(keras.layers.LSTM(64))
		model.add(keras.layers.Dense(32))
		model.add(keras.layers.Dense(1))
		model.compile(loss='huber', optimizer=optimizer)
		batch_size = 1

	elif mode == 'hourly':
		model = keras.Sequential()
		model.add(keras.layers.Input(shape=(1, look_back)))
		model.add(keras.layers.LSTM(128, return_sequences=True))
		model.add(keras.layers.Dropout(0.1))
		model.add(keras.layers.LSTM(64))
		model.add(keras.layers.Dense(32))
		model.add(keras.layers.Dense(1))
		model.compile(loss='mean_squared_error', optimizer=optimizer)
		batch_size = 1
	epochs = 100
	
	early_stopping = keras.callbacks.EarlyStopping(monitor='loss', patience=5)

	model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, callbacks=[early_stopping])
	if save:
		if mode == 'hourly':
			model.save(f'Keras Models\{ticker}2.keras')
		elif mode == 'daily':
			model.save(f'Keras Models\{ticker}1.keras')

	return model

def trainModel(ticker, mode):

	if mode == 'daily':
		look_back = 10
		period = '5y'
		interval = '1d'
	elif mode == 'hourly':
		look_back = 24
		period = '2y'
		interval = '1h'

	loadcsv(ticker, period, interval, mode)
	dataset = readcsv(ticker, mode)

	scaler = MinMaxScaler(feature_range=(0, 1))
	dataset = scaler.fit_transform(dataset)

	end = int(len(dataset) * 0.80)
	train, test = dataset[:end], dataset[end:]
	
	trainX, trainY = create_dataset(train, look_back)
	testX, testY = create_dataset(test, look_back)

	trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
	testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

	makeModel(ticker, look_back, trainX, trainY, mode)
	
def main():
	tickers = ['GOOGL'] #tickers for which models are to be trained on
	mode = 'daily'
	for ticker in tickers:
		print(ticker)
		trainModel(ticker, mode)

if __name__ == '__main__':
	main()