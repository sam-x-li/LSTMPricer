import numpy as np 
import keras
import matplotlib.pyplot as plt
import pandas
from sklearn.preprocessing import MinMaxScaler
import setup

'''
	### LSTM Tester/Viewer ###

	Tests LSTMs on given historical data for tickers specified in setup.py. Then plots the LSTM's predictions against
	the historical data, along with the loss (error). Used to evaluate models - can change the layers in LSTMTrainer.py.

'''

def create_dataset(dataset, look_back=7):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		dataX.append(dataset[i:(i+look_back), :])
		dataY.append(dataset[i + look_back, 0])
	return np.array(dataX), np.array(dataY)

def readcsv(ticker, mode):
	if mode == 'daily':
		dataframe = pandas.read_csv(f'Historical Data\{ticker}_historical_data.csv', skiprows=3, usecols=[1, 2, 3, 4], header=None, engine='python')
	elif mode == 'hourly':
		dataframe = pandas.read_csv(f'Historical Data\{ticker}_hourly_historical_data.csv', skiprows=3, usecols=[1, 2, 3, 4], header=None, engine='python')
	dataset = dataframe.values
	dataset = dataset.astype('float32')
	return dataset

def predict(trainX, testX, testY, scaler, ticker, mode):
	if mode == 'daily':
		model = keras.models.load_model(f"Keras Models\{ticker}1.keras")
	elif mode == 'hourly':
		model = keras.models.load_model(f"Keras Models\{ticker}2.keras")
	
	trainPredict = model.predict(trainX)
	testPredict = model.predict(testX)

	dummy_train = np.zeros((trainPredict.shape[0], 4))  # Create zeros with (589,5) shape
	dummy_train[:, 0] = trainPredict[:, 0]  # Assign predicted Close values to column index 3 (Close)

	dummy_test = np.zeros((testPredict.shape[0], 4))
	dummy_test[:, 0] = testPredict[:, 0]

    # Inverse transform
	trainPredict = scaler.inverse_transform(dummy_train)[:, 0]  # Extract only Close price after transform
	testPredict = scaler.inverse_transform(dummy_test)[:, 0]

	score = model.evaluate(testX, testY, batch_size=1)

	return trainPredict, testPredict, score

def plot(ticker, dataset, look_back, trainPredict, testPredict, scaler, score):
	trainPredictPlot = np.empty_like(dataset[:, 0])
	trainPredictPlot[:] = np.nan
	trainPredictPlot[look_back:len(trainPredict)+look_back] = trainPredict
	testPredictPlot = np.empty_like(dataset[:, 0])
	testPredictPlot[:] = np.nan
	testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1] = testPredict
	plt.plot(scaler.inverse_transform(dataset)[:, 0])
	plt.plot(trainPredictPlot)
	plt.plot(testPredictPlot)
	plt.title(f'{ticker}: Loss = {score:.2E}')
	plt.show()	

def plotModel(ticker, mode):
	if mode == 'daily':
		look_back = 10
	elif mode == 'hourly':
		look_back = 24
	dataset = readcsv(ticker, mode)

	scaler = MinMaxScaler(feature_range=(0, 1))
	dataset = scaler.fit_transform(dataset)

	end = int(len(dataset) * 0.80)
	train, test = dataset[:end], dataset[end:]
	trainX, trainY = create_dataset(train, look_back)
	testX, testY = create_dataset(test, look_back)

	trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], trainX.shape[2]))
	testX = np.reshape(testX, (testX.shape[0], testX.shape[1], testX.shape[2]))
	
	trainPredict, testPredict, score = predict(trainX, testX, testY, scaler, ticker, mode)
	plot(ticker, dataset, look_back, trainPredict, testPredict, scaler, score)

def main():
	mode = 'daily'
	tickers = setup.stocks
	for ticker in tickers:
		plotModel(ticker, mode)

if __name__ == '__main__':
	main()
