# LSTMPricer
Time series stock/asset price prediction using Keras. Scrapes historical data using yfinance, and trains LSTMs to predict future prices. 
Through this project, I learned about ML models, feature engineering, web scraping, and challenges such as noisy data, overfitting and general 
unpredictability of financial data. I also found that such naive implementations of LSTMs are not particularly effective at predicting price changes. 

## Do not use this to make financial/investment decisions. This is not a reliable tool for making informed financial decisions, merely a personal experiment.

---

## Summary

- Collects historical asset price data using yfinance
- Trains LSTMs using Python and Keras on historical price data
- Tune LSTM layers in LSTMTrainer.py
- Test and evaluate models using LSTMViewer.py

## Installation 
```bash
git clone https://github.com/sam-x-li/LSTMPricer.git
cd LSTMPricer
pip install -r requirements.txt
```

## Usage
Run each program:
```bash
cd src
python LSTMPredictor.py
python LSTMTrainer.py
python LSTMViewer.py
```



```
