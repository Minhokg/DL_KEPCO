# Import every library that We will use 
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from keras.models import Model
from keras.layers import LSTM, Dense, RepeatVector, TimeDistributed, Input
from keras.models import Sequential
from keras.layers import GRU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import dot
from keras.layers import BatchNormalization
from keras.layers import concatenate
from keras.callbacks import EarlyStopping
from sklearn.model_selection import KFold
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import RandomizedSearchCV
import warnings
from pykrx import stock
import ipywidgets as widgets
from ipywidgets import interact, interact_manual
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Use 'warnings' to avoid warning messages 
warnings.filterwarnings('ignore')

# import KEPCO(Korea Electric Power Corporation) data 
Kepco = stock.get_market_ohlcv_by_date(fromdate="20110101", todate="20211210", ticker="015760")

# Diagnose whether there are null data 
Kepco.isnull().sum()

# Set the first data and last date as below 
from datetime import datetime
time_first=datetime(2011,1,3)
time_last = datetime(2021,12,10)

# And we can visualize each column of data like market place by using 'interact' function
def f(col):
    col_value=Kepco[col]
    plt.plot(col_value)
interact(f, col=list(Kepco.columns))

# This time, let's divide data into train and test. The train ratio is 0.7
train_ratio = 0.7
train_len = int(train_ratio * Kepco.shape[0])
train_stock = Kepco[:train_len]
test_stock = Kepco[train_len:]

# Do scaling train data by using MinMaxScaler 
train = train_stock.copy()
scalers = {}
for i in train_stock.columns:
    scaler = MinMaxScaler(feature_range=(-1,1))
    s_s = scaler.fit_transform(train[i].values.reshape(-1,1))
    scalers['scaler_'+i]=scaler
    train[i] = s_s
    
# Likewise, do scaling test data too
test = test_stock.copy()
for i in train_stock.columns:
    scaler = scalers['scaler_'+i]
    s_s = scaler.transform(test[i].values.reshape(-1,1))
    test[i] = s_s
    
    
# Next, the below function is for making time series data. 
def split_series(series, n_past,n_future,target_col=list(range(len(Kepco.columns)))):
    X, y = list(), list()
    for window_start in range(len(series)):
        past_end = window_start + n_past
        future_end = past_end + n_future
        if future_end > len(series):
            break
        past, future = series[window_start:past_end,:], series[past_end:future_end,target_col]
        X.append(past)
        y.append(future)
    return np.array(X), np.array(y)
            
# First, Let me use LSTM Model.


