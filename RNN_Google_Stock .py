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
import yfinance as yf 
import ipywidgets as widgets
from ipywidgets import interact, interact_manual
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Use 'warnings' to avoid warning messages 
warnings.filterwarnings('ignore')

# import six-year Google Stock data 
goog = yf.Ticker("GOOGL")
goog_6y = goog.history(period="6y")

# Data that I want to focus on are 'Open','High','Low','Closing', and Volume
goog = goog_6y.iloc[:,0:5]

# Diagnose whether there are null data 
goog.isnull().sum()


# And we can visualize each column of data (time series data) by using 'interact' function
def f(col):
    col_value=goog[col]
    plt.plot(col_value)
interact(f, col=list(goog.columns))

# This time, let's divide data into train and test. The train ratio is 0.7
train_ratio = 0.7
train_len = int(train_ratio * goog.shape[0])
train_stock = goog[:train_len]
test_stock = goog[train_len:]

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
# n_feature is the number of columns 
# In this script, I will predict 'Close'(Price) by using 'Open','High','Low','Close', and 'Volume'.

n_features = train.shape[1]  # 5 
# And I'd like to set the period of past as one month. 
# Except for weekends, one month is 22 days. I will use these days to predict the future one day
n_past = 22
n_future = 1

# Let me use the user definition function that I made above, split_series.
X_train, y_train = split_series(train.values,n_past,n_future, target_col=goog.columns.get_loc('Close'))

# Warning! we must not use test data as it is, but start as must as n_past period before.
# Because we want to predict the first Closing price in test data.
test_rnn = pd.concat([train[-n_past:],test])

# And same as train, make test data
X_test, y_test = split_series(test_rnn.values, n_past,n_future, target_col=goog.columns.get_loc('Close') )

# Before modeling LSTM, We have to do Grid Search.
# In this essay, I'd like to focus on just number of neurons in the first layer.
# First, set up a kfold used in Grid Search 
kfold = KFold(n_splits = 3)

# Second, we need to code the function as below.

def build_lstm(n_neurons):
    model = Sequential()
    model.add(LSTM(units=n_neurons, activation='tanh', input_shape=(n_past, n_features)))
    model.add(Dense(1, activation='tanh'))
    model.compile(loss='mse',optimizer=tf.keras.optimizers.RMSprop(0.001), metrics=['mse'])
    return model

# Set 10, 20, 30 as candidates. And convert to a dictionary format.
n_neurons = [10, 20, 30]
param_grid = dict(n_neurons=n_neurons)

# And use KerasRegressor because we want to predict numeric data, not categorical.
# Next, do randomized search CV
model_candi = KerasRegressor(build_fn = build_lstm)
grid_lstm = RandomizedSearchCV(estimator=model_candi, cv=kfold, param_distributions=param_grid)

# Last fit candidates to find out the best model.
# I set verbose = 0 to hide unnecessary information.
grid_lstm.fit(X_train, y_train, epochs=50, batch_size=20, verbose=0)

# Below is the best lstm model 
best_lstm =grid_lstm.best_estimator_

# Retrain the best model. In this time, I used EarlyStopping to prevent the overfitying problem.
# And I set a validation portion as 0.2. 
# Don't be confused between validation and test data 
es = EarlyStopping(monitor='val_loss', mode='min', patience=5)
history_lstm = best_lstm.fit(X_train, y_train, epochs=50, batch_size=20, validation_split=0.2, callbacks=[es],
                            verbose=0)

# To diagnose the performance of the model, I compared Mean Sqaure Error of train and valid.
train_mse = history_lstm.history['mse']
valid_mse = history_lstm.history['val_mse']

# Plotting the mse of train and validation data 
plt.plot(train_mse, label='train mse')
plt.plot(valid_mse, label='validation mse')
plt.ylabel('mse')
plt.xlabel('epoch')
plt.legend(loc='upper center', bbox_to_anchor=(0.5,-0.15), fancybox=True, shadow=False, ncol=2)

# Last, check the performance of the model numerically by using test data, (not validation).
# There is a need to put back predicted scaled data.
pred_lstm = best_lstm.predict(X_test).reshape(-1,1)
scaler_close = scalers['scaler_Close']
pred_lstm = scaler_close.inverse_transform(pred_lstm)
y_test = scaler_close.inverse_transform(y_test)

# Metrics are mean sqaured error and mean absolute percentage error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
mse_lstm = mean_squared_error(y_test, pred_lstm)
mape_lstm = mean_absolute_percentage_error(y_test, pred_lstm)

# We've seen how to build LSTM Model. 
# We just predicted one day. But can we predict five days, not just one?
# Not only that, can we predict not just 'Close Price' but every column (Open, High, ... etc)
# Yes, through Seq2Seq

# Set the past and future period. But, this time, n_future is 5
n_features = train.shape[1]
n_past= 22
n_future = 5

# And make input and target data of train and test data. This moment, target data are comprised of all columns unlike lstm.
X_train, y_train = split_series(series=train.values,n_past=n_past, n_future=n_future, target_col=list(range(len(goog.columns))))
test_seq2 = pd.concat([train[-n_past:],test])
X_test, y_test = split_series(series=test_seq2.values,n_past=n_past, n_future=n_future, target_col=list(range(len(goog.columns))))

# Now, similary to LSTM Model we saw before, we will do grid search on Seq2Seq
# This time, I set activation functions as candidates
def build_e1d1(activation):
    input_train = Input(shape=(n_past,n_features))
    output_train = Input(shape=(n_future,n_features))
    encoder_last_h1, encoder_last_h2, encoder_last_c = LSTM(
    100, return_sequences=False, return_state=True, activation=activation)(input_train)
    encoder_last_h1 = BatchNormalization(momentum=0.6)(encoder_last_h1)
    encoder_last_c = BatchNormalization(momentum=0.6)(encoder_last_c)
    decoder = RepeatVector(output_train.shape[1])(encoder_last_h1)
    decoder = LSTM(100, activation=activation,return_state=False, return_sequences=True)(
    decoder, initial_state=[encoder_last_h1, encoder_last_c])
    out = TimeDistributed(Dense(output_train.shape[2]))(decoder)
    model = Model(inputs=input_train, outputs=out)
    opt = Adam(learning_rate=0.01)
    model.compile(loss='mse', optimizer=opt, metrics=['mse'])
    return model

# And cadidates are 'elu' and 'relu'
activation = ['elu','relu']
param_grid = dict(activation=activation)

# Next steps are almost same with LSTM
# Randomized Search CV 
model_seq2 = KerasRegressor(build_fn=build_e1d1)
grid_seq2 = RandomizedSearchCV(estimator=model_seq2, cv=kfold, param_distributions=param_grid)
# Fitting
grid_seq2.fit(X_train, y_train, epochs=50)
# Find the best model.
best_seq2 = grid_seq2.best_estimator_
es =EarlyStopping(monitor='val_loss', mode='min', patience=5)
# Refitting 
history_seq2 = best_seq2.fit(X_train, y_train, epochs=50, validation_split=0.2, callbacks=[es],
                            verbose=0)

# Compare MSE of train and validation data 
train_mae = history_seq2.history['mse']
valid_mae = history_seq2.history['val_mse']
plt.plot(train_mse, label='train mse')
plt.plot(valid_mse, label='validation mse')
plt.ylabel('mse')
plt.xlabel('epoch')
plt.legend(loc='upper center', bbox_to_anchor=(0.5,-0.15), fancybox=True, shadow=False, ncol=2)

# Evaluate the performance through test data, not validation
pred_e1d1 = best_seq2.predict(X_test)

for index, i in enumerate(goog.columns):
    scaler = scalers['scaler_'+i]
    pred_e1d1[:,:,index]=scaler.inverse_transform(pred_e1d1[:,:,index])
    y_test[:,:,index] = scaler.inverse_transform(y_test[:,:,index])
    
# We have predicted 5 days. So 
for index,i in enumerate(goog.columns):
    print(i)
    for j in range(0,5):
        print('Day',test.index[j],':')
        print('MSE:', mean_squared_error(y_test[:,j,index], pred_e1d1[:,j,index]))
    print()
    print()



