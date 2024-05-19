import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib. pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping

#Downloading data from Yahoo Finance
data = yf.download('BP', start='2000-01-01', end='2024-01-01')
data = data.reset_index()

#Creating usable dataset from downloaded data
new_data = pd.DataFrame(index=range(0, len(data)), columns=['Date', 'Adj Close'])
for i in range(0, len(data)):
    new_data['Date'][i] = data['Date'][i]
    new_data['Adj Close'][i] = data['Adj Close'][i]

dates = data['Date']
new_data.index = new_data['Date']
new_data.drop('Date', axis=1, inplace=True)
dataset = new_data.values

#Scaling data to fit inside an interval of (0,1)
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(dataset)

#Creating training and validation/test datasets
split = 0.8
train_data = scaled_data[:int(data.shape[0] * split)]
validation_data = scaled_data[int(data.shape[0] * split):]

X_train, y_train, X_test, y_test = [], [], [], []

#Timesteps tells model to use a given number of past days of data to predict next day's stock price ie. a sliding window is formed.
timesteps = 60
for j in range(timesteps, train_data.shape[0]):
    X_train.append(train_data[j-timesteps:j,0])
    y_train.append(train_data[j,0])

for k in range(timesteps, validation_data.shape[0]):
    X_test.append(validation_data[k-timesteps:k,0])
    y_test.append(validation_data[k,0])
    
X_train, y_train, X_test, y_test = np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)

#Reshape training and test data for model
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

#LSTM Model
model = Sequential()
model.add(LSTM(128, input_shape=(X_train.shape[1], 1), return_sequences=False))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
early_stopping = EarlyStopping(monitor='val_loss', mode='min')
history = model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test), verbose=2, shuffle=True, callbacks=[early_stopping])

#Prediction using model
y_pred = model.predict(X_test)

#Performance Metrics (RMSE, r^2, MAE):
print("\nRMSE: " + str(np.sqrt(mean_squared_error(y_test, y_pred)))) #biased estimator
print("r^2 Score: " + str(r2_score(y_test, y_pred)))
print("MAE: " + str(mean_absolute_error(y_test, y_pred))) #unbiased estimator

#Plotting Model Loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train_data', 'validation_data'])
plt.show()

#Plotting Stock Price Prediction
validation_dates = dates[int(data.shape[0] * split) + timesteps:].reset_index(drop=True)
plt.plot(validation_dates, y_test, label='True Value')
plt.plot(validation_dates, y_pred, label='Predicted Value')
plt.title("Stock Price Prediction")
plt.xlabel('Date')
plt.ylabel('Scaled USD')
plt.legend()
plt.show()