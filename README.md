# Stock Price Predictor

## Overview
The project, implemented in Python, uses a long-short term memory (LSTM) recurrent neural network (RNN) model to predict prices for a given stock over a period of time.

## Description
The program firstly downloads the data via the Yahoo Finance API. Note that the first parameter of download() is the stock ticker symbol and so to run the prediction model for other stocks, simply change the first parameter to the appropriate symbol.

A variety of different features are then engineered from the data. These features help provide more accurate predictions as these additional features can help capture trends and patterns in the data. Some of these features are technical indicators, such as Simple Moving Average (SMA), Exponential Moving Average (EMA) and Relative Strength Index (RSI), while the other features are datetime features such as day, month, year, etc.

After creating and electing the features, the data is then scaled. This improves performance as convergence speed and stability of the data is improved due to the narrowing of the scale. The normalisation of the data also reduces the impact of outliers on the training of the model.

Once scaled, the data is then split into training and validation/test datasets. X_train and X_test are filled with a sequence of past stock prices; these act as the input sets for the model (one for training, one for predicting). y_train and y_test are filled with singular stock prices as target values and these are the values that the model will learn to predict based on the X_train and X_test datasets, respectively. 

The training and test sets are then reshaped so that they can be used by the model.

The LSTM model is Sequential, meaning that the model will be a linear stack of layers. The model comprises of an LSTM layer of 128 neurons and a Dense layer which produces a single output value (ie. the predicted stock price). The model is compiled with a mean squared error (MSE) loss function and the Adam optimiser (which is an adaptive learning rate optimiser).

Early stopping is implemented to prevent overfitting as training of the model is stopped when the loss of the validation dataset (val_loss) stops decreasing.

The model is trained with 8 epochs and a batch size of 4. shuffle is set to True (tells that the training data should be shuffled before each epoch) so to improve the robustness of the model.

Epochs are the number of complete passes through the entire training dataset and batch size is the number of training samples used in one forward and backward pass of the model (the dataset is split into smaller subsets called batches). These values were eventually chosen after so that there is a balance between program runtime, accuracy of the model predictions and reduction of overfitting.

The model then predicts the stock prices using X_test as input data.

Various performanice metrics were calculated used to evaluate the model's performance. As the model improves, RMSE (Root Mean Squared Error) and MAE (Mean Absolute Error) should decrease towards 0 while the r^2 score should increase towards 1.

The model losses are plotted and then finally, the predicted stock prices are plotted against the true values (y_test) so that it can be seen how accurate the model is.

## How to install and run the project
Firstly make sure all the required libraries and modules are installed by using pip.

For example: 
```pip install yfinance```

Once all the dependencies are installed, run the program by typing the following command into the terminal:

```python StockPricePredictor.py```

Expect the runtime of the program to be a couple of minutes (< 5 minutes).


