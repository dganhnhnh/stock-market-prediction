
import numpy as np
import pandas as pd
from DataProcessing import Data
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error, root_mean_squared_error
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import helper

class SVR_Model:
    def __init__(self):
        self.name = 'Support Vector Regressor'
        self.model = SVR(kernel='sigmoid',gamma=0.005,C=150)
        self.input_shape = 8


    def prepare_data(self, ticker, train_pct=0.7):
        self.ticker = ticker
        data = helper.prepare_data(ticker, self.input_shape, train_pct, 'standard')
        self.X_train = data['X_train']
        self.Y_train = data['Y_train']
        self.X_test = data['X_test']
        self.Y_test = data['Y_test']

    def train(self):
        self.model.fit(self.X_train,self.Y_train)
        self.model.score(self.X_test,self.Y_test)
        self.Y_pred = self.model.predict(self.X_test)
        return self.Y_pred

    def calculate_loss(self):
        mse, r2, rmse, mape, mae = helper.calculate_loss(self.model, self.X_test, self.Y_test)
        return mse, r2, rmse, mape, mae
    
    def plot_result(self):
        plt.plot(self.Y_pred,color='red',label='Prediction')
        plt.plot(self.Y_test,color='blue',label='True value')
        plt.legend(loc='upper right')
        plt.title(f'{self.ticker} - {self.name}')
        plt.show()