
import numpy as np
import pandas as pd
from DataProcessing import Data
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error, root_mean_squared_error
import matplotlib.pyplot as plt


class ANN_BP_Model:
    def __init__(self, max_iter=400):
        self.name = 'BP-DNN'
        self.input_shape = 8
        self.model = MLPRegressor(
            random_state=44,
            activation='relu',
            solver='adam', 
            max_iter=max_iter,
            hidden_layer_sizes=(100, 100, 100),
            )
        
    def prepare_data(self, ticker, train_pct=0.7):
        self.ticker = ticker
        d = Data()
        # d.preprocess(ticker)
        df = d.get_data(ticker).drop(['Date', 'Volume'], axis=1)

        X = df[df.columns[0:self.input_shape]].values
        Y = df[df.columns[self.input_shape]].values

        train_size = int(train_pct*len(X))

        self.X_train = X[0:train_size]
        self.Y_train = Y[0:train_size]
        self.X_test = X[train_size:len(X)]
        self.Y_test = Y[train_size:len(Y)]

        # from sklearn.preprocessing import MinMaxScaler
        # scaler = MinMaxScaler()
        # self.X_train = scaler.fit_transform(X_train)
        # self.X_test = scaler.transform(X_test)
        
        
    def train(self):
        self.model.fit(self.X_train,self.Y_train)
        self.model.score(self.X_test,self.Y_test)
        self.Y_pred = self.model.predict(self.X_test)
        return self.Y_pred
    
    def calculate_loss(self):
        mse = mean_squared_error(self.Y_test, self.Y_pred)
        print(f'Mean Squared Error: {mse}')

        r2 = r2_score(self.Y_test, self.Y_pred)
        print(f'R-squared: {r2}')

        rmse = root_mean_squared_error(self.Y_test, self.Y_pred)
        print(f'Root Mean Squared Error: {rmse}')

        mape = mean_absolute_percentage_error(self.Y_test, self.Y_pred)
        print(f'Mean Absolute Percentage Error: {mape}')

        mae = np.mean(np.abs(self.Y_pred - self.Y_test))
        print(f'Mean Absolute Error: {mae}')

        return mse, r2, rmse, mape, mae

    def plot_result(self):
        plt.plot(self.Y_pred,color='red',label='Prediction')
        plt.plot(self.Y_test,color='blue',label='True value')
        plt.legend(loc='upper right')
        plt.title(f'{self.ticker} - ANN-BP Regressor')
        plt.show()
