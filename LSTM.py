import numpy as np
from DataProcessing import Data
import tensorflow as tf
from tensorflow import keras as keras
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error, root_mean_squared_error


class LSTM_Model:
    def __init__(self, num_epoch=400, batch_size=1, learning_rate=0.001):
        self.name = 'LSTM'
        self.input_shape = 8
        self.num_epoch = num_epoch
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.rnn_model = keras.Sequential()
        self.rnn_model.add(keras.layers.LSTM(100, input_shape=(
            None, self.input_shape), return_sequences=True, activation='relu'))
        self.rnn_model.add(keras.layers.Dense(1))
        self.rnn_model.summary()

    def prepare_data(self, ticker, train_pct=0.7):
        self.ticker = ticker
        d = Data()
        # d.preprocess(ticker)
        df = d.get_data(ticker).drop(['Date', 'Volume'], axis=1)

        X = df[df.columns[0:self.input_shape]].values
        Y = df[df.columns[self.input_shape]].values

        train_size = int(train_pct*len(X))

        X_train = X[0:train_size]
        self.Y_train = Y[0:train_size]
        X_test = X[train_size:len(X)]
        self.Y_test = Y[train_size:len(Y)]

        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        self.X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
        self.X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

    def train(self):
        Adam = keras.optimizers.Adam(self.learning_rate)
        MSE = keras.losses.MeanSquaredError()

        self.rnn_model.compile(optimizer=Adam, loss=MSE)
        self.history = self.rnn_model.fit(self.X_train, self.Y_train, epochs=self.num_epoch, validation_data=(
            self.X_test, self.Y_test), batch_size=self.batch_size)

        Y_pred = self.rnn_model.predict(self.X_test)
        self.Y_pred = Y_pred.reshape(Y_pred.shape[0])

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
        pass
