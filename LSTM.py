from tensorflow import keras as keras
import helper

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
        data = helper.prepare_data(ticker, self.input_shape, train_pct)
        X_train = data['X_train']
        self.X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
        X_test = data['X_test']
        self.X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
        self.Y_train = data['Y_train']
        self.Y_test = data['Y_test']


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
        mse, r2, rmse, mape, mae = helper.calculate_loss(self.model, self.X_test, self.Y_test)
        return mse, r2, rmse, mape, mae
    
    def plot_result(self):
        pass
