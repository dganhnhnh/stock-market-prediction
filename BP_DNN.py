
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
import helper

class ANN_BP_Model:
    def __init__(self, max_iter=400):
        self.name = 'BP-DNN'
        self.input_shape = 8
        self.model = MLPRegressor(
            random_state=1,
            activation='relu',
            solver='adam', 
            max_iter=max_iter,
            hidden_layer_sizes=(100, 100, 100),
            )
        
    def prepare_data(self, ticker, train_pct=0.7):
        self.ticker = ticker
        data = helper.prepare_data(ticker, self.input_shape, train_pct)
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