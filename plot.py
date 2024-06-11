from DataProcessing import Data
from LSTM import LSTM_Model
from PSO_SVR import PSO_SVR_Model
from SVR import SVR_Model
from BP_DNN import ANN_BP_Model
from GradientBoosting import GB_Model
from RandomForest import RF_Model
import matplotlib.pyplot as plt


d=Data()
list_tickers = d.get_sp500_tickers()

models = [LSTM_Model(num_epoch=200), PSO_SVR_Model(), ANN_BP_Model(max_iter=200), SVR_Model(), GB_Model(), RF_Model()]

for ticker in list_tickers:
    train_pct = 0.7
    input_shape = 8
    
    d = Data()
    d.preprocess(ticker)
    df = d.get_data(ticker).drop(['Date', 'Volume'], axis=1)

    X = df[df.columns[0:input_shape]].values
    Y = df[df.columns[input_shape]].values

    train_size = int(train_pct*len(X))

    X_train = X[0:train_size]
    Y_train = Y[0:train_size]
    X_test = X[train_size:len(X)]
    Y_test = Y[train_size:len(Y)]
        
    plt.plot(Y_test, label='True value')
    model_labels = [model.name for model in models]

    for model in models:
        model.prepare_data(ticker)
        Y_pred= model.train()
        plt.plot(Y_pred, label=model.name)

    plt.xlabel('Time Step')
    plt.ylabel('Close Price')
    plt.title(f'Predictions for Ticker: {ticker}')
    plt.legend()
    plt.grid(True)

    plt.savefig(f'output/plot/{ticker}.png')
    plt.clf()
