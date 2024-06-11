import time
from DataProcessing import Data
from LSTM import LSTM_Model
from PSO_SVR import PSO_SVR_Model
from BP_DNN import ANN_BP_Model
from GradientBoosting import GB_Model
from RandomForest import RF_Model
import numpy as np

d=Data()
list_tickers = d.get_sp500_tickers()

models = [LSTM_Model(num_epoch=200), PSO_SVR_Model(), ANN_BP_Model(max_iter=200), SVR_Model(), GB_Model(), RF_Model()]

for model in models:
    mae = []
    mse = []
    r2 = []
    rmse = []
    mape = []
    avg_runtime = 0
    for ticker in list_tickers:
        start = time.time()

        # call model
        model.prepare_data(ticker)
        model.train()
        
        mse_, r2_, rmse_, mape_, mae_ = model.calculate_loss()
        mae.append(mae_)
        mse.append(mse_)
        r2.append(r2_)
        rmse.append(rmse_)
        mape.append(mape_)

        end = time.time()
        avg_runtime += (end-start)

    mae = np.mean(list(mae))
    mse = np.mean(list(mse))
    r2 = np.mean(list(r2))
    rmse = np.mean(list(rmse))
    mape = np.mean(list(mape))
    avg_runtime = avg_runtime/len(list_tickers)

    with open('output/evaluation.csv', 'a') as f:
        f.write(f'{model.name},{mae:.4f},{mse:.4f},{r2:.4f},{rmse:.4f},{mape:.4f},{avg_runtime:.4f}\n')