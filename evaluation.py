import time
from DataProcessing import Data
from LSTM import LSTM_Model
import numpy as np

mae = []
mse = []
r2 = []
rmse = []
mape = []
avg_runtime = 0

d=Data()
list_tickers = d.get_sp500_tickers()

models = []
models.append(LSTM_Model())

for model in models:
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
        f.write(f'{model.name},{mae},{mse},{r2},{rmse},{mape},{avg_runtime}\n')