import csv
import numpy as np
from DataProcessing import Data
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error, root_mean_squared_error


def prepare_data(index, train_percent=0.8):
    d = Data()
    # d.preprocess(index)
    df = d.get_data(index)
    X = df[df.columns[0:8]].values
    Y = df[df.columns[8]].values

    # TRAIN_PERCENT = 0.8
    train_size = int(train_percent*len(X))

    X_train = X[0:train_size]
    Y_train = Y[0:train_size]
    X_test = X[train_size:len(X)]
    Y_test = Y[train_size:len(Y)]
    print(f'X_train: {X_train.shape}, Y_train: {Y_train.shape}, X_test: {X_test.shape}, Y_test: {Y_test.shape}')
    return X_train, Y_train, X_test, Y_test

def calculate_loss(model, X_test, Y_test):
    print(f'Calculating loss for model {model}...')
    predictions = model.predict(X_test)

    mse = mean_squared_error(Y_test, predictions)
    print(f'Mean Squared Error: {mse}')

    r2 = r2_score(Y_test, predictions)
    print(f'R-squared: {r2}')

    rmse = root_mean_squared_error(Y_test, predictions)
    print(f'Root Mean Squared Error: {rmse}')

    mape = mean_absolute_percentage_error(Y_test, predictions)
    print(f'Mean Absolute Percentage Error: {mape}')

    mae = np.mean(np.abs(predictions - Y_test))
    print(f'Mean Absolute Error: {mae}')

    return mse, r2, rmse, mape, mae

# write header for csv file
def write_header(file_name):
    with open(file_name, 'w') as f:
        fieldnames = ['Model', 'MSE', 'R2', 'RMSE', 'MAPE', 'MAE', 'Runtime']
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        writer.writeheader()

        # f.write('Model,MSE,R2,RMSE,MAPE,MAE\n')

write_header('output/evaluation.csv')