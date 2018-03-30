# ===================================================================================
# Model helper function
# ==================================================================================

import numpy as np
np.random.seed(1)
import numpy as np
import os
import pandas as pd
from keras.models import Model,Sequential,save_model,load_model
from keras.layers import Dense, Input, Dropout, LSTM, Activation,BatchNormalization
from sklearn.model_selection import train_test_split
import random


# ===================================================================================
# Data loading part
# ===================================================================================
def all_stock_name(path = r"C:\working\data_2012_07_2012_12\data"):
    filenames = os.listdir(path)
    temp = [filename for filename in filenames if filename.endswith('.csv')]
    # print temp
    return [x.replace('.csv', '') for x in temp]


def single_stock_data(stock_name):
    data_path = os.path.join(path, stock_name + '.csv')
    df = pd.DataFrame.from_csv(data_path, header=None, parse_dates=[[0, 1]])
    df.columns = ['Open', 'High', 'Low', 'Close', 'Outstanding', 'Turnover']
    # header=['date','time','open','high','low','close','outstanding','turnover']
    return df


def get_file_from_csv(data_path):
    df = pd.DataFrame.from_csv(data_path, header=None, parse_dates=[[0, 1]])
    df.columns = ['Open', 'High', 'Low', 'Close', 'Outstanding', 'Turnover']
    # header=['date','time','open','high','low','close','outstanding','turnover']
    return df


def divide_data(X, Y):
    return train_test_split(X, Y, test_size=0.02, random_state=42)


def norm_ts(Z):
    Z_norm = (Z - Z.min()) / (Z.max() - Z.min())
    invert_func = lambda X: (Z.max() - Z.min()) * X + Z.min()
    return Z_norm, invert_func


def one_step_predict(model, train_features, train_labels):
    model.fit(train_features, train_labels, epochs=1, batch_size=64, shuffle=True)
    return model





def very_simple_benchmark_model(X):
    daytimes = X.shape[1]
    x2 = np.linspace(0, daytimes - 1, num=daytimes)
    out = np.zeros((X.shape[0], 1))
    for s in range(X.shape[0]):
        y1 = X[s, :, 0]
        z = np.polyfit(x2, y1, 2)
        f = np.poly1d(z)
        out[s, 0] = f(daytimes)
    return out

# Generate x and y
# ===================================================================================
def x_y_new(df, indices, outlier_prob=0.5, time_window=60, lookback=55):
    M = len(indices)
    N = time_window + 1
    # add time demension
    X = np.zeros((M, N, 2, 1))
    Y = np.zeros((M))

    # get the close price from the df
    raw_Z = np.zeros((len(df), 7))
    raw_Z[:, 0:6] = df.values
    raw_Z[:, 6] = [x.value for x in df.index]

    norm_Z, invert_func = norm_ts(raw_Z[:, 3])

    # generate trades
    for i in range(M):

        # take a snapshot from the original time series around the trading time
        z_local = norm_Z[indices[i] - lookback: indices[i] - lookback + time_window]

        # create a two line array, one is benchmark trade line, one is actual trade line.
        # the only differences between the two line is the point on actual tradeing time
        # the benchmark line is average between last and next trades, the actual trade line is the actual price
        local_x = np.zeros((N, 2, 1))

        # firstly we create a benchmark trades line, the benchmark trades line contain all normal trades
        # and the price on trading time is calculated by average of the two
        local_x[0:lookback, 0, 0] = z_local[0:lookback]
        local_x[lookback + 1:, 0, 0] = z_local[lookback:]
        local_x[lookback, 0, 0] = (z_local[lookback - 1] + z_local[lookback]) / 2.0

        # then we generate the actual trade line
        local_x[:, 1, 0] = local_x[:, 0, 0]
        reasonable_range = local_x[lookback - 5:lookback + 3, 0, 0]
        min_reasonable = min(reasonable_range)
        max_reasonable = max(reasonable_range)
        temp = random.random()
        if temp > 0.5:
            trade_price = random.random() * (max_reasonable - min_reasonable) + min_reasonable
            Y[i] = 0.0
        elif temp > 0.25:
            trade_price = random.random() * (max_reasonable - min_reasonable) + max_reasonable
            Y[i] = 1.0
        else:
            trade_price = -random.random() * (max_reasonable - min_reasonable) + min_reasonable
            Y[i] = 1.0
        local_x[lookback, 1, 0] = trade_price

        local_x, _ = norm_ts(local_x)

        X[i, :, :, :] = local_x
    return X, Y
