import numpy as np
np.random.seed(1)
import numpy as np
import datetime as dt
import os
import pandas as pd
from model_helper import divide_data

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from keras.models import Model
from keras.layers import Dense, Input, Dropout, LSTM, Activation



def all_stock_name():
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

def x_y_new(df):
    N = 60
    M = len(df) - N - 10
    X = np.zeros((M, N, 1))
    Y = np.zeros((M, 1))
    Z = df.values
    for i in range(M):
        X[i, :, 0] = Z[i:i + N, 3]
        Y[i,0] = (Z[i + N , 3] - Z[i + N-1, 3])/Z[i + N-1, 3]
        #forward = Z[i + N + 1, 3] - Z[i + N, 3]
        #now = Z[i + N, 3] - Z[i + N - 1, 3]
        #if forward * now < 0:
            #Y[i, 0] = 1

    return X, Y


def number_to_category(num):
    return


def return_y_time(d1):
    return (dt.datetime(d1.year, d1.month, d1.day, 14, 30),
            dt.datetime(d1.year, d1.month, d1.day, 14, 31))


# Model: Price_Forecast
def Price_Forecast(input_shape):
    embeddings = Input(shape=input_shape, dtype=np.float64)

    print(embeddings.shape, "....")
    X = embeddings
    X = LSTM(32)(X)
    #X = Dense(1)(X)
    X = Dense(1,activation="tanh")(X)
    model = Model(embeddings, X)
    return model




path = r"C:\working\data_2012_07_2012_12\data"
all_names = all_stock_name()
# print all_names

Pos = []

name = all_names[2]
df = single_stock_data(name)
X, Y = x_y_new(df)

print(X.shape)
print(Y.shape)

slice = 200
X=X[0:slice,:,:]
Y=Y[0:slice,:]

#X = (X - min (X))/(max(X) - min(X))
Y = (Y - min (Y))/(max(Y) - min(Y))


train_features, test_features, train_labels, test_labels = divide_data(X, Y)

inputShape = test_features[1,:,:].shape
model = Price_Forecast(inputShape)
model.summary()

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])

print(train_features.shape)

print(train_labels.shape)


fig = plt.figure()
axes1 = fig.add_subplot(111)
line, = axes1.plot(train_labels,np.zeros(train_labels.shape))
def corr_plt(data):
    line.set_ydata(data)
    return line,


def one_step_predict():
    model.fit(train_features, train_labels, epochs=1, batch_size=64, shuffle=True)
    model.reset_states()
    train_predict = model.predict(train_features)
    return train_predict


ani = animation.FuncAnimation(fig, corr_plt, one_step_predict, interval=2*1000)
plt.show()

