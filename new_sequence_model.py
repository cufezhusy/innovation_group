import numpy as np
np.random.seed(1)
import numpy as np
import datetime as dt
import os
import pandas as pd
from model_helper import divide_data

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from keras.models import Model,Sequential
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

def x_y_new(df,slice = None):
    N = 120
    if slice == None:
        M = len(df) - N -1
    else:
        M = slice

    X = np.zeros((M, N, 1))
    Y = np.zeros((M, 1))
    # get the close price from the df
    Z = df.values
    Z = Z[0:M+N+1,3]

    # calculate the relative return
    #Z_diff = np.diff(Z)/Z[0:-1]
    Z_diff = Z
    Z_diff = (Z_diff - min(Z_diff)) / (max(Z_diff) - min(Z_diff))

    for i in range(M):
        X[i, :, 0] = Z_diff[i:i + N]
        Y[i,0] = Z_diff[i+N]
        #forward = Z[i + N + 1, 3] - Z[i + N, 3]
        #now = Z[i + N, 3] - Z[i + N - 1, 3]
        #if forward * now < 0:
            #Y[i, 0] = 1

    return X, Y

# Model: Price_Forecast
def Price_Forecast(input_shape):

    # create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(4,return_sequences=True,input_shape=input_shape))
    model.add(LSTM(4))
    model.add(Dense(1))
    return model




path = r"C:\working\data_2012_07_2012_12\data"
all_names = all_stock_name()
# print all_names

Pos = []

name = all_names[2]
df = single_stock_data(name)
X, Y = x_y_new(df,slice=5000)

print(X.shape)
print(Y.shape)

train_features, test_features, train_labels, test_labels = divide_data(X, Y)

inputShape = test_features[1,:,:].shape
model = Price_Forecast(inputShape)
model.summary()

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])

print(train_features.shape)

print(train_labels.shape)


fig = plt.figure()
axes1 = fig.add_subplot(121)
axes2 = fig.add_subplot(122)
line, = axes1.plot(train_labels,np.zeros(train_labels.shape),'ro')
line2, = axes2.plot(test_labels,np.zeros(test_labels.shape),'bo')
axes1.set_ylim([min(train_labels),max(train_labels)])
axes2.set_ylim([min(test_labels),max(test_labels)])
def corr_plt(data):
    line.set_ydata(data)
    return line,

def corr_plt2(data):
    line2.set_ydata(data)
    return line2,



def one_step_predict(model):
    model.fit(train_features, train_labels, epochs=1, batch_size=64, shuffle=True)
    train_predict = model.predict(train_features)
    return train_predict,model

aa = []
bb = []
for i in range(10):
    predict,model = one_step_predict(model)
    aa.append(predict.tolist())
    test_predict= model.predict(test_features)
    bb.append(test_predict)

ani = animation.FuncAnimation(fig, corr_plt, aa,interval=1000)
ani2 = animation.FuncAnimation(fig, corr_plt2, bb,interval=1000)
plt.show()



