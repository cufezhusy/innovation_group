import numpy as np
np.random.seed(1)
import numpy as np
import datetime as dt
import os
import pandas as pd
from model_helper import divide_data
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from keras.models import Model,Sequential,save_model,load_model
from keras.layers import Dense, Input, Dropout, LSTM, Activation


# ===================================================================================
# Data loading part
# ===================================================================================
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


# ===================================================================================
# Generate x and y
# ===================================================================================
def x_y_new(df,slice = None, start = 0):
    N = 120
    if slice == None:
        M = len(df) - N -1 - start
    else:
        M = slice

    X = np.zeros((M, N, 1))
    Y = np.zeros((M, 1))
    # get the close price from the df
    Z = df.values
    Z = Z[start:start+ M+N+1,3]

    # calculate the relative return
    #Z_diff = np.diff(Z)/Z[0:-1]
    Z_norm,invert_func  = norm_ts(Z)

    for i in range(M):
        X[i, :, 0] = Z_norm [i:i + N]
        Y[i,0] = Z_norm [i+N]

    return X, Y , invert_func


def norm_ts(Z):
    Z_norm = (Z - min(Z)) / (max(Z) - min(Z))
    invert_func = lambda X: (max(Z) - min(Z))*X + min(Z)
    return Z_norm,invert_func
# ===================================================================================
# Set up the keras model
# ===================================================================================
def Price_Forecast(input_shape):
    # create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(16,return_sequences=True,input_shape=input_shape))
    model.add(LSTM(16))
    model.add(Dense(1))
    return model

def one_step_predict(model):
    model.fit(train_features, train_labels, epochs=1, batch_size=64, shuffle=True)
    return model

# ===================================================================================
# Visualization the model
# ===================================================================================
def animation_train_and_test(train_labels,test_labels,train_predict_step,test_perdict_step):
    # Configure the training plot
    fig = plt.figure()
    axes1 = fig.add_subplot(121)
    axes2 = fig.add_subplot(122)
    line, = axes1.plot(train_labels, np.zeros(train_labels.shape), 'ro')
    axes1.set_title('Training')
    axes1.set_xlabel('Actual')
    axes1.set_ylabel('Model')
    line2, = axes2.plot(test_labels, np.zeros(test_labels.shape), 'bo')
    axes1.set_title('Testing')
    axes2.set_xlabel('Actual')
    axes2.set_ylabel('Model')
    axes1.set_ylim([min(train_labels), max(train_labels)])
    axes2.set_ylim([min(test_labels), max(test_labels)])

    def corr_plt(data):
        line.set_ydata(data)
        return line,

    def corr_plt2(data):
        line2.set_ydata(data)
        return line2,

    ani = animation.FuncAnimation(fig, corr_plt, train_predict_step, interval=1000)
    ani2 = animation.FuncAnimation(fig, corr_plt2, test_perdict_step, interval=1000)
    plt.show()


def very_simple_benchmark_model(X):
    daytimes = X.shape[1]
    x2 = np.linspace(0, daytimes - 1, num=daytimes)
    out = np.zeros((X.shape[0],1))
    for s in range(X.shape[0]):
        y1 = X[s,:,0]
        z = np.polyfit(x2, y1, 2)
        f = np.poly1d(z)
        out[s, 0] = f(daytimes)
    return out



if __name__ == '__main__':
    path = r"C:\working\data_2012_07_2012_12\data"
    all_names = all_stock_name()
    name = all_names[2]
    df = single_stock_data(name)
    dev_set = 20000
    X, Y ,invert_func = x_y_new(df,slice=dev_set)
    train_features, test_features, train_labels, test_labels = divide_data(X, Y)
    inputShape = test_features[1,:,:].shape

    try:
        model = load_model('forcast.h5')
    except:
        model = Price_Forecast(inputShape)
        model.summary()
        model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
        aa = []
        bb = []
        for i in range(10):
            model = one_step_predict(model)
            train_predict = model.predict(train_features)
            aa.append(train_predict.tolist())
            test_predict= model.predict(test_features)
            bb.append(test_predict)

        model.save('forcast.h5')

        animation_train_and_test(train_labels=train_labels,
                                 test_labels=test_labels,
                                 train_predict_step=aa,
                                 test_perdict_step=bb)

        plt.show()




    X_bench, Y_bench, invert_func_bench = x_y_new(df,slice =200,start=3000)
    Y_bench_predict = model.predict(X_bench)
    Y_bench_extrapolation = very_simple_benchmark_model(X_bench)
    plt.plot(invert_func_bench(Y_bench_predict),'r-')
    plt.plot(invert_func_bench(Y_bench_extrapolation),'g-')
    plt.plot(invert_func_bench(Y_bench),'b-')
    plt.show()