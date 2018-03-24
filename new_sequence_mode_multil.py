import numpy as np
np.random.seed(1)
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from keras.models import Model,Sequential,save_model,load_model
from keras.layers import Dense, Input, Dropout, LSTM, Activation
from sklearn.model_selection import train_test_split

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

def divide_data(X,Y):
    return train_test_split(X, Y,test_size = 0.05, random_state = 42)

# ===================================================================================
# Generate x and y
# ===================================================================================
def x_y_new(df,slice = None, start = 0):
    N = 30
    if slice == None:
        M = len(df) - N -1 - start
    else:
        M = slice

    X = np.zeros((M, N, 6))
    Y = np.zeros((M, 2))
    # get the close price from the df
    Z = df.values
    Z = Z[start:start+ M + N,:]

    # calculate the relative return
    #Z_diff = np.diff(Z)/Z[0:-1]
    Z_norm,invert_func_y = norm_ts(Z)

    for i in range(M):
        X[i, :, :] = Z_norm [i:i + N]

    Y[:, 0] = Z_norm[N:N+M,1]  # 'high price'
    Y[:, 1] = Z_norm[N:N+M,2]  # 'low price'
    #Y[:, 1], invert_func_spread = norm_ts((Z[N:N + M, 1] - Z[N:N + M, 2])/ Z[N:N + M, 3])  # 'close price'

    return X, Y , invert_func_y


def norm_ts(Z):
    Z_norm = (Z - Z.min(axis=0)) / (Z.max(axis=0) - Z.min(axis=0))
    invert_func = lambda X: (Z.max(axis=0) - Z.min(axis=0))*X + Z.min(axis=0)
    return Z_norm,invert_func

# ===================================================================================
# Set up the keras model
# ===================================================================================
def Price_Forecast(input_shape):
    # create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(4, input_shape=input_shape))
    #model.add(LSTM(4))
    model.add(Dense(2))
    model.add(Activation('sigmoid'))
    return model

def one_step_predict(model,train_features, train_labels):
    model.fit(train_features, train_labels, epochs=1, batch_size=64, shuffle=True)
    return model

# ===================================================================================
# Visualization the model
# ===================================================================================
def animation_train_and_test(train_labels, test_labels, predict_spread_1, predict_spread_2):
    # Configure the training plot
    fig = plt.figure()
    axes1 = fig.add_subplot(121)
    axes2 = fig.add_subplot(122)
    line, = axes1.plot(train_labels, np.zeros(train_labels.shape), 'ro')
    axes1.set_title('Training')
    axes1.set_xlabel('Actual')
    axes1.set_ylabel('Model')
    line2, = axes2.plot(test_labels, np.zeros(test_labels.shape), 'bo')
    axes2.set_title('Spread')
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

    ani = animation.FuncAnimation(fig, corr_plt, predict_spread_1, interval=1000)
    ani2 = animation.FuncAnimation(fig, corr_plt2, predict_spread_2, interval=1000)
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
    dev_set = 2000
    X, Y, invert_func_y = x_y_new(df,slice=dev_set)
    train_features, test_features, train_labels, test_labels = divide_data(X, Y)
    inputShape = test_features[1,:,:].shape

    try:
        model = load_model('forcast_3d.h5')
    except:
        model = Price_Forecast(inputShape)
        model.summary()
        model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
        aa = []
        bb = []

        idx = 0
        for i in range(20):
            model = one_step_predict(model,train_features, train_labels)
            train_predict = model.predict(train_features)
            aa.append(train_predict[:,0])
            bb.append(train_predict[:,1])

        model.save('forcast_3d.h5')

        animation_train_and_test(train_labels=train_labels[:,0],
                                 test_labels=train_labels[:,1],
                                 predict_spread_1=aa,
                                 predict_spread_2=bb)

        plt.show()




    X_bench, Y_bench, invert_func_bench = x_y_new(df,slice =200,start=3000)
    Y_bench_predict = model.predict(X_bench)
    #Y_bench_extrapolation = very_simple_benchmark_model(X_bench)
    plt.plot(invert_func_bench(Y_bench_predict[:,1]),'r-')
    #plt.plot(invert_func_bench(Y_bench_extrapolation),'g-')
    plt.plot(invert_func_bench(Y_bench[:,1]),'b-')
    plt.show()