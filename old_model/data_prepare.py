# Helper module regarding to test data
import os
import pandas as pd
import datetime as dt
import numpy as np
import pickle
path = r"H:\working\data_2012_07_2012_12\data"

def all_stock_name():
    filenames = os.listdir(path)
    temp = [filename for filename in filenames if filename.endswith('.csv')]
    return [x.replace('.csv' ,'') for x in temp]


def single_stock_data(stock_name = 'SH000018'):
    data_path = os.path.join(path,stock_name+'.csv')
    df = pd.DataFrame.from_csv(data_path,header = None, parse_dates=[[0, 1]])
    df.columns = ['Open','High','Low','Close','Outstanding','Turnover']
    # header=['date','time','open','high','low','close','outstanding','turnover']
    return df


def x_y_for_single_stock(df,stock_name):
    all_dates = list(set([x.date() for x in df.index]))
    all_dates.sort()
    N = len(all_dates)-1
    X = np.zeros((N,210,6,1))
    Y = np.zeros((N,1))
    pos = []
    for i in range(N):
        d1=all_dates[i]
        d2=all_dates[i+1]

        x_start,x_end = return_x_time(d1)
        x_df = df[x_start:x_end]
        temp_x = x_df.values
        temp_x = temp_x / np.sum(temp_x, axis=0,keepdims=True)
        X[i, :, :, :] = temp_x.reshape((210, 6, 1))

        y_start, y_end = return_y_time(d1)
        y_df = df[y_start:y_end]
        Y[i,0] = y_singal(y_df)
        pos.append([stock_name,d1])

    return X,Y,pos

def return_x_time(date):
    return (dt.datetime(date.year,date.month,date.day,9,0),
            dt.datetime(date.year, date.month, date.day, 14, 30))

def y_singal(df):
    price1 = float(df[df.index == df.index.min()]['Close'])
    price2 = float(df[df.index == df.index.max()]['Close'])
    return (price2 - price1)/price1

def number_to_category(num):
    return


def return_y_time(d1):
    return (dt.datetime(d1.year,d1.month,d1.day,14,30),
            dt.datetime(d1.year, d1.month, d1.day, 14,31))

def save_test_case(X,Y,Pos):
    with open('test_case.pickle', 'wb') as fp:
        pickle.dump({"Pos":Pos,"X":X,"Y":Y}, fp)

    return


if __name__ == '__main__':
    #from tqdm import tqdm
    all_names = all_stock_name()
    sample = 20
    X = np.zeros((0,210,6,1))
    Y = np.zeros((0,1))
    Pos = []
    for name in all_names[0:sample]:
        df = single_stock_data(name)
        X_s,Y_s,pos = x_y_for_single_stock(df,name)
        X = np.concatenate((X, X_s), axis=0)
        Y = np.concatenate((Y, Y_s), axis=0)
        Pos.append(pos)

    print(X.shape)
    print(Y.shape)
    print(Pos)
    print(sum(Y))

    save_test_case(X,Y,Pos)

