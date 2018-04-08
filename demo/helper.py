import numpy as np

def generate_x(price,mpes_obj,time):
    # decide how many ticker look back before actual trades happen
    lookback = 20
    # decide how many ticker in total compare with the actual trades
    timewindow = 30

    N = timewindow + 1
    x = np.zeros((N, 2, 1))

    for k in range(time - lookback ,time - lookback + N):
        x[k - time + lookback,0,0] = mpes_obj.get_tick(k)[1]

    x[:,1,0] = x[:,0,0]
    x[lookback,1,0] = price

    norm_x,_ = norm_ts(x)

    out = norm_x.reshape((1,N,2,1))
    return out



def norm_ts(Z):
    Z_norm = (Z - Z.min()) / (Z.max() - Z.min())
    invert_func = lambda X: (Z.max() - Z.min()) * X + Z.min()
    return Z_norm, invert_func

