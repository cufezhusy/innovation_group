from graph import *
from helper import *
from keras import models
import keras
import random
random.seed(9001)

# get X and Y
path = r"SH600036.csv"
df = get_file_from_csv(path)
dev_set = 10000
# random time
time_list = df.index.tolist()
lookback = 20
timewindow = 30
indices = random.sample(range(lookback+1, len(time_list)-timewindow), dev_set)
X,Y = x_y_new(df,indices,lookback = lookback,time_window = timewindow)

num_classes = 2
Y_two= keras.utils.to_categorical(Y.T, num_classes)
# plot something...
idx = 67
#plot_problem(X,Y,idx=idx)

# predict
#from keras.utils import plot_model
model = models.load_model("final_model.h5")

y_hat = model.predict(X)


for i in range(dev_set):
    predict = 1 if y_hat[i, 1] > 0.5 else 0
    print("id:%s, true: %s , predict: %s" %(i,Y[i],predict))

