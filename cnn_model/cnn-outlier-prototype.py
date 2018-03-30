# ===================================================================================
# Main functions to run and train the model
# ==================================================================================
from keras.optimizers import adam
from helper import *
from graph import *
import random


import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

# load the data
path = r"SH600036.csv"
df = get_file_from_csv(path)
# decide how many datas included in the model part
dev_set = 20000
# decide how many ticker look back before actual trades happen
lookback = 20
# decide how many ticker in total compare with the actual trades
timewindow = 30

# generate random trade heppen times
time_list = df.index.tolist()
indices = random.sample(range(lookback+1, len(time_list)-timewindow), dev_set)
# call a function in helper to generate X and Y
X,Y = x_y_new(df,indices,lookback = lookback,time_window = timewindow)

# Divide the data to training set and testing set
X_train,  X_test, Y_train_orig,Y_test_orig= divide_data(X,Y)

# Convert class vectors to binary class matrices.
num_classes = 2
Y_train = keras.utils.to_categorical(Y_train_orig.T, num_classes)
Y_test = keras.utils.to_categorical(Y_test_orig.T, num_classes)


print("number of training examples = " + str(X_train.shape[0]))
print("number of test examples = " + str(X_test.shape[0]))
print("X_train shape: " + str(X_train.shape))
print("Y_train shape: " + str(Y_train.shape))
print("X_test shape: " + str(X_test.shape))
print("Y_test shape: " + str(Y_test.shape))

# ===================================================================================
# Model definition part
# ===================================================================================
model = Sequential()

model.add(Conv2D(64, (8, 2),padding='same',input_shape=X_train.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
model.add(Conv2D(32, (8, 2),padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(4, 4),padding='same'))
model.add(Flatten())
model.add(Dense(num_classes))
model.add(Activation('softmax'))


# ===================================================================================
# Model fitting part
# ===================================================================================
opt = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

model.summary()
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')


batch_history = 1000
y_hat_train_history = []
y_hat_test_history =[]
for i in range(40):
    y_hat_train = model.predict(X_train[0:batch_history, :, :, :])
    y_hat_train_history.append(y_hat_train[:, 1])
    y_hat_test = model.predict(X_test)
    y_hat_test_history.append(y_hat_test[:, 1])

    model.fit(X_train, Y_train,
                  batch_size=100,
                  epochs=5,
                  validation_data=(X_test, Y_test),
                  shuffle=True)

y_hat_train = model.predict(X_train[0:batch_history,:,:,:])
y_hat_train_history.append(y_hat_train[:,1])
y_hat_test = model.predict(X_test)
y_hat_test_history.append(y_hat_test[:, 1])


# call a function to generate the movie
animation_train_and_test(Y_train_orig[0:batch_history],Y_test_orig,y_hat_train_history,y_hat_test_history)

#print(Y_test)
model.save(r"C:\working\innovation_group\cnn_model\final_model.h5")

