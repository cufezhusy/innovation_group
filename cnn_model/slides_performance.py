
from helper import *
from graph import *
import random
import keras
random.seed(9001)

path = r"SH600036.csv"
df = get_file_from_csv(path)
dev_set = 20000
# random time
time_list = df.index.tolist()
lookback = 20
timewindow = 30
indices = random.sample(range(lookback+1, len(time_list)-timewindow), dev_set)

X,Y = x_y_new(df,indices,lookback = lookback,time_window = timewindow)

# Loading the data (signs)
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
model = keras.models.load_model("final_model.h5")


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

for i in range(2):

    model.fit(X_train, Y_train,
                  batch_size=100,
                  epochs=5,
                  validation_data=(X_test, Y_test),
                  shuffle=True)

model.save(r"C:\working\innovation_group\cnn_model\final_model.h5")
Y_hat_train = model.predict(X_train)
Y_hat_test = model.predict(X_test)

get_heat_map(Y_train,Y_hat_train)


get_heat_map(Y_test,Y_hat_test)
