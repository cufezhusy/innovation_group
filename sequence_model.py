import numpy as np
np.random.seed(0)
from keras.models import Model
from keras.layers import Dense, Input, Dropout, LSTM, Activation
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.initializers import glorot_uniform
np.random.seed(1)
from model_helper import load_test_case, divide_data
from model_helper import corr_plt

# Model: Price_Forecast
def Price_Forecast(input_shape):
    """
    Function creating the Price_Forecast model's graph.

    Arguments:
    input_shape -- shape of the input, usually (max_len,)
    word_to_vec_map -- dictionary mapping every word in a vocabulary into its 50-dimensional vector representation
    word_to_index -- dictionary mapping from words to their indices in the vocabulary (400,001 words)

    Returns:
    model -- a model instance in Keras
    """

    ### START CODE HERE ###
    # Define embeddings as the input of the graph, it should be of shape input_shape and dtype 'int32' (as it contains indices).
    embeddings = Input(shape=input_shape, dtype=np.float32)

    print (embeddings.shape, "....")
    # Propagate the embeddings through an LSTM layer with 128-dimensional hidden state
    # Be careful, the returned output should be a batch of sequences.
    X = LSTM(128, return_sequences=True)(embeddings)
    # Add dropout with a probability of 0.5
    X = Dropout(0.5)(X)
    # Propagate X trough another LSTM layer with 128-dimensional hidden state
    # Be careful, the returned output should be a single hidden state, not a batch of sequences.
    X = LSTM(128)(X)
    # Add dropout with a probability of 0.5
    X = Dropout(0.5)(X)
    # Propagate X through a Dense layer with softmax activation to get back a batch of 5-dimensional vectors.
    X = Dense(1, activation='sigmoid')(X)
    #X = Dense(1)(X)
    # Add a softmax activation
    # X = Activation('sigmoid')(X)

    # Create Model instance which converts embeddings into X.
    model = Model(embeddings, X)

    ### END CODE HERE ###

    return model

if __name__ == '__main__':
    # load data
    data = load_test_case()

    X = data['X']
    Y = data['Y']
    pos = data['Pos']

    train_features, test_features, train_labels, test_labels = divide_data(X, Y)

    inputShape = (210, 6)
    print(inputShape)
    model = Price_Forecast(inputShape)
    model.summary()

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

    print(np.squeeze(train_features, axis=3).shape)

    print(train_labels.shape)

    model.fit(np.squeeze(train_features, axis=3), train_labels, epochs=1, batch_size=128, shuffle=True)


    loss, acc = model.evaluate(np.squeeze(test_features, axis=3), test_labels)
    print()
    print("Test accuracy = ", acc)
    predict = model.predict(np.squeeze(test_features, axis=3))
    print(predict)

    corr_plt(predict,test_labels)