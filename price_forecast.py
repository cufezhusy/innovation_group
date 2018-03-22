# module price_forecast.py

# imports
from __future__ import print_function
from __future__ import division

from model_helper import load_test_case, divide_data
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import export_graphviz
import pandas as pd
import random
import datetime
import os
import matplotlib.pyplot as plt
import numpy as np
import math

# functions

def print_shapes():
    print("Training Features Shape:", train_features.shape)
    print("Training Labels Shape:", train_labels.shape)
    print("Test Features Shape:", test_features.shape)
    print("Test Labels Shape:", test_labels.shape)

if __name__ == "__main__":
    pd.set_option('expand_frame_repr', False)

    # Load from common model_helper
    out = load_test_case()
    X = out['X']
    Y = out['Y']
    pos = out['Pos']

    # Saving feature names for later use
    #feature_list = list(features.columns)

    # Split the data into training and testing sets common model_helper
    train_features, test_features, train_labels, test_labels = divide_data(X, Y)
    #print_shapes()

    # Reshape train features from 4 dim to 2 dim
    nsamples, nx, ny, nz = train_features.shape
    d2_train_features = train_features.reshape((nsamples, nx * ny * nz))

    # Instantiate model with 1000 decision trees, set n_jobs = -1 = number of cores
    rf = RandomForestRegressor(n_estimators=1000, n_jobs=-1, random_state=42)

    # Train the model on training data
    #rf.fit(train_features, train_labels)
    rf.fit(d2_train_features, train_labels)

    # Reshape test features from 4 dim to 2 dim
    nsamples, nx, ny, nz = test_features.shape
    d2_test_features = test_features.reshape((nsamples, nx * ny * nz))

    # Make predictions with the model by using the forest's predict method on the test data
    predictions = rf.predict(d2_test_features)
    #print(predictions)

    # Calculate the absolute erros
    errors = abs(predictions - test_labels)
    #print("Errors f(pred - test):", errors)

    # Print out the mean absolute error (mae)
    #print("Mean Absolute Error:", round(np.mean(errors), 2), "degrees.")

    # Calculate mean absolute percentage error (MAPE)
    mape = 100 * (errors / test_labels)
    #print(mape)

    # Calculate and display accuracy
    accuracy = 100 - np.mean(mape)
    #print("Accuracy:", round(accuracy,2), "%.")

    # Get names and current values for all parameters given Estimator
    for key in rf.get_params():
        pass
        #print(key, ":", rf.get_params()[key])
    #print(rf.get_params())

    # Visualize a single decision tree using pydot
    #tree = rf.estimators_[5]

    # Export the timage to a dot file
    #export_graphviz(tree, out_file='tree.dot', feature_names=feature_list, rounded=True,precision=1)

    # Use dot file to create a graph
    #(graph, ) = pydot.graph_from_dot_file('tree.dot')

    # Write graph to a png file
    #graph.write_png('tree.png')

    # Get numerical feature importances
    importances = list(rf.feature_importances_)

    # List of tuples with variable and importance
    feature_importance = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]

    # Sort the feature importances by most important first
    feature_importance = sorted(feature_importance, key = lambda x: x[1], reverse = True)

    # Print out the feature and importances
    [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importance];

    #plot_var_importance()
    #plot_predicted_val()
