# encoding: utf-8
# module price_forecast.py
# https://machinelearningmastery.com/handle-missing-data-python/

# imports
from __future__ import print_function
from __future__ import division

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import export_graphviz
#import pydot
import pandas as pd
import random
import time
import datetime
import os
import matplotlib.pyplot as plt
import numpy as np
import math

# path fix
#os.environ["PATH"] += os.pathsep + "C:\\Program Files (x86)\\Graphviz2.38\\bin\\"

# functions

def plot_var_importance():
    # Set the style
    plt.style.use('fivethirtyeight')
    # list of x locations for plotting
    x_values = list(range(len(importances)))
    # Make a bar chart
    plt.bar(x_values, importances, orientation='vertical')
    # Tick labels for x axis
    plt.xticks(x_values, feature_list, rotation='vertical')
    # Axis labels and title
    plt.xlabel('Variable')
    plt.title('Variable Importances')
    plt.show()

def plot_predicted_val():
    # Dates of training values
    months = features[:, feature_list.index('month')]
    days = features[:, feature_list.index('day')]
    years = features[:, feature_list.index('year')]
    # List and then convert to datetime object
    dates = [str(int(year)) + '-' + str(int(month)) + '-' + str(int(day)) for year, month, day in
             zip(years, months, days)]
    dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in dates]
    # Dataframe with true values and dates
    true_data = pd.DataFrame(data={'date': dates, 'actual': labels})
    # Dates of predictions
    months = test_features[:, feature_list.index('month')]
    days = test_features[:, feature_list.index('day')]
    years = test_features[:, feature_list.index('year')]
    # Column of dates
    test_dates = [str(int(year)) + '-' + str(int(month)) + '-' + str(int(day)) for year, month, day in
                  zip(years, months, days)]
    # Convert to datetime objects
    test_dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in test_dates]
    # Dataframe with predictions and dates
    predictions_data = pd.DataFrame(data={'date': test_dates, 'prediction': predictions})
    # Plot the actual values
    plt.plot(true_data['date'], true_data['actual'], 'b-', label='actual')
    # Plot the predicted values
    plt.plot(predictions_data['date'], predictions_data['prediction'], 'ro', label='prediction')
    plt.xticks(rotation='60')
    plt.legend()
    # Graph labels
    plt.xlabel('Date')
    plt.ylabel('Maximum Temperature (F)')
    plt.title('Actual and Predicted Values')
    plt.show()

if __name__ == "__main__":
    #start_time = time.time()
    pd.set_option('expand_frame_repr', False)

    features = pd.read_csv("C:\\Users\\g47193\\PycharmProjects\\MachineLearning\\data\\aapl_1.csv")

    #print(features[:5])


    print("The shape of our features is:", features.shape)

    # Descriptive statistics for each column
    #print(features.describe())
    #plt.plot(features['temp_2'])
    #plt.show()

    # One-hot encode the data using pandas get_dummies
    #features = pd.get_dummies(features)

    # Display the first rows of the last 12 columns
    #print(features.iloc[:,5:][:5])

    # Use numpy to convert to arrays
    # Labels are the values we want to predict
    labels = np.array(features['Close'])
    #labels = np.array(features['actual'])
    #print(labels)

    # Remove the labels from the features
    # axis 1 refers to the columns
    #features = features.drop(['Close'], axis=1)
    features = features.drop(['Close'], axis=1)
    features = features.drop(['Open'], axis=1)
    features = features.drop(['Return'], axis=1)
    features = features.drop(['Year'], axis=1)
    features = features.drop(['Month'], axis=1)
    features = features.drop(['Day'], axis=1)

    # Saving feature names for later use
    feature_list = list(features.columns)

    # Convert to numpy array
    features = np.array(features)

    # Split the data into training and testing sets Using Skicit-learn
    train_features, test_features, train_labels, test_labels = \
        train_test_split(features, labels, test_size=0.25, random_state = 42)

    #print("Training Features Shape:", train_features.shape)
    #print("Training Labels Shape:", train_labels.shape)
    #print("Test Features Shape:", test_features.shape)
    #print("Test Labels Shape:", test_labels.shape)

    # https://towardsdatascience.com/random-forest-in-python-24d0893d51c0
    # The baseline predictions are the historical averages
    # baseline_preds = test_features[:,feature_list.index('average')]

    # Baseline errors, and display average baseline error
    # baseline_errors = abs(baseline_preds - test_labels)

    # print("Average baseline error: ", round(np.mean(baseline_errors), 2), "degrees.")

    # Instantiate model with 1000 decision trees, set n_jobs = -1 = number of cores
    rf = RandomForestRegressor(n_estimators=1000, n_jobs=-1, random_state=42)

    # Train the model on training data
    rf.fit(train_features, train_labels)

    # Make predictions with the model by using the forest's predict method on the test data
    predictions = rf.predict(test_features)
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

    #print("Duration in Seconds:", (time.time() - start_time))