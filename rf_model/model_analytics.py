# module model_analytics.py

# imports
from __future__ import print_function
import pickle
import numpy as np
import matplotlib.pyplot as plt

class model_result(object):
    def __init__(self, model, predictions, test_labels, rf_feat_imp, feature_list):
        self.model = model
        self.predictions = predictions
        self.test_labels = test_labels
        self.rf_feat_imp = rf_feat_imp
        self.feature_list = feature_list

    def get_model(self):
        return self.model

    def get_predictions(self):
        return self.predictions

    def get_test_labels(self):
        return self.test_labels

    def get_feature_list(self):
        return self.feature_list

    def get_feat_imp(self):
        return self.rf_feat_imp

    def save_model(self,file_name):
        try:
            pickle.dump(self.get_model(),open(file_name + ".pickle",'wb'))
            print ("Model successfully saved.")
        except Exception as e:
            print ("Model not saved", e)

    def model_absolute_errors(self):
        # Calculate the absolute errors
        return abs(self.get_predictions() - self.get_test_labels())

    def model_mean_absolute_error(self):
        # Return the mean absolute error (mae)
        return round(np.mean(self.model_absolute_errors()), 4)

    def model_mean_absolute_percentage_error(self):
        # Calculate mean absolute percentage error (MAPE)
        return 100 * (self.model_absolute_errors() / self.get_test_labels())

    def model_accuracy(self):
        # Calculate and display accuracy
        return 100 - np.mean(self.model_absolute_errors())

    def model_importances(self):
        return list(self.get_feat_imp())

    def model_feature_importance(self):
        # List of tuples with variable and importance
        imp_tmp = self.model_importances()
        feat_list_tmp = self.get_feature_list()
        feature_importance = [(feature, round(imp_tmp, 2)) for feature, imp_tmp in zip(feat_list_tmp, imp_tmp)]

        # Sort the feature importances by most important first
        feature_importance = sorted(feature_importance, key=lambda x: x[1], reverse=True)

        # Print out the feature and importances
        [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importance]

    def plot_var_importance(self):
        # Set the style
        plt.style.use('fivethirtyeight')
        # list of x locations for plotting
        x_values = list(range(len(self.model_importances())))
        # Make a bar chart
        plt.bar(x_values, self.model_importances(), orientation='vertical')
        # Tick labels for x axis
        plt.xticks(x_values, self.get_feature_list(), rotation='vertical')
        # Axis labels and title
        plt.ylabel('Importance')
        plt.xlabel('Variable')
        plt.title('Variable Importances')
        plt.show()

def load_model(file_name):
    try:
        return pickle.load(open(file_name,'rb'))
        print("Model succesfully loaded.")
    except Exception as e:
        print ("Model not loaded", e)