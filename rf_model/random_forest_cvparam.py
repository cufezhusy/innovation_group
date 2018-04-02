# module random_forest_cvparam.py

# imports
from __future__ import print_function
from __future__ import division
from old_model.model_helper import divide_data, load_test_case
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import numpy as np
import time
import csv

# functions
def print_shapes():
    print("Training Features Shape:", train_features.shape)
    print("Training Labels Shape:", train_labels.shape)
    print("Test Features Shape:", test_features.shape)
    print("Test Labels Shape:", test_labels.shape)

def save_cv_param(param_grid, model):
    w = csv.writer(open("cv_param_" + model + ".csv", "w"))
    for key, val in param_grid.items():
        w.writerow([key, val])
    print("Parameters successfully saved.")

def generate_hyper_parameter(model_type, num_est, train_feat, train_lab):
    if model_type == "rf":
        # Instantiate model with num_est decision trees
        rfc = RandomForestRegressor(n_estimators=num_est)
        # Values to test CV-optimize Regressor fully grown tree
        param_grid = {
            'max_features': ['auto', 'sqrt', 'log2'],
            'max_depth': [5, 7, 10],
            'min_samples_leaf': [40, 50, 60]
        }
    elif model_type == "gb":
        # Instantiate model with num_est decision trees
        rfc = GradientBoostingRegressor(n_estimators=num_est)
        # Values to test CV-optimize Gradient-Boosting weak learner
        param_grid = {
            'learning_rate': [0.01, 0.001, 0.0001],
            'max_features': [0.1, 'auto', 'sqrt'],
            'max_depth': [4, 5, 6],
            'min_samples_leaf': [3, 5, 7]
        }
    else:
        "Print no such model is defined for this job."

    CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, n_jobs=4).fit(train_feat, train_lab)
    save_cv_param(CV_rfc.best_params_, model_type)

    print("Job done for " + model_type + ".")

if __name__ == "__main__":
    start_time = time.time()

    # Load from common model_helper
    out = load_test_case()
    X = np.nan_to_num(out['X'])
    Y = np.nan_to_num(out['Y'])
    pos = out['Pos']

    # Split the data into training and testing sets common model_helper
    train_features, test_features, train_labels, test_labels = divide_data(X, Y)

    # Reshape train features from 4 dim to 2 dim
    nsamples, nx, ny, nz = train_features.shape
    d2_train_features = train_features.reshape((nsamples, nx * ny * nz))

    # Reshape train labels from 2 dim to 1 dim
    nsamples, nx = train_labels.shape
    d2_train_labels = train_labels.reshape((nsamples * nx))

    # Call generate_hyper_parameter to search 'initial best' hyper parameters
    generate_hyper_parameter(model_type="rf", num_est=700, train_feat=d2_train_features, train_lab=d2_train_labels)
    generate_hyper_parameter(model_type="gb", num_est=700, train_feat=d2_train_features, train_lab=d2_train_labels)

    print("Duration in Seconds:", (time.time() - start_time))