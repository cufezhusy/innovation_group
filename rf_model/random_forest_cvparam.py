# module random_forest_cvparam.py

# imports
from __future__ import print_function
from __future__ import division
from old_model.model_helper import divide_data, load_test_case
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
import time
import csv

# functions
def print_shapes():
    print("Training Features Shape:", train_features.shape)
    print("Training Labels Shape:", train_labels.shape)
    print("Test Features Shape:", test_features.shape)
    print("Test Labels Shape:", test_labels.shape)

def save_cv_param(param_grid):
    w = csv.writer(open("cv_param.csv", "w"))
    for key, val in param_grid.items():
        w.writerow([key, val])
    print("Parameters successfully saved.")


if __name__ == "__main__":
    pd.set_option('expand_frame_repr', False)

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

    # Instantiate model with 1000 decision trees, set n_jobs = -1 = number of cores for Cross Validation optimization
    rfc = RandomForestRegressor(n_jobs=-1, max_features='sqrt', n_estimators=1000, oob_score=True, random_state=50,
                                max_depth=7, min_samples_leaf=50)

    # Values to test CV-optimize
    param_grid = {
        'n_estimators': [800, 1000, 1200],
        'max_features': ['auto', 'sqrt', 'log2'],
        'random_state': [42, 50],
        'max_depth': [5, 7, 10],
        'min_samples_leaf': [40, 50, 60]
    }

    # Run cv-optimizer baed on grid and fit for
    CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    CV_rfc.fit(d2_train_features, d2_train_labels)

    save_cv_param(CV_rfc.best_params_)

    print("Duration in Seconds:", (time.time() - start_time))