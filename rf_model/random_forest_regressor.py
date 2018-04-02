# module random_forest_cvparam.py

# imports
from __future__ import print_function
from __future__ import division
from old_model.model_helper import divide_data, load_test_case
from rf_model.model_analytics import model_result
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

    # Instantiate model with n decision trees
    rf = RandomForestRegressor(n_jobs=2, max_features='auto', n_estimators=1000, oob_score=True, random_state=42,
                                max_depth=10, min_samples_leaf=40)

    rf.fit(d2_train_features, d2_train_labels)

    predictions = rf.predict(test_features)
    # print(predictions)

    model_helper_obj = model_result(model=rf, predictions=predictions, test_labels=test_labels,
                                    rf_feat_imp=rf.feature_importances_)

    model_helper_obj.save_model("rf_v1")
    model_helper_obj.model_absolute_errors()
    model_helper_obj.model_accuracy()
    model_helper_obj.plot_var_importance()

    print("Duration in Seconds:", (time.time() - start_time))