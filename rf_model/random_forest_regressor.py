# module random_forest_cvparam.py

# imports
from __future__ import print_function
from __future__ import division
from old_model.model_helper import divide_data, load_test_case, model_diff, hist_gram, corr_plt
from rf_model.model_analytics import model_result
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
import time

# functions
def print_shapes():
    print("Training Features Shape:", train_features.shape)
    print("Training Labels Shape:", train_labels.shape)
    print("Test Features Shape:", test_features.shape)
    print("Test Labels Shape:", test_labels.shape)

if __name__ == "__main__":
    pd.set_option('expand_frame_repr', False)

    start_time = time.time()

    # Load from common model_helper
    out = load_test_case()
    X = np.nan_to_num(out['X'])
    Y = np.nan_to_num(out['Y'])
    pos = out['Pos']

    feature_list = ['Open','High','Low','Close','Outstanding','Turnover']

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

    # Reshape test features labels from 4 dim to 2 dim
    nsamples, nx, ny, nz = test_features.shape
    d2_test_features = test_features.reshape((nsamples, nx * ny * nz))

    predictions = rf.predict(d2_test_features)

    model_helper_obj = model_result(model=rf, predictions=predictions, test_labels=test_labels,
                                    rf_feat_imp=rf.feature_importances_, feature_list=feature_list)

    model_helper_obj.save_model("rf_v1")
    print("Mean Absolute Error:",model_helper_obj.model_mean_absolute_error())
    print("Accuracy",model_helper_obj.model_accuracy())
    #model_helper_obj.plot_var_importance()

    model_diff(predictions, test_labels)
    hist_gram(predictions, test_labels)
    corr_plt(predictions, test_labels)

    print("Duration in Seconds:", (time.time() - start_time))