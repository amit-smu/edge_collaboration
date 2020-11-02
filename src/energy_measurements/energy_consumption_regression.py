"""
module for estimating regression models for energy consumption based on number of pixels scanned
(for image capture, storage & network transmission).
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score, KFold
import pickle

if __name__ == "__main__":
    DEGREE = 4
    in_file_path = "img_nw_trans_measurements_rpi.csv"
    output_filename = "img_nw_trans_energy_model_rpi"

    data = pd.read_csv(in_file_path, delimiter="\t")
    # print(data)
    X = data['#pixels'].values
    X = np.reshape(X, (-1, 1))
    Y = data['net_energy'].values
    # print(X)

    # apply poly degree
    poly_features = PolynomialFeatures(degree=DEGREE, interaction_only=False)
    reg_model = LinearRegression(normalize=False)
    X_poly = poly_features.fit_transform(X=X)
    print(X_poly)

    splits = 5
    r_state = 30  # use either 0 or 30
    kfold = KFold(n_splits=splits, shuffle=True, random_state=r_state)
    final_model = None
    best_r2 = -1
    for train, test in kfold.split(X_poly):
        reg_model.fit(X=X_poly[train], y=Y[train])
        # training accuracy
        prediction = reg_model.predict(X=X_poly[train])
        train_r2_score = r2_score(y_true=Y[train], y_pred=prediction)
        # test accuracy
        prediction = reg_model.predict(X=X_poly[test])
        test_r2_score = r2_score(y_true=Y[test], y_pred=prediction)
        print("training accuracy : {}, test accuracy: {}".format(train_r2_score, test_r2_score))
        if test_r2_score > best_r2:
            best_r2 = test_r2_score
            final_model = reg_model

    # print(best_r2)
    score = cross_val_score(reg_model, X=X_poly, y=Y, cv=kfold)
    print("cross-validation scores: {}, avg : {}".format(score, np.mean(score)))
    # write model to disk

    with open(output_filename, 'wb') as out:
        # test prediction
        input_values = np.array([[414720]])
        # input_values = np.array([[2073600]])
        input_values = poly_features.transform(input_values)
        print(final_model.predict(input_values))
        pickle.dump(final_model, out)

    with open(output_filename, 'rb') as inp:
        model = pickle.load(inp)
        input_values = np.array([[124728]])
        # input_values = np.array([[142411]])
        input_values = poly_features.transform(input_values)
        print(final_model.predict(input_values))

        input_values = np.array([[960000]])
        # input_values = np.array([[99218]])
        input_values = poly_features.transform(input_values)
        print(final_model.predict(input_values))
