### YOU WRITE THIS ###
from joblib import load
from preprocess import prep_data
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
from sklearn.metrics import mean_squared_error

from train import poly_model, poly_features

def predict_from_csv(path_to_csv):

    df = pd.read_csv(path_to_csv)
    X, y = prep_data(df)

    X_poly = poly_features.fit_transform(X)

    predictions = poly_model.predict(X_poly)

    return mean_squared_error(y,predictions)

if __name__ == "__main__":
    predictions = predict_from_csv("fish_participant.csv")
    print(predictions)
    
######

### WE WRITE THIS ###
#     from sklearn.metrics import mean_squared_error
#     ho_predictions = predict_from_csv("fish_holdout.csv")
#     ho_truth = pd.read_csv("fish_holdout.csv")["Weight"].values
#     ho_mse = mean_squared_error(ho_truth, ho_predictions)
#     print(ho_mse)
######

