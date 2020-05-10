import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from joblib import dump
from preprocess import prep_data

df = pd.read_csv("fish_participant.csv")

X, y = prep_data(df)

gbr = GradientBoostingRegressor()
gbr.fit(X,y)

dump(gbr, "gbr.joblib")