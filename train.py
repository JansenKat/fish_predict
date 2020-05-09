import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from joblib import dump
from preprocess import prep_data

df = pd.read_csv("fish_participant.csv")

X, y = prep_data(df)

rf = RandomForestRegressor()
rf.fit(X,y)

dump(rf, "rf.joblib")