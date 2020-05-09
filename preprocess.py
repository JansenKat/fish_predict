import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import PolynomialFeatures

def prep_data(df):
    
    label_encoder = LabelEncoder()
    spe = label_encoder.fit_transform(df['Species'])
    df['Species'] = spe

    df = df.assign(lw=df['Length3']*df['Width'])\
    .assign(lhw=df['Length3']+df['Height']+df['Width'])\
    .assign(ratio=(df['Length3']/df['Length2'])*df['Width'])

    X = df[['Species','Height','Length2','Width','lhw','ratio']].values
    y = df['Weight'].values
    
    poly_features = PolynomialFeatures(degree=2)

    return X, y
