import numpy as np
from sklearn.preprocessing import LabelEncoder

def prep_data(df):
    
    label_encoder = LabelEncoder()
    spe = label_encoder.fit_transform(df['Species'])
    df['Species'] = spe

    df = df.assign(lw=df['Length3']*df['Width'])\
        .assign(lhw=df['Length3']+df['Height']+df['Width'])\
        .assign(ratio=(df['Length3']/df['Length2'])*df['Width'])\
        .assign(v=4/3*np.pi*df['Height']/2*df['Width']/2*df['Length3']/2)

    X = df[['Species','Height','Width','Length1','Length2','Length3','lhw','ratio','lw','v']]
    y = df['Weight'].values

    return X, y
