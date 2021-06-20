import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class Config:
    # read the database
    data1 = pd.read_csv('dataset/Admission_Predict.csv')
    data2 = pd.read_csv('dataset/Admission_Predict_Ver1.1.csv')

    admission_df= data1
    admission_df.drop('Serial No.', axis=1, inplace=True)

    # split to train and test
    x = admission_df.drop(columns=['Chance of Admit','University Rating'])
    y = admission_df['University Rating']
    x = np.array(x)
    y = np.array(y)
    y = y.reshape(-1, 1)

    scaler_x = StandardScaler()
    x = scaler_x.fit_transform(x)

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
