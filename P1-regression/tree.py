import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
#from rdkit import Chem

print("xd")
df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")

print("ok")


#store gap values
Y_train = df_train.gap.values
#row where testing examples start
test_idx = df_train.shape[0]
#delete 'Id' column
df_test = df_test.drop(['Id'], axis=1)
#delete 'gap' column
df_train = df_train.drop(['gap'], axis=1)

print("ok")
#idk what this line of code does
df_all = pd.concat((df_train, df_test), axis=0)


## feature engineering here
## just spam that rdk shit


#df_all = df_all.drop(['smiles'], axis=1)
#vals = df_all.values
#X_train = vals[:test_idx]
#X_test = vals[test_idx:]

#RF = RandomForestRegressor()
#RF.fit(X_train, Y_train)
#RF_pred = RF.predict(X_test)

#def write_to_file(filename, predictions):
#    with open(filename, "w") as f:
#        f.write("Id,Prediction\n")
#        for i,p in enumerate(predictions):
#            f.write(str(i+1) + "," + str(p) + "\n")


#write_to_file("sample2.csv", RF_pred)
