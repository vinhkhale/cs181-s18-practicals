import pandas as pd
import numpy as np
import cPickle as pickle
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from rdkit import Chem
from rdkit.Chem import Descriptors

print "reading data..."
df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")
print "done"

#store gap values
Y_train = df_train.gap.values
#row where testing examples start
test_idx = df_train.shape[0]
#delete 'Id' column
df_test = df_test.drop(['Id'], axis=1)
#delete 'gap' column
df_train = df_train.drop(['gap'], axis=1)

#DataFrame with all train and test examples so we can more easily apply feature engineering on
df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)
x = df_all.smiles.astype(str)[281013]
print x
#Convert Smiles to Mol
print "Starting Feature Extraction"
mol_train = []
for (i, x) in enumerate(df_train.smiles.astype(str)):
    #if i < 560000:
    #    continue
    mol = Chem.MolFromSmiles(x, sanitize=False)
    Chem.SanitizeMol(mol, sanitizeOps=Chem.SANITIZE_FINDRADICALS|Chem.SANITIZE_SETHYBRIDIZATION)
    # print "-"
    mol_train.append(mol)
    # print i
    if (i % 10000 == 0):
    	print i
with open("moldata-train", "w") as f:
    pickle.dump(mol_train, f, protocol=pickle.HIGHEST_PROTOCOL)
mol_train = []

for (i, x) in enumerate(df_test.smiles.astype(str)):
    mol = Chem.MolFromSmiles(x, sanitize=False)
    Chem.SanitizeMol(mol, sanitizeOps=Chem.SANITIZE_FINDRADICALS|CHEM.SANITIZE_SETHYBRIDIZATION)
    mol_test.append(mol)
    if (i % 10000 == 0):
        print i
with open("moldata-test", "w") as f:
    pickle.dump(mol_test, f, protocol=pickle.HIGHEST_PROTOCOL)
# mol_all = df_all.smiles.astype(str).apply(lambda x: Chem.MolFromSmiles(x))

exit()
print "A"

#Feature Engineering
feat_num = 257
descriptors = [
    Descriptors.NumHDonors,
    Descriptors.NumHAcceptors,
    Descriptors.fr_halogen
]
for descriptor in descriptors:
    feat = np.vstack(mol_all.apply(lambda x: descriptor(x)))
    df_all['feat_' + str(feat_num)] = pd.DataFrame(feat)
    feat_num += 1

print "B"

#Drop the 'smiles' column
df_all = df_all.drop(['smiles'], axis=1)
vals = df_all.values
X_train = vals[:test_idx]
X_test = vals[test_idx:]

print "C"

RF = RandomForestRegressor()
RF.fit(X_train, Y_train)
RF_pred = RF.predict(X_test)

print "D"

def write_to_file(filename, predictions):
    with open(filename, "w") as f:
        f.write("Id,Prediction\n")
        for i,p in enumerate(predictions):
            f.write(str(i+1) + "," + str(p) + "\n")

write_to_file("pred.csv", RF_pred)
