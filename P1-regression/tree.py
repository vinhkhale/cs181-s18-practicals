import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.Chem import MACCSkeys

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
#merge train and data so we can engineer features
df_all = pd.concat((df_train, df_test), axis=0)
df_all = df_all[:1000] # comment out for full data set
mol_all = [Chem.MolFromSmiles(x) for x in df_all.smiles.astype(str)]
SSSR_len = np.vstack([Chem.GetSSSR(x) for x in mol_all])
print(SSSR_len)
alkyl_smarts = "[CX4]"
alkyl = Chem.MolFromSmarts(alkyl_smarts)
alkyl_count = np.vstack([len(Chem.Mol.GetSubstructMatches(x, alkyl, uniquify = True)) for x in mol_all])
print(alkyl_count)
fps = [FingerprintMols.FingerprintMol(x) for x in mol_all]
maccs = [MACCSkeys.GenMACCSKeys(x) for x in mol_all]
co2 = Chem.MolFromSmiles('C(=O)=O')
fps_co2 = FingerprintMols.FingerprintMol(co2)
fps_co2_sim = np.vstack([DataStructs.FingerprintSimilarity(x, fps_co2) for x in fps])
#fps_co2_sim = np.vstack([DataStructs.FingerprintSimilarity(x, fps_co2, metric=DataStructs.DiceSimilarity) for x in fps])
#CosineSimilarity, SokalSimilarity, RusselSimilarity, RogotGoldbergSimilarity
#AllBitSimilarity, KulczynskiSimilarity, McConnaugheySimilarity, AsymmetricSimilarity, BraunBlanquetSimilarity
print(fps_co2_sim)
maccs_co2 = MACCSkeys.GenMACCSKeys(co2)
maccs_co2_sim = np.vstack([DataStructs.FingerprintSimilarity(x, maccs_co2) for x in maccs])
#maccs_co2_sim = np.vstack([DataStructs.FingerprintSimilarity(x, maccs_co2, metric=DataStructs.DiceSimilarity) for x in maccs])
#CosineSimilarity, SokalSimilarity, RusselSimilarity, RogotGoldbergSimilarity
#AllBitSimilarity, KulczynskiSimilarity, McConnaugheySimilarity, AsymmetricSimilarity, BraunBlanquetSimilarity
print(maccs_co2_sim)
print(np.vstack(df_all.smiles.astype(str)))
#df_all['smiles_len'] = pd.DataFrame(smiles_len)


#smiles_len = np.vstack(df_all.smiles.astype(str).apply(lambda x: len(x)))
#df_all['smiles_len'] = pd.DataFrame(smiles_len)


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
