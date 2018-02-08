import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
import gc
import cPickle as pickle
from multiprocessing import Pool
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")

#store gap values
Y_train = df_train.gap.values
#row where testing examples start
test_idx = df_train.shape[0]
#delete 'Id' column
df_test = df_test.drop(['Id'], axis=1)
#delete 'gap' column
df_train = df_train.drop(['gap'], axis=1)

#DataFrame with all train and test examples so we can more easily apply feature engineering on
df_all = pd.concat((df_train, df_test), axis=0)

print "done reading"

def process_df(df):
    mol_all = df.smiles.astype(str).apply(lambda x: Chem.MolFromSmiles(x, sanitize = False))
    morgan = np.empty([len(mol_all), 256], dtype=np.int8)
    for (i, mol) in enumerate(mol_all):
        Chem.SanitizeMol(mol,sanitizeOps=Chem.SANITIZE_SYMMRINGS|Chem.SANITIZE_KEKULIZE|Chem.SANITIZE_SETHYBRIDIZATION)
        morgan[i] = np.array(AllChem.GetMorganFingerprintAsBitVect(mol,2,nBits=256),dtype=np.int8)
    print "split done"
    return morgan

# df_all = df_all[:144]

df_split = np.array_split(df_all, 144)
pool = Pool()
morgan = pool.map(process_df, df_split)
pool.close()
pool.join()
#morgan = morgan.reshape(len(morgan) * 72, 256)
print "stacking"
morgan = np.vstack(morgan)
print "done"
#print morgan
#print len(morgan)
#exit()
        
#mol_all = df_all.smiles.astype(str).apply(lambda x: Chem.MolFromSmiles(x, sanitize = False))
#for mol in mol_all:
#    Chem.SanitizeMol(mol,sanitizeOps=Chem.SANITIZE_SYMMRINGS|Chem.SANITIZE_KEKULIZE|Chem.SANITIZE_SETHYBRIDIZATION)
#print("done sanitizing")
#morgan = [AllChem.GetMorganFingerprintAsBitVect(x,2,nBits=256) for x in mol_all]


# TRISTAN YANG- IF THIS LINE DOESNT WORK THEN USE THE for i in range(2048) guy
print "converting to dataframe"
df_all = pd.DataFrame(morgan,dtype=np.int8)
print "done"

#for i in range(2048):
 #   morgVec = np.vstack(morgan[x][i] for x in range(len(mol_all)))
  #  df_all['morgVec_'+str(i)] = pd.DataFrame(morgVec, dtype = np.int8)
   # print("good")



#for term in func_group_list:
 #   smarts = Chem.MolFromSmarts(term)
  #  feats = []
 #   for mol in mol_all:
 #       feats.append(len(Chem.Mol.GetSubstructMatches(mol, smarts, uniquify = True)))
    #feats_vstack = np.vstack(feats)
 #   df_all['feat_'+str(feat_num)] = pd.DataFrame(np.vstack(feats))
 #   feat_num += 1
    #with open(str(feat_num) + "-data", "w") as f:
    #    pickle.dump(feats_vstack, f, pickle.HIGHEST_PROTOCOL)
 #   print "processed"             



print "Done reading feats"

#Drop the 'smiles' column
#df_all = df_all.drop(['smiles'], axis=1)
vals = df_all.values
X_train = vals[:test_idx]
X_test = vals[test_idx:]

RF = RandomForestRegressor(n_jobs=-1)
RF.fit(X_train, Y_train)
RF_pred = RF.predict(X_test)

def write_to_file(filename, predictions):
    with open(filename, "w") as f:
        f.write("Id,Prediction\n")
        for i,p in enumerate(predictions):
            f.write(str(i+1) + "," + str(p) + "\n")

write_to_file("pred.csv", RF_pred)
