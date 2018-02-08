import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
import gc
import cPickle as pickle
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from rdkit import Chem
from rdkit.Chem import Descriptors

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

#Feature Engineering
feat_num = 257
descriptors = [
    Descriptors.NumHDonors,
    Descriptors.NumHAcceptors,
    Descriptors.fr_halogen
]

for descriptor in descriptors:
   # i = 0
   # feats = []
   # for smiles in df_all.smiles.astype(str):
   #     mol = Chem.MolFromSmiles(smiles,sanitize=False)
   #     Chem.SanitizeMol(mol,sanitizeOps=Chem.SANITIZE_FINDRADICALS|Chem.SANITIZE_SETHYBRIDIZATION)
   #     feat = descriptor(mol)
   #     feats.append(feat)
   #     if i % 10000 == 0:
   #         gc.collect()
   #         print i
   #     i += 1
   # feats_vstack = np.vstack(feats)
   # with open(str(feat_num) + "-data", "w") as f:
   #     pickle.dump(feats_vstack, f, pickle.HIGHEST_PROTOCOL)
    with open(str(feat_num) + "-data", "r") as f:
        feats_vstack = pickle.load(f)
    df_all['feat_' + str(feat_num)] = pd.DataFrame(feats_vstack)
    feat_num += 1


func_group_list = ["[CX4]", "[$([CX2](=C)=C)]", "[$([CX3]=[CX3])]","[CX3]=[OX1]",
                   "[OX1]=CN","[CX3](=[OX1])[F,Cl,Br,I]",
                   "[CX3H1](=O)[#6]","[CX3](=[OX1])(O)O","[CX3](=O)[OX2H1]","[#6][CX3](=O)[#6]",
                   "[CX3](=O)[OX1H0-,OX2H1]","[OD2]([#6])[#6]", "[H]", "[H+]", "[+H]",
                   "[$(*-[NX2-]-[NX2+]#[NX1]),$(*-[NX2]=[NX2+]=[NX1-])]",
                   "[$([NX3](=[OX1])(=[OX1])O),$([NX3+]([OX1-])(=[OX1])O)] ","[OX2H]","[#6][OX2H]",
                   "[OX2,OX1-][OX2,OX1-]","[#16X2H]", "[SX2]", "[$([#16X4](=[OX1])(=[OX1])([OX2H,OX1H0-])[OX2][#6]),$([#16X4+2]([OX1-])([OX1-])([OX2H,OX1H0-])[OX2][#6])]",
                   "[#6][F,Cl,Br,I]","[F,Cl,Br,I]", "[F,Cl,Br,I].[F,Cl,Br,I].[F,Cl,Br,I]","[CX3](=[OX1])[F,Cl,Br,I]","[$([cX2+](:*):*)]",
                   "[$([cX3](:*):*),$([cX2+](:*):*)] ","[$([cX3](:*):*),$([cX2+](:*):*),$([CX3]=*),$([CX2+]=*)] ",
                   "[$([nX3](:*):*),$([nX2](:*):*),$([#7X2]=*),$([NX3](=*)=*),$([#7X3+](-*)=*),$([#7X3+H]=*)]",
                   "[$([NX4+]),$([NX3]);!$(*=*)&!$(*:*)]","[$([#1X1][$([NX4+]),$([NX3]);!$(*=*)&!$(*:*)])]","[$([SX3]=N)]", "[$([NX1]#*)]",
                   "[R0;D2][R0;D2][R0;D2][R0;D2]","[cR1]1[cR1][cR1][cR1][cR1][cR1]1","[sX2r5]","*/,\[R]=;@[R]/,\*","[cR1]1[cR1][cR1][cR1][cR1][cR1]1.[cR1]1[cR1][cR1][cR1][cR1][cR1]1",
                   "c12ccccc1cccc2","[!H0;F,Cl,Br,I,N+,$([OH]-*=[!#6]),+]","[CX3](=O)[OX2H1]","[$([OH]-*=[!#6])]",
                   "[CX3](=[OX1])[F,Cl,Br,I]", "[NX2-]", "[$([cX2+](:*):*)]",  "[+1]~*~*~[-1]",
                   "[#6,#7;R0]=[#8]", "[!$([#6,H0,-,-2,-3])]", "[!H0;#7,#8,#9]", "[O,N;!H0]-*~*-*=[$([C,N;R0]=O)]", "[#6;X3v3+0]", "[#7;X2v4+0]"]

for term in func_group_list:
    smarts = Chem.MolFromSmarts(term)
    feats = []
    for smiles in df_all.smiles.astype(str):
        mol = Chem.MolFromSmiles(smiles,sanitize=False)
        Chem.SanitizeMol(mol,sanitizeOps=Chem.SANITIZE_FINDRADICALS|Chem.SANITIZE_SETHYBRIDIZATION)
        feats.append(len(Chem.Mol.GetSubstructMatches(x, smarts, uniquify = True))
    feats_vstack = np.vstack(feats)
    feat_num += 1
    with open(str(feat_num) + "-data", "w") as f:
        pickle.dump(feats_vstack, f, pickle.HIGHEST_PROTOCOL)
                 



print "Done reading feats"

#Drop the 'smiles' column
df_all = df_all.drop(['smiles'], axis=1)
vals = df_all.values
X_train = vals[:test_idx]
X_test = vals[test_idx:]

RF = RandomForestRegressor()
RF.fit(X_train, Y_train)
RF_pred = RF.predict(X_test)

def write_to_file(filename, predictions):
    with open(filename, "w") as f:
        f.write("Id,Prediction\n")
        for i,p in enumerate(predictions):
            f.write(str(i+1) + "," + str(p) + "\n")

write_to_file("pred.csv", RF_pred)
