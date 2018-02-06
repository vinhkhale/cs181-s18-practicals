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

func_group_list = ["[CX4]", "[$([CX2](=C)=C)]", "[$([CX3]=[CX3])]","[CX3]=[OX1]","[OX1]=CN", "[CX3](=[OX1])O","[CX3](=[OX1])[F,Cl,Br,I]","[NX3][CX3](=[OX1])[#6]", "[CX3H1](=O)[#6]","CX3](=[OX1])(O)O","[CX3](=O)[OX2H1]","[#6][CX3](=O)[#6]","[CX3](=O)[OX1H0-,OX2H1]","[OD2]([#6])[#6]", "[H]", "[H+]", "[+H]","[NX3;H2,H1;!$(NC=O)]", "[NX3,NX4+][CX4H]([*])[CX3](=[OX1])[O,N]","[NX3][CX3]=[CX3]","[$(*-[NX2-]-[NX2+]#[NX1]),$(*-[NX2]=[NX2+]=[NX1-])]","H2NNH2","[$([NX3](=[OX1])(=[OX1])O),$([NX3+]([OX1-])(=[OX1])O)] ","[NX1]#[CX2]","[OX2H]","[OX2H][#6X3]=[#6]","[#6][OX2H]","[OX2H][CX3]=[OX1]","[OX2,OX1-][OX2,OX1-]","[$(P(=[OX1])([$([OX2H]),$([OX1-]),$([OX2]P)])([$([OX2H]),$([OX1-]),$([OX2]P)])[$([OX2H]),$([OX1-]),$([OX2]P)]),$([P+]([OX1-])([$([OX 2H]),$([OX1-]),$([OX2]P)])([$([OX2H]),$([OX1-]),$([OX2]P)])[$([OX2H]),$([OX1-]),$([OX2]P)])]","[S-][CX3](=S)[#6]", "[#16X2H]", "[SX2]", "[NX3][CX3]=[SX1]","[#16X2H0]","[$([#16X3](=[OX1])[OX2H0]),$([#16X3+]([OX1-])[OX2H0])]","[$([#16X4](=[OX1])(=[OX1])([OX2H,OX1H0-])[OX2][#6]),$([#16X4+2]([OX1-])([OX1-])([OX2H,OX1H0-])[OX2][#6])]","[#6][F,Cl,Br,I]","[F,Cl,Br,I]", "[F,Cl,Br,I].[F,Cl,Br,I].[F,Cl,Br,I]","[CX3](=[OX1])[F,Cl,Br,I]","[$([cX2+](:*):*)]","[$([cX3](:*):*),$([cX2+](:*):*)] ","[$([cX3](:*):*),$([cX2+](:*):*),$([CX3]=*),$([CX2+]=*)] ","[$([nX3](:*):*),$([nX2](:*):*),$([#7X2]=*),$([NX3](=*)=*),$([#7X3+](-*)=*),$([#7X3+H]=*)]", "[$([NX4+]),$([NX3]);!$(*=*)&!$(*:*)]","[$([#1X1][$([NX4+]),$([NX3]);!$(*=*)&!$(*:*)])]","[$([SX3]=N)]", "[$([NX1]#*)]","[R0;D2][R0;D2][R0;D2][R0;D2]","[cR1]1[cR1][cR1][cR1][cR1][cR1]1","[sX2r5]","*/,\[R]=;@[R]/,\*","[cR1]1[cR1][cR1][cR1][cR1][cR1]1.[cR1]1[cR1][cR1][cR1][cR1][cR1]1","c12ccccc1cccc2","[!H0;F,Cl,Br,I,N+,$([OH]-*=[!#6]),+]","[CX3](=O)[OX2H1]","[$([OH]-*=[!#6])]","[$([#16X4](=[OX1])(=[OX1])([#6])[OX2H,OX1H0-]),$([#16X4+2]([OX1-])([OX1-])([#6])[OX2H,OX1H0-])]", "[CX3](=[OX1])[F,Cl,Br,I]", "[NX2-]","[OX2H+]=*","[OX3H2+]", "[$([cX2+](:*):*)]", "[$([NX1-]=[NX2+]=[NX1-]),$([NX1]#[NX2+]-[NX1-2])]", "([!-0!-1!-2!-3!-4].[!+0!+1!+2!+3!+4])", "[#6,#7;R0]=[#8]", "[!$([#6,F,Cl,Br,I,o,s,nX3,#7v5,#15v5,#16v4,#16v6,*+1,*+2,*+3,])]", "[!$([#6,H0,-,-2,-3])]", "[!H0;#7,#8,#9]", "[O,N;!H0]-*~*-*=[$([C,N;R0]=O)]", "[#6;X3v3+0]", "[#7;X2v4+0]"]

for term in func_group_list:
    smarts = Chem.MolFromSmarts(term)
    rel_count = np.vstack([len(Chem.Mol.GetSubstructMatches(x, smarts, uniquify = True)) for x in mol_all])
    
    # adds these guys to the data table, len might be broken
    df_all['smiles_len'] = pd.DataFrame(smiles_len)


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
