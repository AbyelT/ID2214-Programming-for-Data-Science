import numpy as np
import pandas as pd
import pickle
from rdkit import Chem
import rdkit.Chem.rdMolDescriptors as d
import rdkit.Chem.Fragments as f
import rdkit.Chem.Lipinski as l
from rdkit.Chem import AllChem
import past_ass_v2 as pas
from sklearn.ensemble import RandomForestClassifier


def splitSmiles(x, e):
    return x.split(",",1)[e] 

def separate_features(df):
    df["INDEX"] = df["INDEX,SMILES"].apply(lambda x: splitSmiles(x, 0))
    df["SMILES"] = df["INDEX,SMILES"].apply(lambda x: splitSmiles(x, 1))
    df.drop("INDEX,SMILES", axis=1, inplace=True)
    return df

def prepare_data(df):
    #separate index and SMILES
    #df = separate_features(df)
    
    #copy
    df1 = df.copy()
    df2 = pd.DataFrame(index=df1.index)
    df3 = pd.DataFrame(index=df1.index)

    for row in df1.index:
        mol = Chem.MolFromSmiles(df.loc[row, 'SMILES'])
        df2.loc[row, 'num_atoms'] = mol.GetNumAtoms()
        df2.loc[row, 'exact_mol_wt'] = d.CalcExactMolWt(mol)
        df2.loc[row, 'fr_AI_COO'] = f.fr_Al_COO(mol)
        df2.loc[row, 'heavy_atom_count'] = l.HeavyAtomCount(mol)
        #df2.loc[row, 'active'] = df.loc[row, 'ACTIVE']
        finger_print = np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=124))
        for i, value in enumerate(finger_print):
            df3.loc[row, f'bit-{i}'] = finger_print[i]
        #df3.loc[row, 'active'] = df.loc[row, 'ACTIVE']
        if(row % 10000 == 0): 
            print(row)
        
    return df2, df3

def test_final(feature_set1, feature_set2, final_test_set):
    y1 = feature_set1['active'].astype('category')
    y2 = feature_set2['active'].astype('category')
    
    x1 = feature_set1.copy()
    x2 = feature_set2.copy()
    x1.drop(columns=['active', 'index'], inplace=True)
    x2.drop(columns=['active', 'index'], inplace=True)
    
    x1, column_filter = pas.create_column_filter(x1)
    x1, imputation = pas.create_imputation(x1)
    x2, column_filter = pas.create_column_filter(x2)
    x2, imputation = pas.create_imputation(x2)
    
    num_trees = [150]
    score1 = []
    score2 = []
    
    for num in num_trees:
        model1 = RandomForestClassifier(n_estimators=num, random_state=0, max_features='log2')
        model2 = RandomForestClassifier(n_estimators=num, random_state=0, max_features='log2')
        
        #do fit
        model1.fit(x1, y1)
        model2.fit(x2, y2)
        
        #do predict
        score1.append(model1.predict_proba(final_test_set))
        score2.append(model2.predict_proba(final_test_set))
        #do calc auc
        #append to score
        #scores.append(np.mean(cross_val_score(model, X1, y1, scoring="roc_auc", cv=cv, n_jobs=-1)))


    print("test!")
    
def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)
    
    
#start
np.random.seed(100)
command = 1 #use later

#get data
kfold_set1_df = pickle.load( open( "k_fold.pkl", "rb" ) )
kfold_set2_df = pickle.load( open( "k_fold_2.pkl", "rb" ) )
final_test_data = pd.read_csv("test_smiles.csv")
final_test_set = pickle.load( open( "k_fold_2.pkl", "rb" ) )

#create dataframes
training_set1_df = pd.concat(kfold_set1_df)
training_set2_df = pd.concat(kfold_set2_df)

#create final test set
#final_test_set = prepare_data(final_test_data)
save_object(final_test_set, 'final_test_set.pkl')

#test the final test set
test_final(training_set1_df, training_set2_df, final_test_set)

print("ok!")
