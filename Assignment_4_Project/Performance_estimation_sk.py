import numpy as np
import pandas as pd
import time
from sklearn.tree import DecisionTreeClassifier
from IPython.display import display
import  past_ass_v2 as pas
import pickle
from sklearn.datasets import make_classification
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import brier_score_loss
import statistics

def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)

np.random.seed(100)

#test_set = pickle.load( open( 'test_set_2_r100_full.pkl', "rb" ) )
#train_set = pickle.load( open( "k_fold_set_2_r100_full.pkl", "rb" ) )

test_set = pickle.load( open( 'test_set_2_r_70_full.pkl', "rb" ) )
train_set = pickle.load( open( "k_fold_set_2_r_70_full.pkl", "rb" ) )

test_sets_indexes = np.random.choice(len(test_set),(50,int(len(test_set)/50)),replace=False)

test_sets = []

for elem in test_sets_indexes:
    data_set = [test_set.iloc[i] for i in elem]
    df3 = pd.DataFrame(data_set,columns = list(test_set))
    df3.index = range(len(df3.index))
    test_sets.append(df3)

X_train = train_set.copy()
y_train = X_train['active'].astype('category')

X_train.drop(columns=['active', 'index'], inplace=True)
X_train, column_filter = pas.create_column_filter(X_train)
X_train, imputation = pas.create_imputation(X_train)
X_train, normalization = pas.create_normalization(X_train)

c= 50

rf = RandomForestClassifier(n_estimators= c, random_state=0, max_features='log2')
#rf = KNeighborsClassifier(n_neighbors=c)
t0 = time.perf_counter()
rf.fit(X_train, y_train)
print("Training time: {:.2f} s.".format(time.perf_counter()-t0))

results = []
for X_test in test_sets:
    t0 = time.perf_counter()
    y_true = X_test['active'].astype('category')
    X_test.drop(columns=['active', 'index'], inplace=True)
    X_test = pas.apply_column_filter(X_test, column_filter)
    X_test = pas.apply_imputation(X_test, imputation)
    X_test = pas.apply_normalization(X_test, normalization)
    y_pred = rf.predict(X_test)
    y_score = rf.predict_proba(X_test)
    accuracy = accuracy_score(y_true, y_pred)
    try:
        AUC = roc_auc_score(y_true, y_score[:, 1])
    except ValueError:
        pass

    try:
        Brier = brier_score_loss(y_true, y_score[:, 1])
    except ValueError:
        pass

    print("Testing time: {:.2f} s.".format(time.perf_counter()-t0))
    #print("Accuracy: {:.4f}".format(accuracy))
    #print("AUC: {:.4f}".format(AUC))
    #print("Brier score: {:.4f}".format(Brier))
    results.append([accuracy ,Brier,AUC])

results = pd.DataFrame(results,index=np.arange(len(test_sets)),columns=["Accuracy","Brier score","AUC"])
#print()
#display("results",results)

score = np.nanmean(results['AUC'])
sample_standard_deviation = statistics.stdev(results['AUC'])
lower_bound_AUC = score - 2.009 * sample_standard_deviation/np.sqrt(len(test_sets))
upper_bound_AUC = score + 2.009 * sample_standard_deviation/np.sqrt(len(test_sets))
print()
print('Average AUC, numtree, r', score, c, '70')
print('AUC range 95%:', '(',lower_bound_AUC,',',upper_bound_AUC,')')
print()

save_object(results['AUC'] , 'AUC_RF_ntree_50_overs_70.pkl')
