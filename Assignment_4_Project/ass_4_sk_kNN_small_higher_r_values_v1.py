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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier

#from sklearn.linear_model import LogisticRegression

np.random.seed(100)

test_set_1 = pickle.load( open( "test_small.pkl", "rb" ) )
k_fold_set_1 = pickle.load( open( "k_fold_small.pkl", "rb" ) )

test_set_2 = pickle.load( open( "test_set_2_small.pkl", "rb" ) )
k_fold_set_2 = pickle.load( open( "k_fold_set_2_small.pkl", "rb" ) )

test_set_1_r3 = pickle.load( open( "test_set_1_r3_small.pkl", "rb" ) )
k_fold_set_1_r3 = pickle.load( open( "k_fold_set_1_r3_small.pkl", "rb" ) )

test_set_2_r3 = pickle.load( open( "test_set_2_r3_small.pkl", "rb" ) )
k_fold_set_2_r3 = pickle.load( open( "k_fold_set_2_r3_small.pkl", "rb" ) )

test_set_1_r1 = pickle.load( open( "test_set_1_r1_small.pkl", "rb" ) )
k_fold_set_1_r1 = pickle.load( open( "k_fold_set_1_r1_small.pkl", "rb" ) )

test_set_2_r1 = pickle.load( open( "test_set_2_r1_small.pkl", "rb" ) )
k_fold_set_2_r1 = pickle.load( open( "k_fold_set_2_r1_small.pkl", "rb" ) )

test_set_1_r5 = pickle.load( open( "test_set_1_r5_small.pkl", "rb" ) )
k_fold_set_1_r5 = pickle.load( open( "k_fold_set_1_r5_small.pkl", "rb" ) )

test_set_2_r5 = pickle.load( open( "test_set_2_r5_small.pkl", "rb" ) )
k_fold_set_2_r5 = pickle.load( open( "k_fold_set_2_r5_small.pkl", "rb" ) )

test_set_1_r10 = pickle.load( open( "test_set_1_r10_small.pkl", "rb" ) )
k_fold_set_1_r10 = pickle.load( open( "k_fold_set_1_r10_small.pkl", "rb" ) )

test_set_2_r10 = pickle.load( open( "test_set_2_r10_small.pkl", "rb" ) )
k_fold_set_2_r10 = pickle.load( open( "k_fold_set_2_r10_small.pkl", "rb" ) )

test_set_1_r7 = pickle.load( open( "test_set_1_r7_small.pkl", "rb" ) )
k_fold_set_1_r7 = pickle.load( open( "k_fold_set_1_r7_small.pkl", "rb" ) )

test_set_2_r7 = pickle.load( open( "test_set_2_r7_small.pkl", "rb" ) )
k_fold_set_2_r7 = pickle.load( open( "k_fold_set_2_r7_small.pkl", "rb" ) )

train_df1 = pd.DataFrame(columns = list(test_set_1.columns.values))
train_df2 = pd.DataFrame(columns = list(test_set_2.columns.values))

train_df1_r3 = pd.DataFrame(columns = list(test_set_1_r3.columns.values))
train_df2_r3 = pd.DataFrame(columns = list(test_set_2_r3.columns.values))

train_df1_r1 = pd.DataFrame(columns = list(test_set_1_r1.columns.values))
train_df2_r1 = pd.DataFrame(columns = list(test_set_2_r1.columns.values))

train_df1_r5 = pd.DataFrame(columns = list(test_set_1_r5.columns.values))
train_df2_r5 = pd.DataFrame(columns = list(test_set_2_r5.columns.values))

train_df1_r10 = pd.DataFrame(columns = list(test_set_1_r10.columns.values))
train_df2_r10 = pd.DataFrame(columns = list(test_set_2_r10.columns.values))

train_df1_r7 = pd.DataFrame(columns = list(test_set_1_r7.columns.values))
train_df2_r7 = pd.DataFrame(columns = list(test_set_2_r7.columns.values))

for j in range(len(k_fold_set_1)):
    train_df1 = train_df1.append(k_fold_set_1[j] , ignore_index = True)
    train_df2 = train_df2.append(k_fold_set_2[j] , ignore_index = True)
    train_df1_r3 = train_df1_r3.append(k_fold_set_1_r3[j] , ignore_index = True)
    train_df2_r3 = train_df2_r3.append(k_fold_set_2_r3[j] , ignore_index = True)
    train_df1_r1 = train_df1_r1.append(k_fold_set_1_r1[j] , ignore_index = True)
    train_df2_r1 = train_df2_r1.append(k_fold_set_2_r1[j] , ignore_index = True)
    train_df1_r5 = train_df1_r5.append(k_fold_set_1_r5[j] , ignore_index = True)
    train_df2_r5 = train_df2_r5.append(k_fold_set_2_r5[j] , ignore_index = True)
    train_df1_r10 = train_df1_r10.append(k_fold_set_1_r10[j] , ignore_index = True)
    train_df2_r10 = train_df2_r10.append(k_fold_set_2_r10[j] , ignore_index = True)
    train_df1_r7 = train_df1_r7.append(k_fold_set_1_r7[j] , ignore_index = True)
    train_df2_r7 = train_df2_r7.append(k_fold_set_2_r7[j] , ignore_index = True)

y1 = train_df1['active'].astype('category')
y2 = train_df2['active'].astype('category')
y1_r3 = train_df1_r3['active'].astype('category')
y2_r3 = train_df2_r3['active'].astype('category')
y1_r1 = train_df1_r1['active'].astype('category')
y2_r1 = train_df2_r1['active'].astype('category')
y1_r5 = train_df1_r5['active'].astype('category')
y2_r5 = train_df2_r5['active'].astype('category')
y1_r10 = train_df1_r10['active'].astype('category')
y2_r10 = train_df2_r10['active'].astype('category')
y1_r7 = train_df1_r7['active'].astype('category')
y2_r7 = train_df2_r7['active'].astype('category')

#.astype('category')
#column = y.columns

#y = ['A' if y.iloc[i]==0.0 else 'B' for i in range(len(y))]
#r for r in range(df.shape[0]) if r not in bag_i_indexes]
#y = pd.DataFrame(y,columns = column)

#print(y.dtype)
#print(y.unique())
#display(train_df)
#display(train_df2)

X1 = train_df1.copy()
X2 = train_df2.copy()
X1_r3 = train_df1_r3.copy()
X2_r3 = train_df2_r3.copy()
X1_r1 = train_df1_r1.copy()
X2_r1 = train_df2_r1.copy()
X1_r5 = train_df1_r5.copy()
X2_r5 = train_df2_r5.copy()
X1_r10 = train_df1_r10.copy()
X2_r10 = train_df2_r10.copy()
X1_r7 = train_df1_r7.copy()
X2_r7 = train_df2_r7.copy()

X1.drop(columns=['active', 'index'], inplace=True)
X2.drop(columns=['active', 'index'], inplace=True)
X1_r3.drop(columns=['active', 'index'], inplace=True)
X2_r3.drop(columns=['active', 'index'], inplace=True)
X1_r1.drop(columns=['active', 'index'], inplace=True)
X2_r1.drop(columns=['active', 'index'], inplace=True)
X1_r5.drop(columns=['active', 'index'], inplace=True)
X2_r5.drop(columns=['active', 'index'], inplace=True)
X1_r10.drop(columns=['active', 'index'], inplace=True)
X2_r10.drop(columns=['active', 'index'], inplace=True)
X1_r7.drop(columns=['active', 'index'], inplace=True)
X2_r7.drop(columns=['active', 'index'], inplace=True)

X1, column_filter = pas.create_column_filter(X1)
X1, imputation = pas.create_imputation(X1)
X1, normalization = pas.create_normalization(X1)

X2, column_filter = pas.create_column_filter(X2)
X2, imputation = pas.create_imputation(X2)
X2, normalization = pas.create_normalization(X2)

X1_r3, column_filter = pas.create_column_filter(X1_r3)
X1_r3, imputation = pas.create_imputation(X1_r3)
X1_r3, normalization = pas.create_normalization(X1_r3)

X2_r3, column_filter = pas.create_column_filter(X2_r3)
X2_r3, imputation = pas.create_imputation(X2_r3)
X2_r3, normalization = pas.create_normalization(X2_r3)

X1_r1, column_filter = pas.create_column_filter(X1_r1)
X1_r1, imputation = pas.create_imputation(X1_r1)
X1_r1, normalization = pas.create_normalization(X1_r1)

X2_r1, column_filter = pas.create_column_filter(X2_r1)
X2_r1, imputation = pas.create_imputation(X2_r1)
X2_r1, normalization = pas.create_normalization(X2_r1)

X1_r5, column_filter = pas.create_column_filter(X1_r5)
X1_r5, imputation = pas.create_imputation(X1_r5)
X1_r5, normalization = pas.create_normalization(X1_r5)

X2_r5, column_filter = pas.create_column_filter(X2_r5)
X2_r5, imputation = pas.create_imputation(X2_r5)
X2_r5, normalization = pas.create_normalization(X2_r5)

X1_r10, column_filter = pas.create_column_filter(X1_r10)
X1_r10, imputation = pas.create_imputation(X1_r10)
X1_r10, normalization = pas.create_normalization(X1_r10)

X2_r10, column_filter = pas.create_column_filter(X2_r10)
X2_r10, imputation = pas.create_imputation(X2_r10)
X2_r10, normalization = pas.create_normalization(X2_r10)

X1_r7, column_filter = pas.create_column_filter(X1_r7)
X1_r7, imputation = pas.create_imputation(X1_r7)
X1_r7, normalization = pas.create_normalization(X1_r7)

X2_r7, column_filter = pas.create_column_filter(X2_r7)
X2_r7, imputation = pas.create_imputation(X2_r7)
X2_r7, normalization = pas.create_normalization(X2_r7)
#df2, self.one_hot = create_one_hot(df2)
#display(X)

cv = KFold(n_splits=10, random_state=1, shuffle=True)
#model = LogisticRegression()
#num_trees = [10,60,100]
num_trees = [1,3,5,7,10]#number of neighbors
#num_trees = [100]

scores = []
scores2 = []
scores3 = []
scores4 = []
scores5 = []
scores6 = []
scores7 = []
scores8 = []
scores9 = []
scores10 = []
scores11 = []
scores12 = []

for num in num_trees:
    #model = RandomForestClassifier(n_estimators=num, random_state=0, max_features='log2')
    #model = AdaBoostClassifier(n_estimators=num)
    model = KNeighborsClassifier(n_neighbors=num)

    scores.append(np.mean(cross_val_score(model, X1, y1, scoring="roc_auc", cv=cv, n_jobs=-1)))
    scores2.append(np.mean(cross_val_score(model, X2, y2, scoring="roc_auc", cv=cv, n_jobs=-1)))

    scores3.append(np.mean(cross_val_score(model, X1_r7, y1_r7, scoring="roc_auc", cv=cv, n_jobs=-1)))
    scores4.append(np.mean(cross_val_score(model, X2_r7, y2_r7, scoring="roc_auc", cv=cv, n_jobs=-1)))

    scores5.append(np.mean(cross_val_score(model, X1_r3, y1_r3, scoring="roc_auc", cv=cv, n_jobs=-1)))
    scores6.append(np.mean(cross_val_score(model, X2_r3, y2_r3, scoring="roc_auc", cv=cv, n_jobs=-1)))

    scores7.append(np.mean(cross_val_score(model, X1_r1, y1_r1, scoring="roc_auc", cv=cv, n_jobs=-1)))
    scores8.append(np.mean(cross_val_score(model, X2_r1, y2_r1, scoring="roc_auc", cv=cv, n_jobs=-1)))

    scores9.append(np.mean(cross_val_score(model, X1_r5, y1_r5, scoring="roc_auc", cv=cv, n_jobs=-1)))
    scores10.append(np.mean(cross_val_score(model, X2_r5, y2_r5, scoring="roc_auc", cv=cv, n_jobs=-1)))

    scores11.append(np.mean(cross_val_score(model, X1_r10, y1_r10, scoring="roc_auc", cv=cv, n_jobs=-1)))
    scores12.append(np.mean(cross_val_score(model, X2_r10, y2_r10, scoring="roc_auc", cv=cv, n_jobs=-1)))
# report performance
#print(num_trees)
#scores5 = [0.9220658,]

print('original set1',scores)
print('original set2',scores2)
print()
print('original set1 r1x',scores7)
print('original set2 r1x',scores8)
print()
print('original set1 r3x',scores5)
print('original set2 r3x',scores6)
print()
print('original set1 r5x',scores9)
print('original set2 r5x',scores10)
print()
print('original set1 r7x',scores3)
print('original set2 r7x',scores4)
print()
print('original set1 r10x',scores11)
print('original set2 r10x',scores12)
print()


#print('ROC: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))
plt.plot(num_trees, scores, label = 'feature set 1 original')
plt.plot(num_trees, scores2, label = 'feature set 2 original')

plt.plot(num_trees, scores7, label = 'feature set 1 over-sampling = 100%')
plt.plot(num_trees, scores8, label = 'feature set 2 over-sampling = 100%')

plt.plot(num_trees, scores5, label = 'feature set 1 over-sampling = 300%')
plt.plot(num_trees, scores6, label = 'feature set 2 over-sampling = 300%')

plt.plot(num_trees, scores9, label = 'feature set 1 over-sampling = 500%')
plt.plot(num_trees, scores10, label = 'feature set 2 over-sampling = 500%')

plt.plot(num_trees, scores3, label = 'feature set 1 over-sampling = 700%')
plt.plot(num_trees, scores4, label = 'feature set 2 over-sampling = 700%')

plt.plot(num_trees, scores11, label = 'feature set 1 over-sampling = 1000%')
plt.plot(num_trees, scores12, label = 'feature set 2 over-sampling = 1000%')

plt.xlabel('number of neighbors')
plt.legend()
plt.title('AUC vs. feature sets / over-sampling rate/ number of neighbors in kNN')
#plt.savefig("AUC_RF_vs_tree_num_orig.jpg")
plt.show()
