import numpy as np
import pandas as pd
import time
from sklearn.tree import DecisionTreeClassifier
from IPython.display import display
import  past_ass_v2 as pas
import pickle


np.random.seed(100)

test = pickle.load( open( "test.pkl", "rb" ) )
k_fold = pickle.load( open( "k_fold.pkl", "rb" ) )

#test = pickle.load( open( "test_set_2.pkl", "rb" ) )
#k_fold = pickle.load( open( "k_fold_set_2.pkl", "rb" ) )

#display(test)
#print(len(k_fold))
#display(k_fold[0])

#display(test_set_2)
#print(len(k_fold_set_2))
#display(k_fold_set_2[0])

#print(k_fold[0].columns)
#print(k_fold[0].dtypes)
rf = pas.RandomForest()
train_df = pd.DataFrame(columns = list(test.columns.values))
#train_df = pd.DataFrame()
results = []

for i in range(len(k_fold)):
    test_df = k_fold[i]
    #display(test_df)
    for j in range(len(k_fold)):
        if j!= i:
            train_df = train_df.append(k_fold[j] , ignore_index = True)
            #train_df.append([1,1,1,1,1,1,1], ignore_index = True)
            #display(train_df)

    #print('rows', train_df.shape[0])
    #print('columns', train_df.shape[1])
    #print()
    #print('rows set', k_fold[0].shape[0])
    #print('columns set', k_fold[0].shape[1])

    t0 = time.perf_counter()
    rf.fit(train_df)
    print("Training time: {:.2f} s.".format(time.perf_counter()-t0))
    #print("OOB accuracy: {:.4f}".format(rf.oob_acc))

    test_labels = test_df["active"]
    t0 = time.perf_counter()
    predictions = rf.predict(test_df)
    print("Testing time: {:.2f} s.".format(time.perf_counter()-t0))
    print("Accuracy: {:.4f}".format(pas.accuracy(predictions,test_labels)))
    print("AUC: {:.4f}".format(pas.auc(predictions,test_labels))) # Comment this out if not implemented in assignment 1
    print("Brier score: {:.4f}".format(pas.brier_score(predictions,test_labels))) # Comment this out if not implemented in assignment 1
    results.append([pas.accuracy(predictions,test_labels),pas.brier_score(predictions,test_labels),
                  pas.auc(predictions,test_labels)])

results = pd.DataFrame(results,index=np.arange(len(k_fold)),columns=["Accuracy","Brier score","AUC"])
print()
display("results",results)
