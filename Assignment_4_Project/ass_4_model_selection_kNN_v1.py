import numpy as np
import pandas as pd
import time
from sklearn.tree import DecisionTreeClassifier
from IPython.display import display
import  past_ass_v2 as pas
import pickle

#test = pickle.load( open( "test.pkl", "rb" ) )
#k_fold = pickle.load( open( "k_fold.pkl", "rb" ) )
np.random.seed(100)

test_set_2 = pickle.load( open( "test_set_2.pkl", "rb" ) )
k_fold_set_2 = pickle.load( open( "k_fold_set_2.pkl", "rb" ) )

#display(test)
#print(len(k_fold))
#display(k_fold[0])

#display(test_set_2)
#print(len(k_fold_set_2))
#display(k_fold_set_2[0])

#print(k_fold[0].columns)
#print(k_fold[0].dtypes)

#train_df = k_fold[0]
#test_df = k_fold[1]

train_df = k_fold_set_2[0]
test_df = k_fold_set_2[1]

#rf = pas.RandomForest()
#nb_model = pas.NaiveBayes()
knn_model = pas.kNN()

t0 = time.perf_counter()
knn_model.fit(train_df)
print("Training time: {0:.2f} s.".format(time.perf_counter()-t0))

test_labels = test_df["active"]

k_values = [1,3,5,7,9]
results = np.empty((len(k_values),3))

for i in range(len(k_values)):
    t0 = time.perf_counter()
    predictions = knn_model.predict(test_df,k=k_values[i])
    print("Testing time (k={0}): {1:.2f} s.".format(k_values[i],time.perf_counter()-t0))
    results[i] = [pas.accuracy(predictions,test_labels),pas.brier_score(predictions,test_labels),
                  pas.auc(predictions,test_labels)] # Assuming that you have defined auc - remove otherwise

results = pd.DataFrame(results,index=k_values,columns=["Accuracy","Brier score","AUC"])

print()
display("results",results)

train_labels = train_df["active"]
predictions = knn_model.predict(train_df,k=1)
print("Accuracy on training set (k=1): {0:.4f}".format(pas.accuracy(predictions,train_labels)))
print("AUC on training set (k=1): {0:.4f}".format(pas.auc(predictions,train_labels)))
print("Brier score on training set (k=1): {0:.4f}".format(pas.brier_score(predictions,train_labels)))
