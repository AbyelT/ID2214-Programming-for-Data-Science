import numpy as np
import pandas as pd
import time
from sklearn.tree import DecisionTreeClassifier
#import ass_1_1a_vf as fil
#import ass_1_1b_vf as norm
#import ass_1_1c_vf as imp
#import ass_1_1d_vf as disc
#import ass_1_1e_vf as hot
#import ass_1_1f_vf as spl
#import ass_1_1g_v2 as acc
from IPython.display import display
#import ass_1_2a_vf as fol
#import ass_1_2b_vf as bri
#import ass_1_2c_vf as auc
#import assignment_1_summary as ass1
import past_ass_v2 as pas
import pickle

def over_sample(df, rate=10):
    df1 = df.copy()
    labels_1_indexes = np.where(df['active'].values == 1)[0]
    additional_samples = np.random.choice(labels_1_indexes, int((rate)*len(labels_1_indexes)))
    df2 = [df1.iloc[i] for i in additional_samples]
    df3= df1.append(df2)
    #print(len(df), len(df1), len(df2), len(df3))
    #print('initial_1',len(np.where(df['active'].values == 1)[0]))
    #print('final_1',len(np.where(df3['active'].values == 1)[0]))
    #print()
    #print('initial_0',len(np.where(df['active'].values == 0)[0]))
    #print('final_0',len(np.where(df3['active'].values == 0)[0]))
    return df3

def under_sample(df, rate=0.1):
    df1 = df.copy()
    labels_0_indexes = np.where(df['active'].values == 0)[0]
    delete_samples = np.random.choice(labels_0_indexes, int(rate*len(df)))
    df1 = df1.drop(index = delete_samples)
    return df1

def data_split(df, rate_test=0.1, k=10):
    df1 = df.copy()
    labels_1_indexes = np.where(df['active'].values == 1)[0]
    labels_1_indexes_sets = np.random.choice(labels_1_indexes, (k+1,int(len(labels_1_indexes)/(k+1))),replace=False)
    #print('1',len(df1), len(labels_1_indexes), len(labels_1_indexes_sets))
    #print(len(labels_1_indexes_sets[0]))

    labels_0_indexes = np.where(df['active'].values == 0)[0]
    labels_0_indexes_sets = np.random.choice(labels_0_indexes, (k+1,int(len(labels_0_indexes)/(k+1))),replace=False)
    #print('0',len(df1), len(labels_0_indexes), len(labels_0_indexes_sets))
    #print(len(labels_0_indexes_sets[0]))

    data_sets = zip(labels_1_indexes_sets,labels_0_indexes_sets)
    data_sets_list = list(data_sets)

    k_fold_training = []

    for elem in data_sets_list:
        df2 = pd.concat([df1.iloc[elem[0]], df1.iloc[elem[1]]])
        df2_reshuffled_indexes = np.random.choice(len(df2),len(df2),replace=False)
        data_set = [df2.iloc[i] for i in df2_reshuffled_indexes]
        df3 = pd.DataFrame(data_set,columns = list(df1))
        df3.index = range(len(df3.index))
        k_fold_training.append(df3)

    test_set = np.random.choice(len(k_fold_training),1,replace=False)

    #print(test_set)
    #print(len(set_labels_list), set_labels_list)
    #print(len(k_fold_training))

    #print(labels_1_indexes_sets)
    #labels_test = np.random.choice(len(df1), int(rate_test*len(df)),replace=False)
    #labels_training = [i for i in range(len(df1)) if i not in labels_test]
    #print(len(df1), len(labels_test), len(labels_training))
    test = k_fold_training[test_set[0]]
    k_fold_training.pop(test_set[0])

    #print('len test', len(test))
    #print(len(k_fold_training))

    #print(type(test))
    #print(type(k_fold_training[0]))
    #display(test)
    #test = pd.DataFrame(test, columns=list(df.columns))
    #k_fold_training = pd.DataFrame(k_fold_training, columns=list(df.columns))

    return k_fold_training, test

def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)


#set the seed
np.random.seed(100)
feature_set1_df = pd.read_csv("feature_set1.csv")
feature_set2_df = pd.read_csv("feature_set2.csv")

ON_feature_set_df = feature_set1_df.copy()
ON_feature_set_df = feature_set2_df.copy()

#rate = [0.1,1,2,5,10]
rate = 3

ON_feature_set1_df = over_sample(feature_set1_df,rate)
ON_feature_set2_df = over_sample(feature_set2_df,rate)

k_fold_set_1 , test_set_1 = data_split(ON_feature_set1_df)
k_fold_set_2 , test_set_2 = data_split(ON_feature_set2_df)

labels = test_set_1['active']
#label_classes = labels.unique()
#print(label_classes)
labels_0_indexes = np.where(labels.values == 0)[0]
labels_1_indexes = np.where(labels.values == 1)[0]

print('OVERSAMPLING3 0', len(labels_0_indexes))
print('OVERSAMPLING3 1', len(labels_1_indexes))

cut_off_k_fold = int(len(k_fold_set_1)/3)

k_fold_set_1_small = k_fold_set_1[:cut_off_k_fold]
k_fold_set_2_small = k_fold_set_2[:cut_off_k_fold]


test_set_1_small = test_set_1
test_set_2_small = test_set_2

save_object(k_fold_set_1_small, 'k_fold_set_1_r3_small.pkl')
save_object(test_set_1_small, 'test_set_1_r3_small.pkl')

save_object(k_fold_set_2_small, 'k_fold_set_2_r3_small.pkl')
save_object(test_set_2_small, 'test_set_2_r3_small.pkl')
