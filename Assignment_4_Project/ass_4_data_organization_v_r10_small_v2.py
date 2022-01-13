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
import ass_4_data_organization_v_r01_small_v2 as dor


#set the seed
np.random.seed(100)
feature_set1_df = pd.read_csv("feature_set1.csv")
feature_set2_df = pd.read_csv("feature_set2.csv")

ON_feature_set_df = feature_set1_df.copy()
ON_feature_set_df = feature_set2_df.copy()

#rate = [0.1,1,2,5,10]
rate = 10

ON_feature_set1_df = dor.over_sample(feature_set1_df,rate)
ON_feature_set2_df = dor.over_sample(feature_set2_df,rate)

k_fold_set_1 , test_set_1 = dor.data_split(ON_feature_set1_df)
k_fold_set_2 , test_set_2 = dor.data_split(ON_feature_set2_df)

labels = test_set_1['active']

labels_0_indexes = np.where(labels.values == 0)[0]
labels_1_indexes = np.where(labels.values == 1)[0]

print('OVERSAMPLING10 0', len(labels_0_indexes))
print('OVERSAMPLING10 1', len(labels_1_indexes))

cut_off_k_fold = int(len(k_fold_set_1)/3)

k_fold_set_1_small = k_fold_set_1[:cut_off_k_fold]
k_fold_set_2_small = k_fold_set_2[:cut_off_k_fold]

#print('PRIMA', len(k_fold_set_1))
#print('DOPO',len(k_fold_set_1_small))

test_set_1_small = test_set_1
test_set_2_small = test_set_2
#save_object(k_fold_set_1, 'k_fold_set_1_r01.pkl')
#save_object(test_set_1, 'test_set_1_r01.pkl')

#save_object(k_fold_set_2, 'k_fold_set_2_r01.pkl')
#save_object(test_set_2, 'test_set_2_r01.pkl')

dor.save_object(k_fold_set_1_small, 'k_fold_set_1_r10_small.pkl')
dor.save_object(test_set_1_small, 'test_set_1_r10_small.pkl')

dor.save_object(k_fold_set_2_small, 'k_fold_set_2_r10_small.pkl')
dor.save_object(test_set_2_small, 'test_set_2_r10_small.pkl')
#rf = pas.RandomForest()
#nb_model = pas.NaiveBayes()
#knn_model = pas.kNN()
