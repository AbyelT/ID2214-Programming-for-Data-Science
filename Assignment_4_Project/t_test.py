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

#Bonferroni CORRECTION
#alpha new = alpha original /n with n = number of comparisons
#comparisons:
#1.RF_100-100 vs RF_50_70
#2.RF_50_100 vs RF_50_70
#3.kNN_5_100 vs kNN_5_50
#4 kNN_x vs RF_x
# hence corrected alpha = 0.05/4 = 0.0125 and corrected alpha/2 is 2.54425


AUC_RF_ntree_50_overs_70 = pickle.load( open( 'AUC_RF_ntree_50_overs_70.pkl', "rb" ) )
AUC_RF_ntree_50_overs_100 = pickle.load( open( 'AUC_RF_ntree_50_overs_100.pkl', "rb" ) )
AUC_RF_ntree_100_overs_100 = pickle.load( open( 'AUC_RF_ntree_100_overs_100.pkl', "rb" ) )
AUC_kNN_k_5_r_50 = pickle.load( open( 'AUC_kNN_k_5_r_50.pkl', "rb" ) )
AUC_kNN_k_5_r_100 = pickle.load( open( 'AUC_kNN_k_5_r_100.pkl', "rb" ) )

#print(AUC_RF_ntree_50_overs_70)

diff_models = AUC_RF_ntree_100_overs_100 - AUC_RF_ntree_50_overs_70

score = np.nanmean(diff_models)
sample_standard_deviation = statistics.stdev(diff_models)
lower_bound_AUC = score - 2.54425 * sample_standard_deviation/np.sqrt(len(diff_models))
upper_bound_AUC = score + 2.54425 * sample_standard_deviation/np.sqrt(len(diff_models))
print()
#print('Average difference', score)
print('comparison 1:', '(',lower_bound_AUC,',',upper_bound_AUC,')')
print()


diff_models = AUC_RF_ntree_50_overs_100 - AUC_RF_ntree_50_overs_70

score = np.nanmean(diff_models)
sample_standard_deviation = statistics.stdev(diff_models)
lower_bound_AUC = score - 2.54425 * sample_standard_deviation/np.sqrt(len(diff_models))
upper_bound_AUC = score + 2.54425 * sample_standard_deviation/np.sqrt(len(diff_models))
print()
#print('Average difference', score)
print('comparison 2:', '(',lower_bound_AUC,',',upper_bound_AUC,')')
print()

diff_models = AUC_kNN_k_5_r_50 - AUC_kNN_k_5_r_100

score = np.nanmean(diff_models)
sample_standard_deviation = statistics.stdev(diff_models)
lower_bound_AUC = score - 2.54425 * sample_standard_deviation/np.sqrt(len(diff_models))
upper_bound_AUC = score + 2.54425 * sample_standard_deviation/np.sqrt(len(diff_models))
print()
#print('Average difference', score)
print('comparison 3:', '(',lower_bound_AUC,',',upper_bound_AUC,')')
print()

diff_models = AUC_kNN_k_5_r_50 - AUC_RF_ntree_50_overs_70

score = np.nanmean(diff_models)
sample_standard_deviation = statistics.stdev(diff_models)
lower_bound_AUC = score - 2.54425 * sample_standard_deviation/np.sqrt(len(diff_models))
upper_bound_AUC = score + 2.54425 * sample_standard_deviation/np.sqrt(len(diff_models))
print()
#print('Average difference', score)
print('comparison 4:', '(',lower_bound_AUC,',',upper_bound_AUC,')')
print()
