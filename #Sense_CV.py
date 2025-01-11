# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
# Sklearn imports
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
# Tensorflow import
# import tensorflow as tf
# # DiCE imports
# import dice_ml
# from dice_ml import Dice
# from dice_ml.utils import helpers  # helper functions
from xgboost import XGBClassifier, XGBRegressor
from xgboost import plot_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error
from sklearn.linear_model import LinearRegression, BayesianRidge, ridge_regression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
import xgboost
# from category_encoders import TargetEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    mean_absolute_error,
    precision_recall_curve,
    auc,
    cohen_kappa_score,
    confusion_matrix,
    fbeta_score,
    precision_recall_fscore_support,
    roc_curve,
    brier_score_loss,
    accuracy_score
)
import shap
import pyreadstat
from catboost import CatBoostClassifier
from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor
from alibi.explainers import ALE, plot_ale
from sklearn.compose import make_column_selector as selector
import copy
import lightgbm as lgb
from imbens.ensemble import (
    SelfPacedEnsembleClassifier, BalanceCascadeClassifier,
    BalancedRandomForestClassifier, EasyEnsembleClassifier,
    RUSBoostClassifier, UnderBaggingClassifier,
    AdaCostClassifier,AdaUBoostClassifier,AsymBoostClassifier,
    OverBoostClassifier,SMOTEBoostClassifier,KmeansSMOTEBoostClassifier,
    OverBaggingClassifier,SMOTEBaggingClassifier)
from imbens.utils._plot import plot_2Dprojection_and_cardinality
from sklearn.model_selection import cross_validate
# np.random.seed(3407)
# random.seed(3407)
# np.random.seed(2333)
# random.seed(2333)

def process_no_ordered(dataset):
    dataset['Economic_area'] = pd.Categorical(dataset['Economic_area'], 
                                                     categories=[1, 2, 3, 4, 5], ordered=False)
    dataset['Marital'] = pd.Categorical(dataset['Marital'], 
                                                     categories=[1, 2, 3], ordered=False)
    dataset['Standardized_training_methods'] = pd.Categorical(dataset['Standardized_training_methods'], 
                                                     categories=[1, 2, 3], ordered=False)
    dataset['Department'] = pd.Categorical(dataset['Department'], 
                                                     categories=[1,2,3,4,5,6,7], ordered=False)
    
    return dataset

minmaxnorm = False
SMOTE_Process = False
SHAP_Analysis = False #False True
ALE_Analysis = False
print('minmaxnorm: ', minmaxnorm)
print('SMOTE_Process: ', SMOTE_Process)
print('SHAP_Analysis: ', SHAP_Analysis)
print('ALE_Analysis: ', ALE_Analysis)
dataset, meta = pyreadstat.read_sav("0.7 samples.sav")
var_names =  pd.read_excel('Var_names.xlsx')
dataset.columns = var_names['Variable']
dataset = process_no_ordered(dataset)

categorical_columns_selector = selector(dtype_exclude=np.number)
numerical_columns_selector = selector(dtype_include=np.number)
numerical_columns = numerical_columns_selector(dataset)
categorical_columns = categorical_columns_selector(dataset)
numerical_preprocessor = MinMaxScaler()
# pd dummy 处理分类变量，哑变量第一列未删除
cate_encoded = pd.get_dummies(dataset, columns=categorical_columns, drop_first=False, dtype=float)
# minmax 处理连续变量，放缩0-1区间，不影响哑变量编码


dataset_use = copy.deepcopy(cate_encoded)
datasetX = dataset_use.drop(['Experienced_violence', 'See_violence', 'WPVboth',
                             'Experienced_violence_score',
                             'See_violence_score'], axis=1)
Y = dataset_use['WPVboth']
Y = Y.mask(Y==2, 0)
# 发现nan
datasetX.loc[datasetX.isnull().any(axis=1), datasetX.isnull().any()]
Y = Y.drop(datasetX[datasetX.isnull().T.any()].index.values) #label也删
datasetX = datasetX.dropna(how='any') #删除有缺失值的行

Models = {
    'XGB':XGBClassifier(max_depth=9, min_child_weight=6, gamma=7, learning_rate= 0.01369,
                        subsample=0.9302, colsample_bytree=0.65798, eval_metric='auc',
                        scale_pos_weight=3, max_delta_step=3),
    'GBDT':GradientBoostingClassifier(n_estimators=1025,max_depth=14,min_samples_split=2,
                                      min_samples_leaf=10,min_weight_fraction_leaf=0.35527),
    'RF':RandomForestClassifier(n_estimators=120,max_depth=8,min_samples_split=8,
                                min_samples_leaf=8,class_weight='balanced'),
    'LR':LogisticRegression( tol=0.0002825,C=0.033572,fit_intercept=True,
        random_state=0, class_weight='balanced'),
    'Cat':CatBoostClassifier(eval_metric='F1',l2_leaf_reg=0.01272,
                              learning_rate=0.01308,n_estimators=823,depth=15,
                              scale_pos_weight=29,verbose=False),
    'LGB':lgb.LGBMClassifier(metric='auc',is_unbalance=True,
                              random_state=0,n_estimators=1714,
                              reg_alpha=9.99161,
                              reg_lambda=0.001593,colsample_bytree=0.3,
                              subsample=0.7,learning_rate=0.02,
                              max_depth=20,num_leaves=173,min_child_samples=244,
                              min_data_per_groups=7,
                              verbosity=-1),
    
}
# from sklearn.utils import shuffle
# data_for_shuffle = copy.deepcopy(datasetX)
# data_for_shuffle['Y'] = Y
# data_for_shuffle = shuffle(data_for_shuffle)
# datasetX = data_for_shuffle.drop(['Y'] , axis=1).reset_index(drop=True)
# Y = data_for_shuffle['Y'].reset_index(drop=True)


# # scoring = {'prec_macro': 'precision_macro',
# #            'rec_macro': make_scorer(recall_score, average='macro')}
# # scores = cross_validate(clf, X, y, scoring=scoring,
# #                         cv=5, return_train_score=True)



from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2)
compare_result = ['Model', 'Pre_Mean', 'Pre_Std','Recall_Mean', 'Recall_Std',
                  'F1_Mean', 'F1_Std', 'ROCAUC_Mean', 'ROCAUC_Std', 
                  'PRAUC_Mean', 'PRAUC_Std','Brier_Mean', 'Brier_Std',
                  'ACC_Mean', 'ACC_Std']
for model_name in Models:
    print(model_name)
    model = Models[model_name]
    mid_results = ['precision', 'recall', 'f1', 'ROC_AUC', 'aucpr', 'brier', 'acc']
    for train, test in kfold.split(datasetX, Y):
        x_train = datasetX.values[train]
        y_train = Y.values[train]
        x_test = datasetX.values[test]
        y_test = Y.values[test]
        print('Positive in y_test: ', y_test.sum()/y_test.shape[0])
        print('Positive in y_train: ',y_train.sum()/y_train.shape[0])
 
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        y_pred_1_proba = model.predict_proba(x_test)[:, 1]
        y_pred_proba = model.predict_proba(x_test)
        tn, fp, fn, tp  = confusion_matrix(y_test, y_pred_proba.argmax(axis=1)).ravel()
        f2 = fbeta_score(y_test, y_pred, average='binary', beta=2)
        f0_5 = fbeta_score(y_test, y_pred, average='binary', beta=0.5)
        f1 = f1_score(y_test, y_pred, average='binary')
        precision, recall, _, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
        ROC_AUC = roc_auc_score(y_test, y_pred_1_proba, average='weighted')
        (precisions, recalls, _) = precision_recall_curve(y_test, y_pred_1_proba)
        aucpr = auc(recalls, precisions)
        AP = average_precision_score(y_test, y_pred)
        
        brier = brier_score_loss(y_test, y_pred_1_proba)
        acc = accuracy_score(y_test, y_pred)
        fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(x_test)[:,1])
        mid_results = np.row_stack((mid_results,[precision, recall, f1, ROC_AUC, aucpr, brier, acc]))
    compare_result = np.row_stack((compare_result, [model_name,
                                                    mid_results[1:, 0].astype(float).mean().round(3),
                                                    mid_results[1:, 0].astype(float).std().round(3),
                                                    mid_results[1:, 1].astype(float).mean().round(3),
                                                    mid_results[1:, 1].astype(float).std().round(3),
                                                    mid_results[1:, 2].astype(float).mean().round(3),
                                                    mid_results[1:, 2].astype(float).std().round(3),
                                                    mid_results[1:, 3].astype(float).mean().round(3),
                                                    mid_results[1:, 3].astype(float).std().round(3),
                                                    mid_results[1:, 4].astype(float).mean().round(3),
                                                    mid_results[1:, 4].astype(float).std().round(3),
                                                    mid_results[1:, 5].astype(float).mean().round(3),
                                                    mid_results[1:, 5].astype(float).std().round(3),
                                                    mid_results[1:, 6].astype(float).mean().round(3),
                                                    mid_results[1:, 6].astype(float).std().round(3),
                                                    ]))
import datetime
filename = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')+'_CV_results_Minmax_'+str(minmaxnorm)+'_SMOTE_'+str(SMOTE_Process)+'.csv'
np.savetxt(filename, compare_result, delimiter=',', fmt='%s')