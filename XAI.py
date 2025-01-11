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
    # dataset['Gender'] = pd.Categorical(dataset['Gender'], 
    #                                                  categories=[1, 2], ordered=False)
    # dataset['Hukou'] = pd.Categorical(dataset['Hukou'], 
    #                                                  categories=[1, 2], ordered=False)
    # dataset['Left_behind_children'] = pd.Categorical(dataset['Left_behind_children'], 
    #                                                  categories=[1, 2], ordered=False)
    # dataset['Religious'] = pd.Categorical(dataset['Religious'], 
    #                                                  categories=[1, 2], ordered=False)
    dataset['Marital'] = pd.Categorical(dataset['Marital'], 
                                                     categories=[1, 2, 3], ordered=False)
    # dataset['Live_with_parents'] = pd.Categorical(dataset['Live_with_parents'], 
    #                                                  categories=[1, 2], ordered=False)
    # dataset['Lunch_break'] = pd.Categorical(dataset['Lunch_break'], 
    #                                                  categories=[1, 2], ordered=False)
    # dataset['Alcohol_consumption'] = pd.Categorical(dataset['Alcohol_consumption'], 
    #                                                  categories=[1, 2], ordered=False)
    # dataset['Smoke'] = pd.Categorical(dataset['Smoke'], 
    #                                                  categories=[1, 2], ordered=False)
    # dataset['Exercise'] = pd.Categorical(dataset['Exercise'], 
    #                                                  categories=[1, 2], ordered=False)
    # dataset['Chronic_disease'] = pd.Categorical(dataset['Chronic_disease'], 
    #                                                  categories=[1, 2], ordered=False)
    # dataset['Experienced_medical_complaint'] = pd.Categorical(dataset['Experienced_medical_complaint'], 
    #                                                  categories=[1, 2], ordered=False)
    # dataset['See_medical_complaint'] = pd.Categorical(dataset['See_medical_complaint'], 
    #                                                  categories=[1, 2], ordered=False)
    dataset['Standardized_training_methods'] = pd.Categorical(dataset['Standardized_training_methods'], 
                                                     categories=[1, 2, 3], ordered=False)
    # dataset['Aim_policy'] = pd.Categorical(dataset['Aim_policy'], 
    #                                                  categories=[0, 1], ordered=False)
    # dataset['Aim_higher_class_hospital'] = pd.Categorical(dataset['Aim_higher_class_hospital'], 
    #                                                  categories=[0, 1], ordered=False)
    # dataset['Aim_big_city'] = pd.Categorical(dataset['Aim_big_city'], 
    #                                                  categories=[0, 1], ordered=False)
    # dataset['Aim_love_nursing'] = pd.Categorical(dataset['Aim_love_nursing'], 
    #                                                  categories=[0, 1], ordered=False)
    # dataset['Aim_parents_will'] = pd.Categorical(dataset['Aim_parents_will'], 
    #                                                  categories=[0, 1], ordered=False)
    # dataset['Aim_nursing_ability'] = pd.Categorical(dataset['Aim_nursing_ability'], 
    #                                                  categories=[0, 1], ordered=False)
    dataset['Department'] = pd.Categorical(dataset['Department'], 
                                                     categories=[1,2,3,4,5,6,7], ordered=False)
    
    return dataset

minmaxnorm = False
SMOTE_Process = False
SHAP_Analysis = True #False True
ALE_Analysis = True
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
                                      min_samples_leaf=10,
                                      min_weight_fraction_leaf=0.35527),
    'RF':RandomForestClassifier(n_estimators=120,max_depth=8,min_samples_split=8,
                                min_samples_leaf=8,class_weight='balanced'),
    'LR':LogisticRegression( tol=0.0002825,C=0.033572,fit_intercept=True,
        random_state=0, class_weight='balanced'),
    'CatBoost':CatBoostClassifier(eval_metric='F1',l2_leaf_reg=0.01272,
                              learning_rate=0.01308,n_estimators=823,depth=15,
                              scale_pos_weight=29,verbose=False),
    'LightGBM':lgb.LGBMClassifier(metric='auc',is_unbalance=True,
                              random_state=0,n_estimators=1714,
                              reg_alpha=9.99161,
                              reg_lambda=0.001593,colsample_bytree=0.3,
                              subsample=0.7,learning_rate=0.02,
                              max_depth=20,num_leaves=173,min_child_samples=244,
                              min_data_per_groups=7,
                              verbosity=-1),
}

x_train, x_test, y_train, y_test = train_test_split(datasetX,
                                                    Y,
                                                    test_size=0.2,
                                                    random_state=23333,
                                                    shuffle = True)
print('Baseline task')
print('Positive in y_test: ', y_test.sum()/y_test.shape[0])
print('Positive in y_train: ',y_train.sum()/y_train.shape[0])
if SMOTE_Process==True:
    from imblearn.over_sampling import SMOTE
    sm = SMOTE(random_state=42)
    x_train, y_train = sm.fit_resample(x_train, y_train)

compare_results = ['model_name','dataset', 'TN','FP','FN','TP','Pre','Rec','F0.5','F1',
                   'F2','AP','ROC_AUC','AUCPR','Brier','ACC']
plt.figure(figsize=(10,10))
for i in Models.keys():
    print(i)
    if i == 'Tab':
        y_train = y_train.values
        y_test = y_test.values
        x_train = x_train.values
        x_test = x_test.values

    model = Models[i]
    model.fit(x_train, y_train)
    # test set
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
    plt.plot(fpr, tpr, label='%s ROC (area = %0.3f)' % (i, ROC_AUC))
    compare_results = np.row_stack((compare_results,[i, 'test', tn, fp, fn, tp, precision, recall, 
                                   f0_5, f1, f2, AP, ROC_AUC, aucpr, brier, acc]))

import datetime
filename = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')+'_Compare_results_Minmax_'+str(minmaxnorm)+'_SMOTE_'+str(SMOTE_Process)+'.csv'
np.savetxt(filename, compare_results, delimiter=',', fmt='%s')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('1-Specificity(False Positive Rate)')
plt.ylabel('Sensitivity(True Positive Rate)')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig('ROC.png', dpi=330, bbox_inches='tight')
plt.show()   # Display

## LR coefficient
# nomogram
LR_model = Models['LR']
nomo_list = ['name', 'coef', 'min',	'max']
for var in range(x_train.shape[1]):
    nomo_list = np.row_stack((nomo_list, [x_train.columns[var],
                                          LR_model.coef_[0][var],
                                          x_train[x_train.columns[var]].min(),
                                          x_train[x_train.columns[var]].max()]
                                                 ))
nomo_list = pd.DataFrame(nomo_list[1:,:], columns=nomo_list[0,:])
nomo_list['coef'] = pd.to_numeric(nomo_list['coef'])
nomo_list['abs_values'] = nomo_list['coef'].abs()
nomo_list_sorted = nomo_list.sort_values(by='abs_values', ascending=False)

# 显示排序后的DataFrame
print(nomo_list)
LR_model.intercept_
LR_model.coef_


# # # model_best = Models['Cat']
if SHAP_Analysis==True:
    # import scipy as sp
    # partition_tree = shap.utils.partition_tree(datasetX)
    # plt.figure(figsize=(15, 6))
    # sp.cluster.hierarchy.dendrogram(partition_tree, labels=datasetX.columns, 
    #                                 leaf_font_size=10)
    # plt.title("Hierarchical Clustering Dendrogram")
    # # plt.xlabel("feature")
    # # plt.ylabel("distance")
    # plt.savefig('SHAP_result/Hierarchical Clustering Dendrogram.png', dpi=330, bbox_inches='tight')
    # plt.show()


    model_best = Models['LightGBM']
    y_pred_train  = model_best.predict(x_train)
    y_pred_train_proba  = model_best.predict_proba(x_train)[:, 1]
    Analysis_data = copy.deepcopy(x_train)
    Analysis_data['GT'] = y_train
    Analysis_data['Pre'] = y_pred_train    
    Analysis_data['Proba'] = y_pred_train_proba   
    Analysis_True = Analysis_data[Analysis_data['Pre']==Analysis_data['GT']]
    # x_train = x_train.reset_index(drop=True)
    # # https://catboost.ai/docs/concepts/shap-values
    # y_pred = model_best.predict(x_train)
    # For GBDT
    # explainer = shap.TreeExplainer(model_best, x_train, model_output='probability')
    # shap_values = explainer.shap_values(x_train)
    # For CAT
    # explainer = shap.TreeExplainer(model_best)
    # shap_values = explainer(x_train)
    # For ensemble lib https://shap.readthedocs.io/en/latest/example_notebooks/tabular_examples/model_agnostic/Census%20income%20classification%20with%20scikit-learn.html
    # For ensemble lib https://shap.readthedocs.io/en/latest/example_notebooks/tabular_examples/model_agnostic/Simple%20California%20Demo.html
    # def f(x):
    #     return model_best.predict_proba(x)[:, 1]
    # med = x_train.median().values.reshape((1, x_train.shape[1]))
    # explainer = shap.Explainer(f, med)
    # shap_values = explainer(x_train)
    
    def f(x):
        return model_best.predict_proba(x)[:, 1]    
    explainer = shap.Explainer(f, x_train)
    shap_values = explainer(x_train)

    
    # TP: 5 TN: 3, 9
    shap.plots.waterfall(shap_values[5], show=False)
    plt.savefig('SHAP_result/Instance_TP_SHAP.png', dpi=330, bbox_inches='tight')
    plt.show()
 
    shap.plots.waterfall(shap_values[3], show=False)
    plt.savefig('SHAP_result/Instance_TN_SHAP.png', dpi=330, bbox_inches='tight')
    plt.show()
    
    # https://shap.readthedocs.io/en/latest/example_notebooks/api_examples/plots/heatmap.html
    # https://shap.readthedocs.io/en/latest/generated/shap.plots.heatmap.html
    shap.plots.heatmap(shap_values[:500], show=False)
    plt.savefig('SHAP_result/heatmap.png', dpi=330, bbox_inches='tight')
    plt.show()
    
    
    shap.summary_plot(shap_values, x_train, plot_type="bar", show=False,
                      feature_names=datasetX.columns)
    plt.savefig('SHAP_result/Summary_bar_GBDT.png', dpi=330, bbox_inches='tight')
    plt.show()
    plt.close ()
    shap.summary_plot(shap_values, x_train, show=False,
                      feature_names=datasetX.columns)
    plt.savefig('SHAP_result/Summary_ori.png', dpi=330, bbox_inches='tight')
    plt.show()
    
    vals= np.abs(shap_values.values).mean(0)
    feature_importance = pd.DataFrame(list(zip(datasetX.columns,vals)),columns=['col_name','feature_importance_vals'])
    feature_importance.sort_values(by=['feature_importance_vals'],ascending=False,inplace=True)
    feature_importance.head()
    
    # explainer = shap.TreeExplainer(model_best)
    # shap_values = explainer(x_train)
    
    # shap.summary_plot(shap_values, x_train, plot_type="bar", show=False,
    #                   feature_names=datasetX.columns)

if ALE_Analysis == True:
    model_ale = ALE(model_best.predict, feature_names=datasetX.columns, 
                    target_names=['Violence'])
    model_exp = model_ale.explain(x_train.values) 
    for i in list(feature_importance['col_name'][:7]):
        plot_ale(model_exp, features=[i], n_cols=2, fig_kw={'figwidth':6, 'figheight': 3})
        path = 'SHAP_result/ALE' + i + '.png'
        plt.savefig(path, dpi=330, bbox_inches='tight')
    