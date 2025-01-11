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
import optuna
from optuna.integration import CatBoostPruningCallback




def objective(trial: optuna.Trial) -> float:
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
    
    dataset_use = copy.deepcopy(dataset)
    datasetX = dataset_use.drop(['Experienced_violence', 'See_violence', 'WPVboth',
                                 'Experienced_violence_score',
                                 'See_violence_score'], axis=1)
    Y = dataset_use['WPVboth']
    Y = Y.mask(Y==2, 0)
    # 发现nan
    datasetX.loc[datasetX.isnull().any(axis=1), datasetX.isnull().any()]
    Y = Y.drop(datasetX[datasetX.isnull().T.any()].index.values) #label也删
    datasetX = datasetX.dropna(how='any') #删除有缺失值的行
    print(Y.mean())
    # step 2: Optuna
    data, target = datasetX.values, Y.values
    train_x, x_test, train_y, y_test = train_test_split(data, target, test_size=0.2,
                                                          random_state=23333,shuffle = True)
    
    # https://xgboost.readthedocs.io/en/stable/parameter.html
    # https://xgboost.readthedocs.io/en/stable/tutorials/param_tuning.html
    param = {
        'max_depth': trial.suggest_int('max_depth', 2, 10),
        'min_child_weight':  trial.suggest_int('min_child_weight', 0, 10),
        'gamma':  trial.suggest_int('gamma', 0, 20),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 1.0, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'eval_metric': trial.suggest_categorical('eval_metric', ['auc','aucpr','logloss']),
        'scale_pos_weight': trial.suggest_int('scale_pos_weight', 1, 100),
        'max_delta_step':  trial.suggest_int('max_delta_step', 0, 10),
    }
    
    model = XGBClassifier(**param)  
    
    model.fit(train_x,train_y,verbose=False)
    
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
    print('{0} TN: {1} FP: {2} FN: {3} TP: {4} | Pre: {5:.3f} Rec: {6:.3f} F0.5: {7:.3f} F1: {8:.3f} F2: {9:.3f} AP: {10:.3f}| ROC_AUC: {11:.3f} AUCPR: {12:.3f} Brier: {13:.4f} ACC: {14:.4f}'
      .format('Model', tn, fp, fn, tp, precision, recall, f0_5, f1, f2, AP, ROC_AUC, aucpr, brier, acc))
    return f1


if __name__ == "__main__":
    
  
    study = optuna.create_study(
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5), direction="maximize"
    )
    study.optimize(objective, n_trials=1000, timeout=600)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))


