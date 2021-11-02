from typing import List, Dict, Union
import numpy as np

# Speed sklearn lib
from sklearnex import patch_sklearn
patch_sklearn()

from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import lightgbm as lgb
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, roc_curve, auc


def roc_10p_function(y, y_pred):
    if type(y_pred[0]) == np.ndarray:
        y_pred = y_pred[:, 0]
    # calculation of false positive and true positive rates for all classification thresholds.
    fpr, tpr, threshold = roc_curve(y, y_pred)
    # We keep only what is inferior to 10%
    fpr_10p, tpr_10p = fpr[fpr < 0.1], tpr[fpr < 0.1]
    if len(fpr_10p) > 1:
        return auc(fpr_10p, tpr_10p)
    return 0


# Grid-search
def gridSearch(x, y, model, parameters: Dict[str, Union[List[str], List[int]]], cv=4) -> dict:
    roc_10p = make_scorer(roc_10p_function, greater_is_better=True, needs_proba=True)
    scoring = {"roc_10%": roc_10p}

    grid_search = GridSearchCV(estimator=model,
                               param_grid=parameters,
                               scoring=scoring,
                               cv=cv,
                               n_jobs=-1,
                               verbose=1,
                               refit='roc_10%',
                               # pre_dispatch = 5
                               )
    grid_search.fit(x, y)
    best_parameters = grid_search.best_params_

    return best_parameters


def logistic(X_train: np.ndarray, X_test: np.ndarray, Y_train: np.ndarray) -> np.ndarray:
    log = linear_model.LogisticRegression()
    log.fit(X_train, Y_train)
    Y_pred = log.predict_proba(X_test)[:, 1]
    return Y_pred


def randomForest(X_train: np.ndarray, X_test: np.ndarray, Y_train: np.ndarray, grid_search=True) -> np.ndarray:
    if grid_search == True:
        rfr = RandomForestClassifier(n_jobs=-1)
        parameters = {'n_estimators': [500, 1000, 2000, 3000],
                      'min_samples_leaf': [1, 5],
                      'criterion': ['gini', 'entropy'],
                      'max_depth': [None, 1],
                      'min_samples_split': [2, 5, 10],
                      'max_features': ['log2', 'sqrt']
                      }

        parameters_tuned = gridSearch(X_train, Y_train, rfr, parameters)
        print(parameters_tuned)
        rfr = RandomForestClassifier(n_jobs=-1, criterion=parameters_tuned['criterion'],
                                     max_depth=parameters_tuned['max_depth'],
                                     min_samples_leaf=parameters_tuned['min_samples_leaf'],
                                     max_features=parameters_tuned['max_features'],
                                     min_samples_split=parameters_tuned['min_samples_split'],
                                     n_estimators=parameters_tuned['n_estimators'])

    else:
        rfr = RandomForestClassifier(n_jobs=-1, criterion='entropy', max_depth=None, min_samples_leaf=1,
                                     max_features='sqrt', min_samples_split=5, n_estimators=2000)
    rfr.fit(X_train, Y_train)
    Y_pred = rfr.predict_proba(X_test)[:, 1]
    return Y_pred


def xgBoost(X_train: np.ndarray, X_test: np.ndarray, Y_train: np.ndarray) -> np.ndarray:
    xgb = XGBClassifier(max_depth=15, n_jobs=-1)
    xgb.fit(X_train, Y_train)
    Y_pred = xgb.predict_proba(X_test)[:, 1]
    return Y_pred


def lightGbm(X_train: np.ndarray, X_test: np.ndarray, Y_train: np.ndarray, grid_search=True) -> np.ndarray:
    if grid_search == True:
        lgbm = lgb.LGBMClassifier()

        parameters = {
            'num_leaves': [5, 10, 15, 20],
            'min_data_in_leaf': [25, 50, 100, 200],
            'max_depth': [5, 10, 15, 20],
            'learning_rate': [0.1, 0.05, 0.01, 0.001],
            'n_estimators': [100, 500, 1000, 1500],
        }

        parameters_tuned = gridSearch(X_train, Y_train, lgbm, parameters)
        print(parameters_tuned)
        lgbm = lgb.LGBMClassifier(n_jobs=-1, num_leaves=parameters_tuned['num_leaves'],
                                  min_data_in_leaf=parameters_tuned['min_data_in_leaf'],
                                  max_depth=parameters_tuned['max_depth'],
                                  learning_rate=parameters_tuned['learning_rate'],
                                  n_estimators=parameters_tuned['n_estimators'])

    else:
        lgbm = lgb.LGBMClassifier(objective='binary', n_jobs=-1, learning_rate=0.05, max_bin=16, max_depth=15,
                                  min_data_in_leaf=100,
                                  n_estimators=500, num_leaves=15, reg_alpha=0, reg_lambda=0, subsample=0.001)
    lgbm.fit(X_train, Y_train)
    Y_pred = lgbm.predict_proba(X_test)[:, 1]
    return Y_pred


def naiveBayes(X_train: np.ndarray, X_test: np.ndarray, Y_train: np.ndarray) -> np.ndarray:
    gnb = GaussianNB()
    gnb.fit(X_train, Y_train)
    Y_pred = gnb.predict_proba(X_test)[:, 1]
    return Y_pred
