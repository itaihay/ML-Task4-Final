import os
import time

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, precision_recall_fscore_support
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder

import utils

DATASETS_PATH = './classification_datasets'

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

all_scores = list()
for dataset_name in os.listdir(DATASETS_PATH):
    df = pd.read_csv(os.path.join(DATASETS_PATH, dataset_name))
    y_clean = df.iloc[:, -1]
    X_clean = df.drop(df.columns[-1], axis=1)

    le = preprocessing.LabelEncoder()
    le.fit(y_clean)
    y = le.transform(y_clean)
    X = X_clean.copy()

    curr_fold = 0
    kf = KFold(n_splits=10, random_state=10, shuffle=True)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]

        rf_clf = RandomForestClassifier()

        rs = RandomizedSearchCV(estimator=rf_clf,
                                param_distributions=random_grid,
                                n_iter=18,
                                cv=3,
                                verbose=2,
                                random_state=42,
                                n_jobs=-1)

        rs.fit(X_train, y_train)

        enc = OneHotEncoder(handle_unknown='ignore')
        enc.fit(y.reshape(-1, 1))

        y_pred = rs.best_estimator_.predict(X_test)
        y_prob = rs.best_estimator_.predict_proba(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob, multi_class='ovr', average='macro')

        fpr_arr, tpr_arr, _ = roc_curve(enc.transform(y_test.reshape(-1, 1)).toarray().ravel(), y_prob.ravel())
        fpr = np.average(fpr_arr)
        tpr = np.average(tpr_arr)

        precision, recall, _, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
        auc_pr_score = utils.auc_pr(enc.transform(y_test.reshape(-1, 1)).toarray(), y_prob, len(enc.categories_))

        X_1000 = pd.concat([X_test.iloc[0:1, :]] * 1000, ignore_index=True)
        start_predict = time.time()
        rs.best_estimator_.predict(X_1000)
        end_predict = time.time()

        predict_time = end_predict - start_predict

        start_fit = time.time()
        RandomForestClassifier(**rs.best_params_).fit(X_train, y_train)
        end_fit = time.time()

        fit_time = end_fit - start_fit

        all_scores.append((dataset_name,
                           'baseline',
                           curr_fold,
                           rs.best_params_,
                           accuracy,
                           tpr,
                           fpr,
                           precision,
                           auc,
                           auc_pr_score,
                           fit_time,
                           predict_time))

        curr_fold = curr_fold + 1

print(all_scores)
