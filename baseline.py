import os
import time

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, precision_recall_fscore_support
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder

import utils

SCORES_COLUMNS = ['dataset_name', 'model', 'fold_n', 'best_params', 'accuracy', 'tpr', 'fpr', 'precision', 'auc-roc',
                 'auc-pr', 'fit_time_second', '1000_predict_time_seconds']

DATASETS_PATH = './classification_datasets'

RANDOM_GRID = {'n_estimators': [int(x) for x in np.linspace(start=200, stop=2000, num=10)],
               'max_features': ['auto', 'sqrt'],
               'max_depth': [int(x) for x in np.linspace(10, 110, num=11)],
               'min_samples_split': [2, 5, 10],
               'min_samples_leaf': [1, 2, 4],
               'bootstrap': [True, False]}

all_scores = list()
start_all_run_time = time.time()

# For each dataset
for dataset_name in os.listdir(DATASETS_PATH):
    print(dataset_name)
    df = pd.read_csv(os.path.join(DATASETS_PATH, dataset_name))

    X, y, encoder_y = utils.preprocess_data(df)

    # 10 fold cv
    curr_fold = 0
    kf = StratifiedKFold(n_splits=10, random_state=10, shuffle=True)
    for train_index, test_index in kf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]

        rf_clf = RandomForestClassifier()

        rs = RandomizedSearchCV(estimator=rf_clf,
                                param_distributions=RANDOM_GRID,
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
        precision, recall, _, _ = precision_recall_fscore_support(y_test, y_pred, warn_for=('precision', 'recall'))

        # If binary
        if len(enc.get_feature_names()) <= 2:
            auc = roc_auc_score(y_test, y_prob[:, 1])
            fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])
            auc_pr_score = metrics.auc(recall, precision)
        else:
            try:
                auc = roc_auc_score(y_test, y_prob, multi_class='ovo', average='macro')
            except ValueError as e:
                print(f"ERROR {dataset_name} - {y_test} - {y_prob}")
                auc = -1

            fpr, tpr, _ = roc_curve(enc.transform(y_test.reshape(-1, 1)).toarray().ravel(), y_prob.ravel())
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
                           np.average(tpr),
                           np.average(fpr),
                           np.average(precision),
                           auc,
                           auc_pr_score,
                           fit_time,
                           predict_time))

        curr_fold = curr_fold + 1

scores_df = pd.DataFrame(all_scores, columns=SCORES_COLUMNS)
scores_df.to_csv(f'./results/baseline/{utils.get_experiment_file_name("baseline")}.csv')
end_all_run_time = time.time()
print(all_scores)
print(end_all_run_time-start_all_run_time)

