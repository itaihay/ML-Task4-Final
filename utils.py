import ast
import typing
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn import metrics, preprocessing


def get_experiment_file_name(folder_name: str) -> str:
    return f'{folder_name}-{datetime.now().strftime("%Y%m%d-%H%M%S")}'


def auc_pr(y_true, y_pred, num_classes):
    precision = dict()
    recall = dict()
    auc = list()
    for i in range(num_classes):
        precision[i], recall[i], _ = metrics.precision_recall_curve(y_true[:, i],
                                                                    y_pred[:, i])
        auc.append(metrics.auc(recall[i], precision[i]))

    return np.average(auc)


def preprocess_data(df: pd.DataFrame) -> typing.Tuple[pd.DataFrame, pd.DataFrame, preprocessing.LabelEncoder]:
    df = df.dropna()
    y = df.iloc[:, -1]
    X = df.drop(df.columns[-1], axis=1)

    X_out = X.copy()

    # Preprocess X categorical data
    obj_cols = X_out.select_dtypes(include=['object', 'category']).columns
    if len(obj_cols) > 0:
        obj_data = X_out[obj_cols]
        obj_cols = obj_data.apply(preprocessing.LabelEncoder().fit_transform)
        X_out[obj_cols.columns] = obj_cols

    le_y = preprocessing.LabelEncoder()
    le_y.fit(y)
    y_out = le_y.transform(y)

    return X_out, y_out, le_y


def get_params_from_string(params: str) -> dict:
    tmp_df = pd.DataFrame([params], columns=['params'])
    tmp_df = tmp_df['params'].astype('str')
    tmp_df = tmp_df.apply(lambda x: ast.literal_eval(x))
    tmp_df = tmp_df.apply(pd.Series)

    return tmp_df.iloc[0].to_dict()

