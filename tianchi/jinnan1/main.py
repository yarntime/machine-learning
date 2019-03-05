# encoding: utf-8
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import catboost as ctb
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from scipy import sparse
import warnings
import time
import sys
import os
import re
from sklearn.metrics import mean_squared_error
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")

def timeTranSecond(t):
    try:
        t,m,s=t.split(":")
    except:
        if t=='1900/1/9 7:00':
            return (23*3600)/3600
        elif t=='1900/1/1 2:30':
            return (21*3600+30*60)/3600
        elif t=='1/21/1900 0:00':
            return (21*3600+30*60)/3600
        elif t=='1900/1/22 0:00':
            return (22*3600)/3600
        elif t=='1/12/1900 0:00':
            return (12*3600)/3600
        elif t==-1:
            return -1
        else:
            return 0

    try:
        tm = (int(t)*3600+int(m)*60+int(s))/3600
    except:
        return (30*60)/3600

    return tm

def getDuration(se):
    try:
        sh,sm,eh,em=re.findall(r"\d+\.?\d*",se)
    except:
        if se == -1:
            return -1

    try:
        if int(sh)>int(eh):
            tm = (int(eh)*3600+int(em)*60-int(sm)*60-int(sh)*3600)/3600 + 24
        else:
            tm = (int(eh)*3600+int(em)*60-int(sm)*60-int(sh)*3600)/3600
    except:
        if se=='19:-20:05':
            return 1
        elif se=='15:00-1600':
            return 1

    return tm

def lgbFeval(preds, lgbtrain):
    label = lgbtrain.get_label()
    score = mean_squared_error(label,preds)*0.5
    return 'lgbFeval',score,False

def xgbFeval(preds, xgbtrain):
    label = xgbtrain.get_label()
    score = mean_squared_error(label,preds)*0.5
    return 'xgbFeval',score

def train(train_path, test_path, output_path):
    train = pd.read_csv(train_path, encoding = 'gb18030')
    test = pd.read_csv(test_path, encoding = 'gb18030')
    test_id = test[u'样本id']

    for df in [train, test]:
        df.drop(['B3', 'B13', 'A13', 'A18', 'A23'], axis=1, inplace=True)

    good_cols = list(train.columns)
    for col in train.columns:
        rate = train[col].value_counts(normalize=True, dropna=False).values[0]
        if rate > 0.9:
            good_cols.remove(col)

    train = train[(train[u'收率'] > 0.87) & (train['B14'] > 40) & (train['A6'] < 50)]

    good_cols.append('A1')
    good_cols.append('A3')
    good_cols.append('A4')

    good_cols.remove('sample_id')
    train = train[good_cols]
    good_cols.remove(u'收率')
    test  = test[good_cols]

    target = train[u'收率']
    del train[u'收率']
    data = pd.concat([train,test],axis=0,ignore_index=True)
    data = data.fillna(-1)

    for f in ['A5','A7','A9','A11','A14','A16','A24','A26','B5','B7']:
        try:
            data[f] = data[f].apply(timeTranSecond)
        except:
            continue
    for f in ['A20','A28','B4','B9','B10','B11']:
        data[f] = data.apply(lambda df: getDuration(df[f]), axis=1)

    categorical_columns = [f for f in data.columns]
    numerical_columns = [f for f in data.columns if f not in categorical_columns]

    for f in ['B14']:
        data[f+'_median'] = data[f].median()
        data[f+'_std'] = data[f].std()
        data[f+'_max'] = data[f].max()
        data[f+'_min'] = data[f].min()
        data[f+'**2'] = data[f]**2

    data['b14/a1_a3_a4_a19_b1_b12'] = data['B14']/(data['A1']+data['A3']+data['A4']+data['A19']+data['B1']+data['B12'])
    data['b14_a1_a3_a4_a19_b1_b12'] = data['B14'] + data['A1']+data['A3']+data['A4']+data['A19']+data['B1']+data['B12']
    data['b14*a1_a3_a4_a19_b1_b12'] = data['B14'] * (data['A1']+data['A3']+data['A4']+data['A19']+data['B1']+data['B12'])

    numerical_columns.append('b14/a1_a3_a4_a19_b1_b12')
    numerical_columns.append('b14_a1_a3_a4_a19_b1_b12')
    numerical_columns.append('b14*a1_a3_a4_a19_b1_b12')

    data['b14*b12'] = data['B14'] * data['B12']
    numerical_columns.append('b14*b12')

    data['b14/b1'] = data['B14'] / data['B1']
    numerical_columns.append('b14/b1')

    data['b14*a19'] = data['B14'] * data['A19']
    numerical_columns.append('b14*a19')

    data['b14/a4'] = data['B14'] / data['A4']
    numerical_columns.append('b14/a4')

    data['b14+a4'] = data['B14'] + data['A4']
    numerical_columns.append('b14+a4')

    data['B11*B14'] = data['B11'] * data['B14']
    numerical_columns.append('B11*B14')

    data['A7*A8'] = data['A7'] * data['A8']
    numerical_columns.append('A7*A8')

    data['A9*A10'] = data['A10'] * data['A9']
    numerical_columns.append('A9*A10')

    data['A10*A11'] = data['A10'] * data['A11']
    numerical_columns.append('A10*A11')

    data['A16*A17'] = data['A16'] * data['A17']
    numerical_columns.append('A16*A17')

    data['A25*A26'] = data['A25'] * data['A26']
    numerical_columns.append('A25*A26')

    data['B10*B11'] = data['B10'] * data['B11']
    numerical_columns.append('B10*B11')

    data['B12*B14'] = data['B12'] * data['B14']
    numerical_columns.append('B12*B14')

    data['A5*A7'] = data['A5'] * data['A7']
    numerical_columns.append('A5*A7')

    data['A9*A11'] = data['A9'] * data['A11']
    numerical_columns.append('A9*A11')

    data['A19*A21'] = data['A19'] * data['A21']
    numerical_columns.append('A19*A21')

    data['B8*B10'] = data['B8'] * data['B10']
    numerical_columns.append('B8*B10')

    data['B10*B12'] = data['B10'] * data['B12']
    numerical_columns.append('B10*B12')

    data['A11*A14'] = data['A11'] * data['A14']
    numerical_columns.append('A11*A14')

    data['A12*A15'] = data['A12'] * data['A15']
    numerical_columns.append('A12*A15')

    data['A11*A15'] = data['A11'] * data['A15']
    numerical_columns.append('A11*A15')

    data['A16*A19'] = data['A16'] * data['A19']
    numerical_columns.append('A16*A19')

    data['A19*A22'] = data['A19'] * data['A22']
    numerical_columns.append('A19*A22')

    del data['A1']
    del data['A3']
    del data['A4']
    categorical_columns.remove('A1')
    categorical_columns.remove('A3')
    categorical_columns.remove('A4')

    for f in categorical_columns:
        data[f] = data[f].map(dict(zip(data[f].unique(), range(0, data[f].nunique()))))
    train = data[:train.shape[0]]
    test  = data[train.shape[0]:]

    train['target'] = target
    train['intTarget'] = pd.cut(train['target'], 5, labels=False)
    train = pd.get_dummies(train, columns=['intTarget'])
    li = ['intTarget_0.0','intTarget_1.0','intTarget_2.0','intTarget_3.0','intTarget_4.0']
    mean_columns = []
    for f1 in categorical_columns:
        cate_rate = train[f1].value_counts(normalize=True, dropna=False).values[0]
        if cate_rate < 0.90:
            for f2 in li:
                col_name = 'B14_to_'+f1+"_"+f2+'_mean'
                mean_columns.append(col_name)
                order_label = train.groupby([f1])[f2].mean()
                train[col_name] = train['B14'].map(order_label)
                miss_rate = train[col_name].isnull().sum() * 100 / train[col_name].shape[0]
                if miss_rate > 0:
                    train = train.drop([col_name], axis=1)
                    mean_columns.remove(col_name)
                else:
                    test[col_name] = test['B14'].map(order_label)

    train.drop(li+['target'], axis=1, inplace=True)

    X_train = train[mean_columns+numerical_columns].values
    X_test = test[mean_columns+numerical_columns].values
    # one hot
    enc = OneHotEncoder()
    for f in categorical_columns:
        enc.fit(data[f].values.reshape(-1, 1))
        X_train = sparse.hstack((X_train, enc.transform(train[f].values.reshape(-1, 1))), 'csr')
        X_test = sparse.hstack((X_test, enc.transform(test[f].values.reshape(-1, 1))), 'csr')

    y_train = target.values

    param = {'num_leaves': 120,
             'min_data_in_leaf': 30,
             'objective':'regression',
             'max_depth': -1,
             'learning_rate': 0.01,
             "min_child_samples": 30,
             "boosting": "gbdt",
             "feature_fraction": 0.9,
             "bagging_freq": 1,
             "bagging_fraction": 0.9 ,
             "bagging_seed": 11,
             "metric": 'mse',
             "lambda_l1": 0.1,
             "verbosity": -1}

    folds = KFold(n_splits=5, shuffle=True, random_state=2018)
    oof_lgb = np.zeros(len(train))
    predictions_lgb = np.zeros(len(test))

    for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train, y_train)):
        print("fold n°{}".format(fold_+1))
        trn_data = lgb.Dataset(X_train[trn_idx], y_train[trn_idx])
        val_data = lgb.Dataset(X_train[val_idx], y_train[val_idx])

        num_round = 10000
        clf = lgb.train(param, trn_data, num_round, feval=lgbFeval, valid_sets = [trn_data, val_data], verbose_eval=200, early_stopping_rounds = 100)
        oof_lgb[val_idx] = clf.predict(X_train[val_idx], num_iteration=clf.best_iteration)

        predictions_lgb += clf.predict(X_test, num_iteration=clf.best_iteration) / folds.n_splits

    xgb_params = {'eta': 0.005, 'max_depth': 10, 'subsample': 0.8, 'colsample_bytree': 0.8,
                  'objective': 'reg:linear', 'eval_metric': 'rmse', 'silent': True, 'nthread': 16}

    folds = KFold(n_splits=5, shuffle=True, random_state=2018)
    oof_xgb = np.zeros(len(train))
    predictions_xgb = np.zeros(len(test))

    for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train, y_train)):
        print("fold n°{}".format(fold_+1))
        trn_data = xgb.DMatrix(X_train[trn_idx], y_train[trn_idx])
        val_data = xgb.DMatrix(X_train[val_idx], y_train[val_idx])

        watchlist = [(trn_data, 'train'), (val_data, 'valid_data')]
        clf = xgb.train(dtrain=trn_data, num_boost_round=20000, feval = xgbFeval, evals=watchlist, early_stopping_rounds=200, verbose_eval=100, params=xgb_params)
        oof_xgb[val_idx] = clf.predict(xgb.DMatrix(X_train[val_idx]), ntree_limit=clf.best_ntree_limit)
        predictions_xgb += clf.predict(xgb.DMatrix(X_test), ntree_limit=clf.best_ntree_limit) / folds.n_splits

    train_stack = np.vstack([oof_lgb,oof_xgb]).transpose()
    test_stack = np.vstack([predictions_lgb, predictions_xgb]).transpose()

    folds_stack = RepeatedKFold(n_splits=5, n_repeats=2, random_state=4590)
    oof_stack_xgb = np.zeros(train_stack.shape[0])
    predictions = np.zeros(test_stack.shape[0])

    for fold_, (trn_idx, val_idx) in enumerate(folds_stack.split(train_stack,target)):
        print("fold {}".format(fold_))
        trn_data, trn_y = train_stack[trn_idx], target.iloc[trn_idx].values
        val_data, val_y = train_stack[val_idx], target.iloc[val_idx].values

        clf_3 = xgb.XGBRegressor()
        clf_3.fit(trn_data, trn_y)

        oof_stack_xgb[val_idx] = clf_3.predict(val_data)
        predictions += clf_3.predict(test_stack) / 10

    print("LGB score: {}".format((mean_squared_error(oof_lgb, target)*0.5)))
    print("XGB score: {}".format((mean_squared_error(oof_xgb, target)*0.5)))
    print("STACK score: {}".format((mean_squared_error(target.values, oof_stack_xgb)*0.5)))

    sub_df = pd.DataFrame()
    sub_df[0] = test_id
    sub_df[1] = predictions
    sub_df[1] = sub_df[1].apply(lambda x:round(x, 3))
    sub_df.to_csv(output_path, index=False, header=None)

if __name__ == '__main__':
    train("./data/jinnan_round1_train_20181227.csv", "./data/optimize.csv", "submit_optimize.csv")
    train("./data/jinnan_round1_train_20181227.csv", "./data/FuSai.csv", "submit_FuSai.csv")
