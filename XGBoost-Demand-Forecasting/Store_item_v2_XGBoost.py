import numpy as np 
import pandas as pd
import datetime
import xgboost as xgb
from sklearn.model_selection import train_test_split
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
sample = pd.read_csv('../input/sample_submission.csv')
sample = pd.read_csv('../input/sample_submission.csv')
train['date'] = pd.to_datetime(train['date'])
test['date'] = pd.to_datetime(test['date'])

cal = calendar()
holidays = cal.holidays(start=train['date'].min(), end=test['date'].max())
#https://stackoverflow.com/questions/29688899/pandas-checking-if-a-date-is-a-holiday-and-assigning-boolean-value

train['month'] = train['date'].dt.month
train['day'] = train['date'].dt.dayofweek
train['year'] = train['date'].dt.year
train['week_of_year']  = train['date'].dt.weekofyear
train['weekday']  = train['date'].dt.dayofweek
train['holiday'] = train['date'].isin(holidays)*1

test['month'] = test['date'].dt.month
test['day'] = test['date'].dt.dayofweek
test['year'] = test['date'].dt.year
test['week_of_year']  = test['date'].dt.weekofyear
test['weekday']  = test['date'].dt.dayofweek
test['holiday'] = test['date'].isin(holidays)*1


col = [i for i in test.columns if i not in ['date','id']]
y = 'sales'

train_x, train_cv, y, y_cv = train_test_split(train[col],train[y], test_size=0.2, random_state=2018)

def XGB_regressor(train_X, train_y, test_X, test_y, feature_names=None, seed_val=2017, num_rounds=1600):
    param = {}
    param['objective'] = 'reg:linear'
    param['eta'] = 0.03
    #param['gamma'] = 0.1885914265710256
    param['max_depth'] = 8
    param['silent'] = 1
    param['eval_metric'] = 'mae'
    param['min_child_weight'] = 1
    param['subsample'] = 0.5947
    param['colsample_bytree'] = 0.66123
    param['seed'] = seed_val
    num_rounds = num_rounds

    plst = list(param.items())

    xgtrain = xgb.DMatrix(train_X, label=train_y)

    if test_y is not None:
        xgtest = xgb.DMatrix(test_X, label=test_y)
        watchlist = [ (xgtrain,'train'), (xgtest, 'test') ]
        model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=20)
    else:
        xgtest = xgb.DMatrix(test_X)
        model = xgb.train(plst, xgtrain, num_rounds)
        
    return model
model = XGB_regressor(train_X = train_x, train_y = y, test_X = train_cv, test_y = y_cv)
y_test = model.predict(xgb.DMatrix(test[col]), ntree_limit = model.best_ntree_limit)


sample['sales'] = y_test
sample.to_csv('simple_xgb_starter.csv', index=False)