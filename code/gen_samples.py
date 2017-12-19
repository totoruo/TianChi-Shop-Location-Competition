import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import pickle, os, re, operator, gc
from tqdm import tqdm
from multiprocessing import Pool
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import xgboost as xgb

behavior = pd.read_csv('../data/user_shop_behavior.csv')
behavior.loc[:, 'time_stamp'] = pd.to_datetime(behavior['time_stamp'])
shop = pd.read_csv('../data/shop_info.csv')
train = behavior.merge(shop[['shop_id', 'mall_id']], how='left', on='shop_id')
train['row_id'] = train.index
test = pd.read_csv('../data/test.csv')

res = None
train_samples = []
test_samples = []
for mall_id in tqdm(train.mall_id.unique()):
    sub_train = train[train.mall_id == mall_id]
    sub_test = test[test.mall_id == mall_id]

    train_set = []
    for index, row in sub_train.iterrows():
        wifi_dict = {}
        for wifi in row.wifi_infos.split(';'):
            bssid, signal, flag = wifi.split('|')
            wifi_dict[bssid] = int(signal)
        train_set.append(wifi_dict)
    
    test_set = []
    for index, row in sub_test.iterrows():
        wifi_dict = {}
        for wifi in row.wifi_infos.split(';'):
            bssid, signal, flag = wifi.split('|')
            wifi_dict[bssid] = int(signal)
        test_set.append(wifi_dict)
        
    v = DictVectorizer(sparse=False, sort=False)
    train_set = v.fit_transform(train_set)
    test_set = v.transform(test_set)
    train_set[train_set==0]=np.NaN
    test_set[test_set==0]=np.NaN
    sub_train = pd.concat([sub_train.reset_index(),pd.DataFrame(train_set)], axis=1)
    sub_test = pd.concat([sub_test.reset_index(),pd.DataFrame(test_set)], axis=1)
    
    lbl = LabelEncoder()
    lbl.fit(list(sub_train['shop_id'].values))
    sub_train['label'] = lbl.transform(list(sub_train['shop_id'].values))
    num_class=sub_train['label'].max()+1    
    feature = [x for x in sub_train.columns if x not in ['user_id','label','shop_id','time_stamp','mall_id','wifi_infos','row_id']]    
    params = {
            'objective': 'multi:softprob',
            'eta': 0.1,
            'max_depth': 9,
            'eval_metric': 'merror',
            'seed': 0,
            'num_class':num_class,
            'silent' : 1
            }
    num_rounds=60

    # params = {
    #         'objective': 'multi:softprob',
    #         'eta': 0.05,
    #         'max_depth': 7,
    #         'eval_metric': 'merror',
    #         'colsample_bytree': 0.6,
    #         'sub_sample': 0.6,
    #         'colsample_bylevel': 0.6,
    #         'seed': 0,
    #         'num_class':num_class,
    #         'silent' : 1
    #         }
    # num_rounds=300

    xgbtrain = xgb.DMatrix(sub_train[feature], sub_train['label'])
    xgbtest = xgb.DMatrix(sub_test[feature])
    watchlist = [ (xgbtrain,'train'), (xgbtrain, 'test') ]
    model = xgb.train(params, xgbtrain, num_rounds, watchlist, early_stopping_rounds=15, verbose_eval=False)
    preds=model.predict(xgbtest)
    for row, pred in enumerate(preds):
        row_id = sub_test['row_id'].iloc[row]
        predSorted = (-pred).argsort()
        for i in range(10):
            test_samples.append({'row_id':row_id,'shop_id':lbl.inverse_transform(predSorted[i]),'prob':pred[predSorted[i]]})
            if pred[predSorted[i]]>0.99:
                break
    preds=model.predict(xgbtrain)
    for row, pred in enumerate(preds):
        row_id = sub_train['row_id'].iloc[row]
        predSorted = (-pred).argsort()
        for i in range(10):
            train_samples.append({'row_id':row_id,'shop_id':lbl.inverse_transform(predSorted[i]),'prob':pred[predSorted[i]]})

train_samples.to_pickle(open('../data/train_samples_top10.pkl', 'wb'))
test_samples.to_pickle(open('../data/test_samples_top10.pkl', 'wb'))

