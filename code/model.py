import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import pickle, os, re, operator, gc
from collections import Counter
from tqdm import tqdm
from multiprocessing import Pool
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb

train = pd.read_csv('../data/user_shop_behavior.csv')
train.loc[:, 'time_stamp'] = pd.to_datetime(train['time_stamp'])
train['hour'] = train.time_stamp.dt.hour
train['day'] = train.time_stamp.dt.day
train['weekday'] = train.time_stamp.dt.weekday

shop = pd.read_csv('../data/shop_info.csv')
le = LabelEncoder()
shop['category_id'] = le.fit_transform(shop['category_id'])
train['row_id'] = train.index

test = pd.read_csv('../data/test.csv')
test.loc[:, 'time_stamp'] = pd.to_datetime(test['time_stamp'])
test['hour'] = test.time_stamp.dt.hour
test['day'] = test.time_stamp.dt.day
test['weekday'] = test.time_stamp.dt.weekday

num_partitions = 15 #number of partitions to split dataframe
num_cores = 15 #number of cores on your machine

# df并行apply windows下可能会报错
def parallelize_dataframe(df, func):
    df_split = np.array_split(df, num_partitions)
    pool = Pool(num_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df

def best_wifi_parallelize(data):
    data[['best_wifi','best_wifi_strength']] = data.apply(get_best_wifi, axis=1)
    return data

def get_best_wifi(row):
    best_wifi_strength = -112
    for wifi in row.wifi_infos.split(';'):
        bssid, signal, flag = wifi.split('|')
        if int(signal) > best_wifi_strength:
            best_wifi = bssid
            best_wifi_strength = int(signal)
    return pd.Series([best_wifi, best_wifi_strength])

train = parallelize_dataframe(train, best_wifi_parallelize)
test = parallelize_dataframe(test, best_wifi_parallelize)

# 样本前21天提取特征
train_data = train[train.time_stamp.dt.day<=21]
train_data = train_data.merge(shop, how='left', on='shop_id')
test_data = train[(train.time_stamp.dt.day>10) & (train.time_stamp.dt.day<=31)]
test_data = test_data.merge(shop, how='left', on='shop_id')
train = train[(train.time_stamp.dt.day>21) & (train.time_stamp.dt.day<=31)]

# merge候选样本
train.rename(columns={'shop_id': 'shop_id_true'}, inplace=True)
samples = pickle.load(open('../data/train_samples_top10.pkl', 'rb'))
samples = samples.groupby('row_id').head(4)
train = train.merge(samples, 'left', 'row_id')
train = train.merge(shop, how='left', on='shop_id')
train['target'] = np.where(train['shop_id_true']==train['shop_id'], 1, 0)

samples = pickle.load(open('../data/test_samples_top10.pkl', 'rb'))
samples = samples.groupby('row_id').head(9)
test = test.merge(samples, 'left', 'row_id')
del test['mall_id']
test = test.merge(shop, how='left', on='shop_id')

def haversine_array(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    AVG_EARTH_RADIUS = 6371000  # in m
    lat = lat2 - lat1
    lng = lng2 - lng1
    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2
    h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))
    return h

def gen_feature(data, df):
    # shop
    shop_count = data.groupby(['shop_id']).size().reset_index().rename(columns={0:'shop_count'})
    shop_weekday_count = data.groupby(['shop_id','weekday']).size().reset_index().rename(columns={0:'shop_weekday_count'})
    shop_day_count = data.groupby(['shop_id','day']).size().reset_index()
    shop_day_count_var = shop_day_count.groupby(['shop_id'])[0].var().reset_index().rename(columns={0:'shop_day_count_var'})
    shop_hour_count = data.groupby(['shop_id','hour']).size().reset_index().rename(columns={0:'shop_hour_count'})
    
    # GPS
    data['la_dist'] = np.abs(data['latitude_x'] - data['latitude_y'])
    data['lo_dist'] = np.abs(data['longitude_x'] - data['longitude_y'])
    gps_hist = data[['shop_id', 'longitude_x', 'latitude_x', 'lo_dist', 'la_dist']].groupby('shop_id').mean().reset_index()
    gps_hist.columns = ['shop_id', 'hist_lo', 'hist_la', 'hist_lo_dist', 'hist_la_dist']

    # cate
    cate_count = data.groupby(['category_id']).size().reset_index().rename(columns={0:'cate_count'})
    cate_hour_count = data.groupby(['category_id','hour']).size().reset_index().rename(columns={0:'cate_hour_count'})
    
    # wifi
    data['shop_wifi_count'] = data.wifi_infos.str.count(';')+1
    shop_wifi_count_mean = data[['shop_id','shop_wifi_count']].groupby('shop_id').mean().reset_index().rename(columns={'shop_wifi_count':'shop_wifi_count_mean'})
    shop_wifi_count_sum = data[['shop_id','shop_wifi_count']].groupby('shop_id').sum().reset_index().rename(columns={'shop_wifi_count':'shop_wifi_count_sum'})
    best_wifi_count = data.groupby(['shop_id','best_wifi']).size().reset_index().rename(columns={0:'best_wifi_count'})

    # user
    user_ave_price = data[['user_id','price']].groupby('user_id').mean().reset_index().rename(columns={'price':'user_ave_price'})
    user_count = data.groupby('user_id').size().reset_index().rename(columns={0:'user_count'})
    user_shop_count = data.groupby(['user_id','shop_id']).size().reset_index().rename(columns={0:'user_shop_count'})
    
    # last shop sample
    last = data.sort_values(['shop_id','time_stamp'],ascending=False).drop_duplicates('shop_id')
    last = last[['shop_id', 'longitude_x', 'latitude_x', 'wifi_infos']]
    last.columns = ['shop_id', 'last_longitude', 'last_latitude', 'last_wifi_infos']
    
    # merge fature
    df = df.merge(shop_count, 'left', 'shop_id')
    df = df.merge(shop_day_count_var, 'left', 'shop_id')
    df = df.merge(shop_weekday_count, 'left', ['shop_id','weekday'])
    df = df.merge(shop_hour_count, 'left', ['shop_id','hour'])
    
    df = df.merge(gps_hist, 'left', 'shop_id')
    df['real_lo_dist'] = np.abs(df.longitude_y-df.longitude_x)
    df['real_la_dist'] = np.abs(df.latitude_y-df.latitude_x)
    df.loc[:, 'real_haversine_dist'] = haversine_array(df['latitude_x'].values, df['longitude_x'].values, df['latitude_y'].values, df['longitude_y'].values)
    df.loc[:, 'hist_haversine_dist'] = haversine_array(df['latitude_x'].values, df['longitude_x'].values, df['hist_la'].values, df['hist_lo'].values)

    df = df.merge(cate_count, 'left', 'category_id')
    df = df.merge(cate_hour_count, 'left', ['category_id', 'hour'])
    df['cate_hour_rate'] = df.cate_hour_count/df.cate_count

    df['wifi_count'] = df.wifi_infos.str.count(';')+1
    df = df.merge(shop_wifi_count_mean, 'left', 'shop_id')
    df = df.merge(shop_wifi_count_sum, 'left', 'shop_id')
    df['shop_wifi_count_rate'] = df.wifi_count/df.shop_wifi_count_mean
    df = df.merge(best_wifi_count, 'left', ['shop_id', 'best_wifi'])
    df['best_wifi_rate1'] = df.best_wifi_count/df.shop_count
    df['best_wifi_rate2'] = (df.best_wifi_count+100000)/(df.shop_count+200000)

    df = df.merge(user_count, 'left', 'user_id')
    df = df.merge(user_shop_count, 'left', ['user_id','shop_id'])
    df = df.merge(user_ave_price, 'left', 'user_id')
    df['price_rate'] = df.user_ave_price/df.price
    
    df = df.merge(last, 'left', 'shop_id')
    df.loc[:, 'last_dist_diff'] = haversine_array(df['latitude_x'].values, df['longitude_x'].values, df['last_latitude'].values, df['last_longitude'].values)
    
    return df

train = gen_feature(train_data, train)
test = gen_feature(test_data, test)

# 提取shop历史wifi信息
def extract_wifi_dict(data, tag):
    path = '../data/{}_wifi_dict'.format(tag)
    if os.path.exists(path):
        wifi_dict = pickle.load(open(path, 'rb'))
        return wifi_dict
    else:
        wifi_strength_mean = {}
        wifi_strength_max = {}
        wifi_sum_dict = {}
        mall_wifi_set = {}
        for shop_id in tqdm(data.shop_id.unique()):
            sub_data = data[data.shop_id == shop_id]
            wifi_strength_mean[shop_id] = {}
            wifi_strength_max[shop_id] = {}
            wifi_sum_dict[shop_id] = {}
            t = {}
            for i, row in sub_data.iterrows():
                if row.mall_id not in mall_wifi_set:
                    mall_wifi_set[row.mall_id] = set()
                for wifi in row.wifi_infos.split(';'):
                    bssid, signal, flag = wifi.split('|')
                    mall_wifi_set[row.mall_id].add(bssid)
                    if bssid not in t:
                        t[bssid] = []
                    t[bssid].append(int(signal))
            for bssid in t:
                wifi_strength_mean[shop_id][bssid] = np.mean(t[bssid])
                wifi_strength_max[shop_id][bssid] = np.max(t[bssid])
                wifi_sum_dict[shop_id][bssid] = len(t[bssid])

        wifi_dict = {}
        wifi_dict['wifi_sum_dict'] = wifi_sum_dict
        wifi_dict['wifi_strength_mean'] = wifi_strength_mean
        wifi_dict['wifi_strength_max'] = wifi_strength_max
        wifi_dict['mall_wifi_set'] = mall_wifi_set
        pickle.dump(wifi_dict, open(path, 'wb'))
        return wifi_dict

# 提取关于wifi的特征
apply_features = ['wifi_sum','wifi_num','wifi_connect_sum',
                 'wifi_dist_2_mean','wifi_dist_2_max','last_wifi_dist',
                  'best_wifi_dist_2_mean','best_wifi_dist_2_max',
                 'wifi_large_than_mean','wifi_large_than_max',
                 'wifi_dist_median','wifi_dist_mean','wifi_dist_std',
                 'top1_diff','top2_diff','top3_diff','top4_diff','top5_diff','top6_diff','top7_diff','top8_diff',]

def wifi_info_parallelize(data):
    data[apply_features] = data.apply(wifi_info, axis=1)
    return data

def wifi_info(row):       
    wifi_sum = 0; wifi_num = 0; wifi_connect_sum = 0
    wifi_dist_2_mean = 0; wifi_dist_2_max = 0; last_wifi_dist = 0
    wifi_large_than_mean = 0; wifi_large_than_max = 0
    
    if row.shop_id not in wifi_sum_dict:
        return pd.Series([np.NaN]*len(apply_features))

    last_wifi = {}
    for wifi in row.last_wifi_infos.split(';'):
        bssid, signal, flag = wifi.split('|')
        last_wifi[bssid] = int(signal)
        
    wifi_dist_list = []
    wifi_dict = {}
    for wifi in row.wifi_infos.split(';'):
        bssid, signal, flag = wifi.split('|')
        signal = int(signal)
        last_wifi_dist += np.abs(signal-last_wifi.get(bssid,-130))
        if bssid in mall_wifi_set[row.mall_id]:
            if flag == 'true':
                wifi_connect_sum = wifi_sum_dict[row.shop_id].get(bssid,0)
            if bssid in wifi_sum_dict[row.shop_id]:
                wifi_num += 1
                wifi_sum += wifi_sum_dict[row.shop_id][bssid]
                wifi_dict[bssid] = signal
                s = wifi_strength_mean[row.shop_id][bssid]
                wifi_dist_2_mean += np.abs(signal-s)
                wifi_dist_list.append(signal-s)
                if signal > s:
                    wifi_large_than_mean += 1
                    
                s = wifi_strength_max[row.shop_id][bssid]
                wifi_dist_2_max += np.abs(signal-s)
                if signal > s:
                    wifi_large_than_max += 1
            else:                     
                wifi_dist_2_mean += np.abs(signal+130)
                wifi_dist_2_max += np.abs(signal+130)
                wifi_dist_list.append(np.abs(signal+130))

                
    best_wifi_dist_2_mean = np.abs(wifi_strength_mean[row.shop_id].get(row.best_wifi, np.NaN) - row.best_wifi_strength)
    best_wifi_dist_2_max = np.abs(wifi_strength_max[row.shop_id].get(row.best_wifi, np.NaN) - row.best_wifi_strength)
    wifi_dict = sorted(wifi_dict.items(), key = lambda b:b[1], reverse = True)
    x = {}
    for i in range(8):
        if i >= len(wifi_dict):
            x['top{}_diff'.format(i+1)] = np.NaN
        else:
            bssid, signal = wifi_dict[i]
            x['top{}_diff'.format(i+1)] = signal-wifi_strength_mean[row.shop_id][bssid]
        
    if len(wifi_dist_list) == 0:
        wifi_dist_median = np.NaN
        wifi_dist_mean = np.NaN
        wifi_dist_std = np.NaN
    else:
        wifi_dist_median = np.median(wifi_dist_list)
        wifi_dist_mean = np.mean(wifi_dist_list)
        wifi_dist_std = np.std(wifi_dist_list)
        
    return_list =   [wifi_sum,wifi_num,wifi_connect_sum,
                     wifi_dist_2_mean,wifi_dist_2_max,last_wifi_dist,
                     best_wifi_dist_2_mean,best_wifi_dist_2_max,
                     wifi_large_than_mean,wifi_large_than_max,
                     wifi_dist_median,wifi_dist_mean,wifi_dist_std,
                     x['top1_diff'],x['top2_diff'],x['top3_diff'],x['top4_diff'],x['top5_diff'],
                     x['top6_diff'],x['top7_diff'],x['top8_diff'],]
    
    return pd.Series(return_list)

wifi_dict = extract_wifi_dict(train_data, 'train')
wifi_sum_dict = wifi_dict['wifi_sum_dict']
wifi_strength_max = wifi_dict['wifi_strength_max']
wifi_strength_mean = wifi_dict['wifi_strength_mean']
mall_wifi_set = wifi_dict['mall_wifi_set']
train = parallelize_dataframe(train, wifi_info_parallelize)
train['wifi_sum_rate'] = train['wifi_sum'] / train['shop_count']

# online
wifi_dict = extract_wifi_dict(test_data, 'test')
wifi_sum_dict = wifi_dict['wifi_sum_dict']
wifi_strength_max = wifi_dict['wifi_strength_max']
wifi_strength_mean = wifi_dict['wifi_strength_mean']
mall_wifi_set = wifi_dict['mall_wifi_set']
test = parallelize_dataframe(test, wifi_info_parallelize)
test['wifi_sum_rate'] = test['wifi_sum'] / test['shop_count']

le = LabelEncoder()
train['mall_id_labeled'] = le.fit_transform(train['mall_id'])
test['mall_id_labeled'] = le.transform(test['mall_id'])


# rank模型
train_group = train.groupby('row_id', sort=False).size().reset_index()
test_group = test.groupby('row_id', sort=False).size().reset_index()
features = ['latitude_x','longitude_x','real_la_dist','real_lo_dist','hist_haversine_dist','hist_la_dist','hist_lo_dist','real_haversine_dist',
            'category_id','cate_count','cate_hour_count', 'cate_hour_rate','wifi_num','wifi_connect_sum','wifi_sum_rate',
            'wifi_dist_2_mean','wifi_sum','best_wifi_rate1','best_wifi_rate2','best_wifi_strength','shop_wifi_count_sum','best_wifi_count',
            'shop_wifi_count_mean','user_count','user_shop_count','user_ave_price','price_rate','mall_id_labeled',
            'shop_day_count_var','shop_weekday_count','best_wifi_dist_2_mean','last_dist_diff','last_wifi_dist','shop_hour_count','best_wifi_dist_2_max',
            'wifi_dist_median','wifi_dist_mean','wifi_dist_std','wifi_large_than_max','wifi_dist_2_max','wifi_large_than_mean',
            'top2_diff','top3_diff','top4_diff','top5_diff','top6_diff','top7_diff','top8_diff',
           ]
dtrain = lgb.Dataset(train[features].values, label=train['target'].values)
dtrain.set_group(train_group[0])
dtest = lgb.Dataset(test[features].values)
dtest.set_group(test_group[0])
params = {'learning_rate': 0.1, 
          'max_depth': -1,
          'objective': 'lambdarank', 
          'metric': 'binary_error',
         }
model = lgb.train(params, dtrain, num_boost_round=1500, verbose_eval=200)

pred = model.predict(test[features].values)
res = test.copy()
res['pred'] = pred
res['fuck'] = res['pred']*1.2 + res['prob']
res = res.sort_values(['row_id', 'fuck'], ascending=False).drop_duplicates('row_id')
res[['row_id', 'shop_id']].to_csv('../data/sub.csv', index=False)