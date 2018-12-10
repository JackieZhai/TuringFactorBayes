# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 10:15:21 2018

@author: dell
"""
from fetchJYDB import secu_loader
import numpy as np
import pdb
import pandas as pd
import time
def get_data(pool_id, csv_name):
    trading_data = pd.read_csv(csv_name) 
    close_price = trading_data.groupby('SecuCode')
    close_m1 = close_price['AdjClosePrice'].transform(lambda x:x.shift(-1))
    closes = close_price['AdjClosePrice'].apply(lambda x:x)
    trading_data['vwap'] =  10000*trading_data['RatioAdjustingFactor']*trading_data['TurnoverValue']/trading_data['TurnoverVolume']
    return_d1 = (close_m1 - closes)/(closes+0.001)
    trading_data['rate'] = return_d1
    trading_data['open'] = trading_data['AdjOpenPrice']
    trading_data['close'] = trading_data['AdjClosePrice']
    trading_data['high'] = trading_data['AdjHighPrice']
    trading_data['amount'] = trading_data['TurnoverValue']
    trading_data['low'] = trading_data['AdjLowPrice']
    trading_data['volume'] = trading_data['TurnoverVolume']
    trading_data['cap'] = trading_data['TotalMV']
    trading_data['sector'] = trading_data['FstIndustryCSI']
    trading_data['industry'] = trading_data['SndIndustryCSI']
    trading_data['subindustry'] = trading_data['TrdIndustryCSI']
    trading_data['ones'] = 1
    trading_day = trading_data['TradingDay'].drop_duplicates().sort_values().tolist()
    pool = get_pool(pool_id,trading_day[0])
    data_list = []
    for secucode in pool:
        data_list.append(trading_data.loc[trading_data['SecuCode']==int(secucode)])
    pd_trading_data = pd.concat(data_list)
    return pd_trading_data
def get_pool(pool_id, date):
    pool = secu_loader.get_IndexComponent2(pool_id,date)
#trading_data = get_data()
    return pool[pool_id].tolist()
if __name__== "__main__":
    pool_data = get_data('000300')
    print(pool_data)
"""
SHCI = ["000001", u"上证综指", 1]
hs300 = ["000300", u"沪深三百", 3145]
zz500 = ["000905", u"中证五百", 4978]
zz800 = ["000906", u"中证八百", 4982]
SmallCap = ["399005", u"中小盘指数", 7542]
GEMI = ["399006", u"创业板指", 11089]
SZCI = ["399106", u"深证综指", 1059]
ASCI = ["399317", u"A股指数", 3475]
"""