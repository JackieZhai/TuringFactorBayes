"""
Modified on Thu Dec 13 15:49 2018
@author: zhaihao

TODO:
1. 定义sum(X,N)/N中的两个N为同一个const_node
2. 定义regebeta(X,sequence(N),N)中的两个N为同一个const_node
3. 限定sma(X,N,M)中的N>M
4. 限定std(X,N)中的N>=2
5. 限定corr(X,N)中的N>=2
6. 进一步实现benchmark相关(galpha075)
"""

from utils import *
from formula import formula_to_tree
from sklearn.linear_model import LinearRegression
from scipy.stats import linregress
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from numbers import Number
import math
from math import floor
import numpy as np
import pandas as pd
import copy
import pdb
import tensorflow as tf
from tensorflow.nn import sigmoid_cross_entropy_with_logits as scewl
from tensorflow import reduce_mean as rm
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import global_list as gl
from operator import itemgetter

# 获取列表的第二个元素
def takeSecond(elem):
    return elem[1]

class Calculator():

    __tick_to_unit_time = 60

    def __init__(self, data):
        self.data = data

    def __del__(self):
        del self.data

    def delete(self):
        self.do_deletion(self.tree)

    def do_deletion(self, tree):
        if isinstance(tree, node):
            for child in tree.children:
                if child.data is not None:
                    self.do_deletion(child)
        if isinstance(tree, node) or isinstance(tree, paramnode):
            tree.data = None

    def get_paramnode(self, tree):
        list = []
        if isinstance(tree, paramnode):
            return [tree]
        elif isinstance(tree, constnode):
            return []
        else:
            for children in tree.children:
                result = self.get_paramnode(children)
                list.extend(result)
        return list

    def set_tree(self, tree):
        self.tree = tree

    def calculate_features(self, tree):
        # 把传入的树设置为self.tree
        if isinstance(tree, str):
            tree = formula_to_tree(tree)
        self.set_tree(tree)

        # 把所有的变量节点替换为实际数据
        paramnode_list = self.get_paramnode(self.tree)
        for leaf in paramnode_list:
            leaf.data = self._get_var(leaf.name)

        # 递归调用计算至跟根节点数据
        if self.tree.data is None:
            self.tree.data = self.do_calculation(self.tree)

        # 返回的数据去除inf
        self.tree.data = self.tree.data.replace([np.inf, -np.inf], np.nan)
#        pdb.set_trace()
        return self.tree.data

    def _get_var(self, token):
        if token == 'ret':
            return (self.data['close'] - self.data['open']) / self.data['open']

        elif token == 'dtm':
            # (OPEN<=DELAY(OPEN,1)?0:MAX((HIGH-OPEN),(OPEN-DELAY(OPEN,1))))
            dtm = np.maximum(self.data['high'] - self.data['open'], 
                             self.data['open'] - self.data['open'].shift(self._to_int(1)))
            dtm[self.data['open'] <= self.data['open'].shift(self._to_int(1))] = 0
            return dtm

        elif token == 'dbm':
            # (OPEN>=DELAY(OPEN,1)?0:MAX((OPEN-LOW),(OPEN-DELAY(OPEN,1))))
            dbm = np.maximum(self.data['open'] - self.data['low'], 
                             self.data['open'] - self.data['open'].shift(self._to_int(1)))
            dbm[self.data['open'] <= self.data['open'].shift(self._to_int(1))] = 0
            return dbm

        elif token == 'tr':
            # MAX(MAX(HIGH-LOW,ABS(HIGH-DELAY(CLOSE,1))),ABS(LOW-DELAY(CLOSE,1)))
            max1 = self.data['high'] - self.data['low']
            max2 = np.abs(self.data['high'] - self.data['close'].shift(1))
            max3 = np.abs(self.data['low'] - self.data['close'].shift(1))
            tr = np.maximum(np.maximum(max1, max2), max3)
            return tr
            
        elif token == 'hd':
            # HIGH-DELAY(HIGH,1)
            return self.data['high'] - self.data['high'].shift(self._to_int(1))
        
        elif token == 'ld':
            # DELAY(LOW,1)-LOW
            return self.data['low'].shift(self._to_int(1)) - self.data['low']
        elif token == 'indclass.sector':

            return [self.data['sector'],self.data['cap']]
        elif token == 'indclass.industry':

            return [self.data['industry'],self.data['cap']]
        elif token == 'indclass.subindustry':
#            pdb.set_trace()
            return [self.data['subindustry'],self.data['cap']]

        elif token in self.data:

            return self.data[token].copy()

        else:

            raise ValueError('%s is not a valid variable name' % token)


    def _to_constant_df(self, num):
        """
        guard against single num 
        turn a single number into a constant dataframe whose index is consistent with others
        """
        if isinstance(num, pd.DataFrame):
#            pdb.set_trace()
            return num
        else:
            return self.data['ones'].copy() * num

    def _to_int(self, num):
        """
        This is to be used as a wrapper of input window size (i.e. past d days)
        This is to guard against decimal window size
        """
        assert isinstance(num, Number), 'Is not number in _to_int'
        return floor(self.__tick_to_unit_time * num)

    def do_calculation(self, tree):
        # 向下递归
        for child in tree.children:
            if child.data is None:
                child.data = self.do_calculation(child)

        ops = [tree.children[i].data for i in range(len(tree.children))]

        # 计算该层
        if tree.name == 'max':
            # MAX(A, B): 在 A,B 中选择最大的数
            return np.maximum(self._to_constant_df(ops[0]), self._to_constant_df(ops[1]))
        
        elif tree.name == 'rank':
#            print(ops[0].rank(axis=1, pct=True, ascending=True))
#            pdb.set_trace()
            # RANK(A): 向量 A 升序排序
            # by default the smallest ranks first: rank=0
            return ops[0].rank(axis=1, pct=True, ascending=True)
    
        elif tree.name == 'min':
            # MIN(A, B): 在 A,B 中选择最小的数
#            pdb.set_trace()
            return np.minimum(self._to_constant_df(ops[0]), self._to_constant_df(ops[1]))
        elif tree.name == 'signedpower':
            return ops[0]**ops[1]
        elif tree.name == 'std':
            # STD(A, n)): 序列A过去n个标准差
            return ops[0].rolling(self._to_int(ops[1])).std()
        
        elif tree.name == 'corr':
            # CORR(A, B, n): 序列A、B过去n个的相关系数
            num = self._to_int(ops[2])
            val = ops[0].rolling(num,min_periods=1).corr(ops[1].rolling(num,min_periods=1))
            val[np.isinf(val)] = 0
#            pdb.set_trace()
            return val
        
        elif tree.name == 'delta':
            # DELTA(A, n): A_i - A_{i-n}
#            pdb.set_trace()
            return ops[0] - ops[0].shift(self._to_int(ops[1]))
        
        elif tree.name == 'log':
            # LOG(A): 自然对数函数
            if not isinstance(ops[0], Number):
                tol = 1e-5
                ops[0][(ops[0]>=0)&(ops[0]<tol)] = tol
            return np.log(ops[0])
        
        elif tree.name == 'sum':
            # SUM(A, n): 序列A过去n天求和
            return ops[0].rolling(self._to_int(ops[1]), min_periods=1).sum()
        
        elif tree.name == 'abs':
            #  ABS(A): 绝对值函数
            return ops[0].abs()
        
        elif tree.name == 'tsargmax':
            # tsargmax(x, d) = which day tsmax(x, d) occurred on
            return ops[0].rolling(self._to_int(ops[1]),min_periods=1).apply(np.argmax)

        elif tree.name == 'tsargmin':
            # tsargmax(x, d) = which day tsmax(x, d) occurred on
            return ops[0].rolling(self._to_int(ops[1]),min_periods=1).apply(np.argmin)
        
        elif tree.name == 'adv':
            num = self._to_int(ops[0])
            return self.data['volume'].rolling(num,min_periods=1).mean()
        
        elif tree.name == 'indneutralize':
#            pdb.set_trace()
            size = ops[0].shape
            total = ops[0].mul(ops[1][1],axis=1)
            for i in range(size[0]):
                date = ops[0].index[i]
                industry = {}
                cap = {}
                mean = {}
                var = {}
                for j in range(size[1]):
                    secucode = ops[0].columns[j]
                    industry_info = int(ops[1][0].loc[date,secucode])
                    try:
                        industry[industry_info] += total.loc[date,secucode]
                        cap[industry_info] += ops[1][1].loc[date,secucode]
                    except KeyError:
                        industry[industry_info] = total.loc[date,secucode]
                        cap[industry_info] = ops[1][1].loc[date,secucode]
                for industry_value in list(industry.keys()):
                    mean[industry_value] = industry[industry_value]/cap[industry_value]
                for j in range(size[1]):
                    secucode = ops[0].columns[j]
                    industry_info = int(ops[1][0].loc[date,secucode])
                    try:
                        var[industry_info] += (ops[1][1].loc[date,secucode]/cap[industry_info])*(ops[0].loc[date,secucode]-mean[industry_info])**2
                    except KeyError:
                        var[industry_info] = (ops[1][1].loc[date,secucode]/cap[industry_info])*(ops[0].loc[date,secucode]-mean[industry_info])**2
                for j in range(size[1]):
                    secucode = ops[0].columns[j]
                    industry_info = int(ops[1][0].loc[date,secucode])
                    ops[0].loc[date,secucode] = (ops[0].loc[date,secucode]-mean[industry_info])/(math.sqrt(var[industry_info])+0.01)
#                pdb.set_trace()
            return ops[0]
        
        elif tree.name == 'highday':
            num = self._to_int(ops[1])
#            nums = self._to_constant_df(num-1,ones,1)
            return num-1-ops[0].rolling(self._to_int(ops[1]),min_periods=1).apply(np.argmax)
        elif tree.name == 'lowday':
            num = self._to_int(ops[1])
#            nums = self._to_constant_df(num-1,ones,1)
            return num-1-ops[0].rolling(self._to_int(ops[1]),min_periods=1).apply(np.argmin)
        elif tree.name == 'tofloat':
            return (ops[0] * 1).astype(float)
              
        elif tree.name == 'sequence':
            onss = self._to_constant_df(1)
            for i in range(onss.shape[1]):
                for j in range(1,onss.shape[0]):
                    onss.iloc[j,i] =  onss.iloc[j-1,i]
            return onss
        elif tree.name == 'regbeta':
            num = self._to_int(ops[2])
            beta = ops[0].copy()
            for i in range(ops[0].shape[1]):
                for n in range(num,ops[0].shape[0]):
                    y = ops[0].iloc[n-num:n, i]
                    x = ops[1].iloc[:num, i]
                    beta.iloc[n, i] = np.polyfit(x, y, deg=1)[0]
            return beta             
                  
        
        elif tree.name == 'tsrank':
            # TSRANK(A, n): 序列A的末位值在过去n天的顺序排位
            # ref: https://stackoverflow.com/questions/38856551/panda-rolling-window-percentile-rank
            # by default the smallest ranks first: rank=0
            # It is slow ...
            #_rank = lambda df: pd.Series(df).rank(pct=True, ascending=False).iloc[-1]

            # by Yijie Zhang
            _rank = lambda x: (x >= (np.array(x))[-1]).sum() / float(x.shape[0])
#            pdb.set_trace()
            return ops[0].rolling(self._to_int(ops[1]),min_periods=1).apply(_rank)
        
        elif tree.name == 'scale':
            # scale(x, a) = rescaled x such that sum(abs(x)) = a (the default is a = 1)
            # from Yifei's code
            _scale = lambda x: x.div(np.abs(x).sum())
            return ops[0].apply(_scale)

        elif tree.name == 'sign':
            # SIGN(A): 符号函数
            # sign function: >0 -> 1; <0 -> -1; ==0 -> 0
            return np.sign(ops[0])
        
        elif tree.name == 'covariance':
            # COVIANCE (A, B, n): 序列A、B过去n天协方差
            # covariance(x, y, d) = time-serial covariance of x and y for the past d days
            num = self._to_int(ops[2])
            val = ops[0].rolling(num,min_periods=1).corr(ops[1].rolling(num,min_periods=1))
            val[np.isinf(val)] = 0
            return val
        
        elif tree.name == 'delay':
            # DELAY(A, n): value of x d days ago
            return ops[0].shift(self._to_int(ops[1]))
        
        elif tree.name == 'tsmax':
            # TSMAX(A, n): 序列A过去n天的最大值
            return ops[0].rolling(self._to_int(ops[1]), min_periods=1).max()
            
        elif tree.name == 'tsmin':
            # TSMIN(A, n): 序列A过去n天的最小值
            return ops[0].rolling(self._to_int(ops[1]), min_periods=1).min()
        
        elif tree.name == 'prod':
            # PROD(A, n): 序列A过去n天累乘
            _prod = lambda df: np.prod(df)
            return ops[0].rolling(self._to_int(ops[1]), min_periods=1).apply(_prod)
        
        elif tree.name == 'count':
            # COUNT(condition, n): 计算前n期满足条件condition的样本个数
            return (ops[0] * 1).rolling(self._to_int(ops[1]), min_periods=1).sum()
        
        elif tree.name == 'mean':
            # MEAN(A, n): 序列A过去n天均值
            return ops[0].rolling(self._to_int(ops[1]), min_periods=1).mean()
            
        elif tree.name == 'sma':
            # SMA(A, n, m): Y_{i+1} = (A_i * m + Y_i * (n - m))/n，其中Y表示最终结果
            # 这里添加了__num_tick_in_unit但是由于数据点之间间隔也不规则，这样做是否合理还要想想
            n = self._to_int(ops[1])
            m = self._to_int(ops[2])
            if m > n:
                return self._to_constant_df(np.nan)
            alpha1 = m / n
            alpha2 = 1 - alpha1
            def _sma(x):
                y = x.copy()
                v = y.values
                for i in range(1, x.shape[0]):
                    # if there is a nan in v, the following will be nan
                    if not (np.isfinite(v[i-1])):
                        v[i] = x.iloc[i-1]
                    else:
                        v[i] = alpha1 * x.iloc[i-1] + alpha2 * v[i-1]
                return y
            return ops[0].apply(_sma)
        
        elif tree.name == 'ema':
            # EMA(A, n): 按照n的衰减，指数加权平滑
            return ops[0].ewm(span=self._to_int(ops[1])).mean()
            
        elif tree.name == 'wma':
            # WMA(A, n): 计算A前n期样本加权平均值权重为0.9^i，(i表示样本距离当前时点的间隔)
            window_num = self._to_int(ops[1])
            weights = np.power(0.9, np.linspace(start=window_num, stop=0, num=window_num))
            weights = weights / np.sum(weights)
            def _wma(x):
                return np.nansum(x * weights)
            return ops[0].rolling(window_num,min_periods=1).apply(_wma)
            
        elif tree.name == 'decaylinear':
            # DECAYLINEAR(A, d): 对A序列计算移动平均加权，其中权重对应d,d-1,...,1(权重和为1)，
            # 每单位时间权重下降1
#            pdb.set_trace()
            window_num = self._to_int(ops[1])
            weights = np.linspace(window_num, 0, window_num, endpoint=False)
            weights = weights / np.sum(weights)
            def _decay_linear_avg(x):
                return np.nansum(x * weights)
            return ops[0].rolling(window_num).apply(_decay_linear_avg)            
        elif tree.name == 'sumac':
            # SUMAC(A): 计算A的前n项的累加
            return ops[0].cumsum()
            
        elif tree.name == 'prodac':
            # PRODAC(A): 计算A的前n项的累乘
            return ops[0].cumprod()

        elif tree.name == '+':
            return ops[0] + ops[1]
        
        elif tree.name == '-':
            return ops[0] - ops[1]
        
        elif tree.name == '*':
            return ops[0] * ops[1]
        
        elif tree.name == '/':
            # set denominator against zero
            if not isinstance(ops[1], Number):
                tol = 1e-5
                ops[1][(ops[1]>=0)&(ops[1]<tol)] = tol
                ops[1][(ops[1]<=0)&(ops[1]>-tol)] = -tol
            return ops[0] / ops[1]
        
        elif tree.name == '?':
            # trinary operation A?B:C
            if isinstance(ops[1], Number):
                B = self._to_constant_df(ops[1])
            else:
                B = ops[1]
            if isinstance(ops[2], Number):
                C = self._to_constant_df(ops[2])
            else:
                C = ops[2]
#            pdb.set_trace()
            C[ops[0]] = B
            return C
        
        elif tree.name == '<':
            return ops[0] < ops[1]
        
        elif tree.name == '>':
            return ops[0] > ops[1]
        
        elif tree.name == '<=':
            return ops[0] <= ops[1]
        
        elif tree.name == '>=':
            return ops[0] >= ops[1]
        
        elif tree.name == '==':
            return ops[0] == ops[1]
        
        elif tree.name == '||':
            return ops[0] | ops[1]
        
        elif tree.name == '&':
            return ops[0] & ops[1]
        
        elif tree.name == '^':
            # HanTian Zheng: modified
            # return ops[1].pow(ops[0])
            return pow(ops[0], ops[1]) if isinstance(ops[0], Number) else ops[0].pow(ops[1])

        else:

            raise ValueError('%s is not a valid function' % tree.name)

    def get_score(self, score_name):
        if self.tree.data is None:
            print("You haven't done the calculation!")
            exit()
            
        if score_name == 'rsquare':
            con = pd.concat([self.tree.data, self.data['label']], axis=1)
            con = con.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how='any')
            if con.shape[0] == 0:
                return 0
            
            _, _, r_value, _, _ = linregress(con.iloc[:, 0].values, con.iloc[:, 1].values)
            return r_value
        
        elif score_name == 'spearmanr':
            treedata = self.tree.data.fillna(0).values
            # 做一个rank_treedata[]对应treedata[]每一个元素的rank
            # print(treedata[201][3],treedata[231][3],treedata[241][3],treedata[291][3],treedata[299][3])
            rank_treedata = [[0 for col in range(treedata.shape[1])] for row in range(treedata.shape[0])]
            for i in range(treedata.shape[1]):
                treedata_with_number = []
                for j in range(treedata.shape[0]):
                    treedata_with_number.append([j, treedata[j][i]])
                treedata_with_number.sort(key=takeSecond)
                for k in range(len(treedata_with_number)):
                    rank_treedata[treedata_with_number[k][0]][i] = k
            # print(rank_treedata[201][3],rank_treedata[231][3],rank_treedata[241][3],rank_treedata[291][3],rank_treedata[299][3])
            labeldata = self.data['label'].fillna(0).values
            mean_spearmanr = 0
            count = 0
            # for i in range(len(treedata)):
            for i in range(200, 299):
                x, _ = spearmanr(rank_treedata[i], labeldata[i])
                if(np.isnan(x)):
                    continue
                else:
                    mean_spearmanr += x
                    count+=1
            mean_spearmanr = mean_spearmanr/count
            return mean_spearmanr

        elif score_name == 'pearsonr_old':
            treedata = self.tree.data.fillna(0).values
            labeldata = self.data['label'].fillna(0).values
            mean_pearsonr = 0
            count = 0
            # for i in range(len(treedata)):
            for i in range(100, 999):
                try:
                    x = pearsonr(treedata[i],labeldata[i])
                    if(np.isnan(x[0])):
                        continue
                    else:
                        mean_pearsonr += x[0]
                        count+=1
                except Exception:
                    pass
            try:
            	mean_pearsonr =mean_pearsonr/count
            except Exception:
                pass
            return mean_pearsonr
        
        elif score_name == 'pearsonr_new':
            treedata = self.tree.data.fillna(0).values
            labeldata = self.data['label'].fillna(0).values
            mean_pearsonr = 0
            count = 0
            for i in range(100, 999):
                # 按Factor排序删除中间股票
                new_treedata = []
                new_labeldata = []
                stock_len = len(treedata[i])
                treedata_set = []
                for j in range(stock_len):
                    treedata_set.append((j, treedata[i][j]))
                treedata_reset = sorted(treedata_set, key=itemgetter(1))
                reserve_set = []
                for j in range((int)(stock_len*0.3), (int)(stock_len*0.7)):
                    reserve_set.append(treedata_reset[j][0])
                for j in range(stock_len):
                    if j not in reserve_set:
                        new_treedata.append(treedata[i][j])
                        new_labeldata.append(labeldata[i][j])
                try:
                    x = pearsonr(new_treedata, new_labeldata)
                    if(np.isnan(x[0])):
                        continue
                    else:
                        mean_pearsonr += x[0]
                        count+=1
                except Exception:
                    pass
            try:
            	mean_pearsonr =mean_pearsonr/count
            except Exception:
                pass
            return mean_pearsonr

        elif score_name == 'pearsonr_os':
            treedata = self.tree.data.fillna(0).values
            labeldata = self.data['label'].fillna(0).values
            mean_pearsonr = 0
            count = 0
            for i in range(len(treedata)):
                try:
                    x = pearsonr(treedata[i],labeldata[i])
                    if(np.isnan(x[0])):
                        continue
                    else:
                        mean_pearsonr += x[0]
                        count+=1
                except Exception:
                    pass

#            if con.shape[0] == 0:
#                return 0
            try:
            	mean_pearsonr =mean_pearsonr/count
            except Exception:
                pass
#            pdb.set_trace()
            return mean_pearsonr
        
        elif score_name == 'entropy':
            topx = 20 # 表示取前几支最大return作为label=1
            treedata = self.tree.data.fillna(0).values
            labeldata = self.data['label'].fillna(0).values
            mean_entropy = 0
            count = 0
            tf.InteractiveSession()
            for i in range(100, 999):
                try:
                    labeldata_arg = np.argsort(labeldata[i])
                    for j in range(0, len(labeldata_arg)-topx):
                        labeldata[i][labeldata_arg[j]]=0
                    for j in range(len(labeldata_arg)-topx, len(labeldata_arg)):
                        labeldata[i][labeldata_arg[j]]=1
                    x = rm(scewl(logits=np.abs(treedata[i]), labels=labeldata[i])).eval()
                except Exception:
                    x = 99999
                # print('$$$$$$$$$$$$')
                # print(np.abs(treedata[i]))
                # print(labeldata[i])
                # print(i, x)
                mean_entropy += x
                count+=1
            mean_entropy = mean_entropy/count
            return mean_entropy


