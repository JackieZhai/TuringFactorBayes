# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 15:49 2018
@author: zhaihao

Data: multi_basic_data_tuning.csv
Formula: good_to_tune.txt
Object Function: IC(pearsonr)
Parameter Bound: [1, 100] (integer)
Number of Start Point: 10
Number of Iterations: 50

TODO:
(√)1. 计算IC的时候，把因子算的中间(40%)的股票忽略，再和Return做IC试试
(√)2. 去除计算时候的各种错误取值情况（NaN、INF等）
(没思路)3. 调整一些BayesianOptimization的超参数，找到合适的初始点
4. 加入样本外测试，防止过拟合
(见calculator.py)5. 加入参数的限制（首先是要统计有哪几种限制）
"""

import sys
sys.path.append('../mongo')
from bayes_opt import BayesianOptimization
from get_data import get_data
from calculator import Calculator
from formula import *
from utils import *
from os.path import join as joindir
from os import makedirs
import multiprocessing
import pandas as pd
import numpy as np
import traceback
import logging
import copy
import time
import gc
import pdb
import global_list as gl

# 载入公式的文件
INPUT_FORMULA_FILE = 'original_formulas_test.txt'
# 每个公式要跑几遍
times_loop = 1
# 选择股票池
pool_id = '000300'

pool_list = {'000001':'SHCI','000300':'hs300','000905':'zz500','000906':'zz800','399005':'SmallCap','399006':'GEMI','399106':'SZCI','399317':'ASCI'}
OUTPUT_RESULT_DIR = './tuning_result_bayes_v2'
OUTPUT_FORMULA_DIR = joindir(OUTPUT_RESULT_DIR, 'formula')
OUTPUT_FORMULA_FILE = joindir(OUTPUT_FORMULA_DIR, 'tuned_formulas_bayes.txt')
OUTPUT_DATA_DIR = joindir(OUTPUT_RESULT_DIR, 'data')
OUTPUT_DATA_IS_DIR = joindir(OUTPUT_DATA_DIR, 'insample') # empty now
OUTPUT_DATA_OS_DIR = joindir(OUTPUT_DATA_DIR, 'outsample') # empty now
OUTPUT_AUX_DIR = joindir(OUTPUT_RESULT_DIR, 'auxilliary')
OUTPUT_SCORE_FILE = joindir(OUTPUT_AUX_DIR, 'score.csv')
OUTPUT_FAILURE_FILE = joindir(OUTPUT_AUX_DIR, 'failure.csv')
OUTPUT_LOG_FILE = joindir(OUTPUT_AUX_DIR, 'log.txt')

makedirs(OUTPUT_FORMULA_DIR, exist_ok=True)
makedirs(OUTPUT_DATA_IS_DIR, exist_ok=True)
makedirs(OUTPUT_DATA_OS_DIR, exist_ok=True)
makedirs(OUTPUT_AUX_DIR, exist_ok=True)

logging.basicConfig(filename=OUTPUT_LOG_FILE)
logger = logging.StreamHandler()
logger.setFormatter(logging.Formatter(logging.BASIC_FORMAT))
logging.getLogger().addHandler(logger)

def data2ztp_format(data):
    res = {}
    res['open'] = data.pivot(index='TradingDay',columns='SecuCode',values='open').fillna(method='ffill').fillna(method='bfill')
    res['close'] = data.pivot(index='TradingDay',columns='SecuCode',values='close').fillna(method='ffill').fillna(method='bfill')
    res['high'] = data.pivot(index='TradingDay',columns='SecuCode',values='high').fillna(method='ffill').fillna(method='bfill')
    res['low'] = data.pivot(index='TradingDay',columns='SecuCode',values='low').fillna(method='ffill').fillna(method='bfill')
    res['vwap'] = data.pivot(index='TradingDay',columns='SecuCode',values='vwap').fillna(method='ffill').fillna(method='bfill')
    res['volume'] = data.pivot(index='TradingDay',columns='SecuCode',values='volume').fillna(method='ffill').fillna(method='bfill')
    res['amount'] = data.pivot(index='TradingDay',columns='SecuCode',values='amount').fillna(method='ffill').fillna(method='bfill')
    res['label'] = data.pivot(index='TradingDay',columns='SecuCode',values='rate').fillna(method='ffill').fillna(method='bfill')
    res['cap'] = data.pivot(index='TradingDay',columns='SecuCode',values='cap').fillna(method='ffill').fillna(method='bfill')
    res['sector'] = data.pivot(index='TradingDay',columns='SecuCode',values='sector').fillna(method='ffill').fillna(method='bfill')
    res['industry'] = data.pivot(index='TradingDay',columns='SecuCode',values='industry').fillna(method='ffill').fillna(method='bfill')
    res['subindustry'] = data.pivot(index='TradingDay',columns='SecuCode',values='subindustry').fillna(method='ffill').fillna(method='bfill')
    data.loc[:, 'ones'] = 1
    res['ones'] = data.pivot(index='TradingDay',columns='SecuCode',values='ones').fillna(method='ffill').fillna(method='bfill')
    return res

# 加载数据
pre_data = get_data(pool_id, 'is.csv')
pre_data_os = get_data(pool_id, 'os.csv')
print("Accessed the database.")
# 数据标准化
global data
data = data2ztp_format(pre_data)
global data_os
data_os = data2ztp_format(pre_data_os)
print('Formated the data.')
gl._init()
gl.set_value('data', data)
gl.set_value('data_os', data_os)

def get_constnode(tree, last_ts=False):
    node_list = []
    if isinstance(tree, constnode):
        if last_ts:
            # last_ts 表示上一层的节点为time series相关的节点，仅该部分节点需要调参
            if tree.data < 1:
                # # 如果原本传入参数小于1，不对其调参
                # tree.isTune = False
                tree.isTune = True
                return []
            else:
                tree.isTune = True
                return [tree]
        else:
            tree.isTune = False
            return []
    elif isinstance(tree, paramnode):
        # 如果是变量节点，不对其调整
        return []
    else:
        # 判断现在的此运算下是否可能包含需要调参的直接子节点
        current_ts = (tree.name != '?') and ((tree.name in ts_list) or (tree.name in bi_ts_list))
        for children in tree.children:
            result = get_constnode(children, current_ts)
            node_list.extend(result)
        return node_list

def check_update(tree): # 使新换的参数符合限制要求
    if isinstance(tree, node):
        if (tree.name == '/') and (tree.children[0].name == 'sum'): # 参数限制(1)
            tree.children[0].children[1].name = tree.children[1].name
            tree.children[0].children[1].data = tree.children[1].data
        if tree.name == 'regbeta': # 参数限制(2)
            if tree.children[0].name == 'mean':
                tree.children[1].children[0].name = tree.children[0].children[1].name
                tree.children[1].children[0].data = tree.children[0].children[1].data
                tree.children[2].name = tree.children[0].children[1].name
                tree.children[2].data = tree.children[0].children[1].data
            else:
                tree.children[2].name = tree.children[1].children[0].name
                tree.children[2].data = tree.children[1].children[0].data
    for children in tree.children:
        check_update(children)

def calculation(tree, verbose=0):
    data = gl.get_value('data')
    try:
        start = time.clock()
        calculator = Calculator(data)
        calculated_data = calculator.calculate_features(tree)
        end1 = time.clock()
        # 取后面数据算皮尔森，剔除NaN，为此把get_score改了一下
        score = calculator.get_score('pearsonr_new')
        end2 = time.clock()
        calculator.delete()
        # pdb.set_trace()
        del calculator # useless
        del data
        gc.collect()
        if verbose >= 1:
            print("The feature is using time %fs, %fs with score %f." \
                % (end1-start, end2-end1, score))
        if verbose >= 2:
            if isinstance(tree, str):
                print("%s" % tree)
            else:
                print("%s" % tree_to_formula(tree))
        return (score, tree, calculated_data)
    except Exception:
        print(traceback.format_exc())

def calculation_os(tree, verbose=0):
    data_os = gl.get_value('data_os')
    try:
        start = time.clock()
        calculator = Calculator(data_os)
        calculated_data = calculator.calculate_features(tree)
        end1 = time.clock()
        # 取后面数据算皮尔森，剔除NaN，为此把get_score改了一下
        score = calculator.get_score('pearsonr_new_os')
        end2 = time.clock()
        calculator.delete()
        # pdb.set_trace()
        del calculator # useless
        del data_os
        gc.collect()
        if verbose >= 1:
            print("The out sample is using time %fs, %fs with score %f." \
                % (end1-start, end2-end1, score))
        if verbose >= 2:
            if isinstance(tree, str):
                print("%s" % tree)
            else:
                print("%s" % tree_to_formula(tree))
        return (score, tree, calculated_data)
    except Exception:
        print(traceback.format_exc())

now_tree = formula_to_tree('(mean(close,1)/close)') # 初始化一个无关紧要的树
def get_tree_answer(**params):
    constnode_list = get_constnode(now_tree)
    for count1, key in enumerate(params):
        for count2, const_node in enumerate(constnode_list):
            if count1==count2:
                const_node.change_value((int)(params[key]))
    new_tree = copy.deepcopy(now_tree)
    check_update(new_tree)
    ans = calculation(new_tree)[0]
    if np.isnan(ans) or np.isinf(ans):
        return 0
    else:
        return np.abs(ans)

def fine_tuning(tree, verbose=1): 
    original_score = calculation(tree)[0]
    original_score_os = calculation_os(tree)[0]
    constnode_list = get_constnode(tree)
    constnode_num = len(constnode_list)
    
    print(' - There are %d parameters to be tuned in \n   %s' \
        % (constnode_num, tree_to_formula(tree, for_print=True)))
    
    if constnode_num == 0:
        print('Warning: There is no need to tune.')
        return 

    global now_tree
    now_tree = copy.deepcopy(tree)

    constnode_list_pbound = {}
    constnode_list_pbound_ori = {}
    constnode_list_pbound_ori.update({'target': [0.05]})
    for i in range(constnode_num):
        constnode_list_pbound.update({'para'+str(i): (1, 100)})
        constnode_list_pbound_ori.update({'para'+str(i): [constnode_list[i].data]})

    bo = BayesianOptimization(get_tree_answer, constnode_list_pbound)
    bo.explore(constnode_list_pbound_ori)
    bo.maximize(init_points=10, n_iter=100)

    best_score = bo.res['max']['max_val']
    for count1, key in enumerate(bo.res['max']['max_params']):
        for count2, const_node in enumerate(constnode_list):
            if count1==count2:
                const_node.change_value((int)(bo.res['max']['max_params'][key]))
    best_tree = copy.deepcopy(tree)
    best_score_os = calculation_os(best_tree)[0]
    print('--------------------------------------------------------------------')
    print(' - Tuned formula:\n   %s' % tree_to_formula(best_tree, for_print=True))
    print(' + Tuned score = %.5f' % best_score)
    print('--------------------------------------------------------------------')
    result = {
        'original_formula': tree_to_formula(now_tree), 
        'original_score_is': original_score, 
        'original_score_os': original_score_os,
        'tuned_formula': tree_to_formula(best_tree), 
        'tuned_score_is': best_score,
        'tuned_score_os': best_score_os,
    }
    return best_tree, result


if __name__ == '__main__':
    
    with open(INPUT_FORMULA_FILE, 'r') as f:
        formulas = f.readlines()

    tuned_formulas = []
    result = []
    failure_list = pd.DataFrame()
    
    with open(OUTPUT_FORMULA_FILE, 'w') as f:
        f.write('')
    
    for count, expr in enumerate(formulas):
        print(expr)
        alpha_name = expr.split('@')[0].strip()
        alpha_expr = expr.split('@')[1].strip('\n')
        # pdb.set_trace()
        try:
            print('> Tuning %s: ' % alpha_name)
            tree = formula_to_tree(alpha_expr)
            tuning_result = []
            best_data = []
            for i in range(times_loop):
                tree, tuning_result = fine_tuning(tree)
            alpha_name = 'tuned_'+alpha_name
            tuned_formulas.append((alpha_name, tuning_result['tuned_formula']))
            result.append(tuning_result)
            # 把调好的公式写入到文件中
            with open(OUTPUT_FORMULA_FILE, 'a+') as f:
                f.write('%s_%s_bayes @ %s \n' % (alpha_name, pool_list[pool_id], tuning_result['tuned_formula']))
        except NotImplementedError as e:
            print('Failed: Not Implemented {0}\n'.format(e.args[0]))
            failure_list.loc[alpha_name, 'error'] = e.args[0]
        except AssertionError as e:
            print('Failed: Assertation Error {0}\n'.format(e.args[0]))
            failure_list.loc[alpha_name, 'error'] = e.args[0]
        except ValueError as e:
            print('Failed: Value Error {0}\n'.format(e.args[0]))
            failure_list.loc[alpha_name, 'error'] = e.args[0]
        except KeyboardInterrupt:
            raise
        except:
            # raise
            print('Failed: Unknown Error: %s @ %s' % (alpha_name, alpha_expr))
            failure_list.loc[alpha_name, 'error'] = 'Unknow Error'

    pd.DataFrame(result).to_csv(OUTPUT_SCORE_FILE)
    failure_list.to_csv(OUTPUT_FAILURE_FILE)
