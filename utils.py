
# 定义各种常量

var_names = ['open', 'high', 'low', 'close', 'vwap', 'volume', 'amount', 'ones', 'ret', 'dtm','cap', 'dbm', 'tr', 'hd', 'ld','indclass.sector','indclass.industry','indclass.subindustry']

unary_list = ['abs', 'log', 'sign', 'sumac','sequence','tofloat','adv','rank','scale']

ts_list = ['sum', 'wma', 'ema', 'tsmax', 'tsmin','tsargmax','tsargmin', 'delay', 'std', 'delta', 'count', 
    'tsrank', 'mean', 'decaylinear', 'prod','highday','lowday']

binary_list = ['indneutralize','signedpower','min', 'max', '+', '-', '*', '/', '>', '<', '==', '>=', '<=', '||', '^', '&']

bi_ts_list = ['corr', 'covariance', 'sma', '?','regbeta']

all_func_list = unary_list + ts_list + binary_list + bi_ts_list

func_op_num = {
    'abs': 1, 'log': 1, 'sign': 1, 'sumac': 1,'sequence':1,'tofloat':1,'highday':2,'lowday':2,
    'sum': 2, 'wma': 2, 'ema': 2, 'tsmax': 2, 'tsmin': 2, 'tsargmax':2,'rank':1,'scale':1,
    'delay': 2, 'std': 2, 'delta': 2, 'count': 2,'tsargmin':2,'adv':1,'signedpower':2,
    'tsrank': 2, 'mean': 2, 'decaylinear': 2, 'prod': 2, 'indneutralize':2,
    'min': 2, 'max': 2, 'corr': 3, 'covariance': 3, 'sma': 3, '?': 3,'regbeta':3,
    '+': 2, '-': 2, '*': 2, '/': 2, '>': 2, '<': 2, '==': 2, 
    '>=': 2, '<=': 2, '||': 2, '^': 2, '&': 2
}

class bcolors:
    # 给变量一个紫色
    VAR_NODE = '\033[95m'
    # 没用上
    PARAM_NODE = '\033[94m'
    # 给普通参数（不可调）一抹绿色
    CONST_NODE = '\033[92m'
    # 给可调参数一坨黄色并且加粗
    CONST_NODE_FOR_TUNE = '\033[93m\033[1m'
    # 给运算符和函数醒目的红色
    OP_NODE = '\033[91m'
    # 结束他们的颜色
    END = '\033[0m'

    # 如何加粗和加下划线（没用上）
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class fwrapper:
    def __init__(self, childcount, name):
        self.childcount = childcount
        self.name = name

class node:
    def __init__(self, fw, children):
        self.name = fw.name
        self.children = children
        self.data = None

    # def __del__(self):
    #     del self.data
    #     for child in self.children:
    #         del child

    def evaluate(self, inp):
        results = [n.evaluate(inp) for n in self.children]
        return self.function(results)

    def display(self, indent=''):
        print(indent + bcolors.OP_NODE + self.name + bcolors.END + ' (op node)')
        for c in self.children:
            c.display(indent.replace('-', ' ') + '|---')

class paramnode:
    __var_names = var_names

    def __init__(self, idx):
        self.idx = idx
        self.name = self.__var_names[self.idx]
        self.data = None

    # idx is the location in the parameters tuple
    def evaluate(self, inp):
        return inp[self.idx]

    def display(self, indent=''):
        print(indent + bcolors.VAR_NODE + self.name + bcolors.END + ' (var node)')

class constnode:
    def __init__(self, v):
        self.name = v
        self.data = self.name
        self.isTune = None

    def evaluate(self, inp):
        return self.name

    def change_value(self, value):
        self.name = value
        self.data = value

    def display(self, indent=''):
        print(indent + bcolors.CONST_NODE + str(self.name) + bcolors.END + ' (const node)')

func_list = []

for i, func in enumerate(all_func_list):
    func_i = fwrapper(func_op_num[func], str(func))
    func_list.append(func_i)
    
    
