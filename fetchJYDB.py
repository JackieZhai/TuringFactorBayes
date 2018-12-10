#-*- coding=utf-8 -*-
import MySQLdb as mdb
import numpy as np
import pandas as pd
import datetime
from sqlalchemy import create_engine
import pdb
import warnings
warnings.filterwarnings('ignore')


default_config_115 = {
    'host': '175.25.50.115',
    'port': 3306,
    'user': 'xtech',
    'passwd': 'x-tech-123',
    'db': 'jydb',
    'charset': 'gbk'
    }

xtech_config = {
    'host': '175.25.50.115',
    'port': 3306,
    'user': 'xtech',
    'passwd': 'x-tech-123',
    'db': 'xtech',
    'charset': 'gbk',    
}

class FetchData(object):
    def __init__(self, config):
        self.config = config
        #self.conn = mdb.connect(**config)
        #self.conn.autocommit(1)
        self.conn = None
        
    def __del__(self):
        if self.conn is not None:
            self.conn.close()
        
    def _exec(self, query, describe=False):
        if self.conn is None:
            self.conn = mdb.connect(**self.config)
            self.conn.autocommit(1)
        try:
#            pdb.set_trace()
            self.conn.ping(True)
            cursor = self.conn.cursor()
            count = cursor.execute(query)
            cursor.scroll(0, mode='absolute')
            results = cursor.fetchall()
            description = cursor.description
#            pdb.set_trace()
        except:
            #print(query, ' error')
            #import traceback
            #traceback.print_exc()
            try: # retry
                self.conn = mdb.connect(**self.config)
                self.conn.autocommit(1)
                cursor = self.conn.cursor()
                count = cursor.execute(query)
                cursor.scroll(0, mode='absolute')
                results = cursor.fetchall()
                description = cursor.description
            except:
                self.conn.rollback()
                cursor.close()
                return None, None
        cursor.close()
#        pdb.set_trace()
        if describe == False:
            return results
        else:
            columns = []
            for x in description:
                columns.append(x[0])
            return columns,results
            
    def fetch_tables(self, verbose=False):
        tables = []
        query = 'SHOW TABLES'
        columns,results = self._exec(query)
        for x in results:
            tables.append(x[0])
        if verbose == True:
            print(results)
        return tables
	
    def fetch_columns(self, table_name, verbose=False):
        columns = []
        query = 'SHOW COLUMNS FROM %s' %table_name
        columns,results = self._exec(query)
        for x in results:
            columns.append(x[0])
        if verbose == True:
            print(results)
        return columns

    def _fetch_index(self, table_name, verbose=False):
        ids = []
        query = 'SHOW INDEX FROM %s' %table_name
        columns,results = self._exec(query)
        for x in results:
            ids.append(x[0])
        if verbose == True:
            print(results)
        return ids

    def ping(self, table_name, verbose=False):
        query =  'SELECT * FROM %s LIMIT %s' %(table_name, 1)
        columns,result = self._exec(query)
        if verbose == True:
            print (result[0])
        if result is None:
            print('Query failed!')
        else:
            print('Query success!')
        return result


    def head(self, table_name, limit=5):
        query =  'SELECT * FROM %s  ORDER BY id ASC LIMIT %s' %(table_name, limit)
        columns, results = self._exec(query, True)
        #columns = self.fetch_columns(table_name)
        df_head = pd.DataFrame(np.array(results), columns = columns)
        print (df_head)
        return df_head

    def tail(self, table_name, limit=5):
        query =  'SELECT * FROM %s ORDER BY id DESC LIMIT %s' %(table_name, limit)
        columns, results = self._exec(query,True)
        #columns = self.fetch_columns(table_name)
        df_tail = pd.DataFrame(np.array(results), columns = columns)
        #print (df_tail)
        return df_tail

    def getdata(self, table_name, columns=None, date_name=None, start_date=None, end_date=None):
        if isinstance(columns, list):
            columns = ",".join(columns)
        else:
            columns = '*'

        if date_name is None:
            query =  'SELECT %s FROM %s' %(columns, table_name)
        else :
            start_date = '"'+start_date+'"'
            end_date = '"'+end_date+'"'
            query =  'SELECT %s FROM %s  where %s >= %s and %s <= %s order by %s asc;' %(columns, table_name, date_name, start_date, date_name, end_date, date_name)

        columns,results = self._exec(query,True)
        #print(results)
        if results is None:
            print('Query Failed!')
            df = None
        else:
            df = pd.DataFrame(np.array(results), columns=columns)
        return df

    def get_minmax(self, table_name, columns):
        code = self.getdata(table_name, columns)
        print (code.min(), code.max())
        return (code.min(), code.max())
    
    def get_day_data(self, table_name, date_name, date):
        query = 'SELECT * FROM {} where {} = "{}" '.format(table_name, date_name, date)
        columns,results = self._exec(query, True)
        if results is None:
            print('Query Failed!')                   
            df = None
        else:
            #df_col = self.fetch_columns(table_name)
            df = pd.DataFrame(np.array(results), columns = columns)
            df.index = np.arange(df.shape[0])
        return df
      
    def get_quarter_data(self, table_name, date_name, quarter_time, limit):
        query = 'SELECT * FROM %s where %s <= "%s" ORDER BY %s DESC limit %s' %(table_name, date_name, quarter_time, date_name, limit)
        columns,results = self._exec(query, True)
        if results is None:                                   
            print('Query Failed!')                            
            df = None
        else:
            #df_col = self.fetch_columns(table_name)
            df = pd.DataFrame(np.array(results), columns = columns)
            df.index = np.arange(df.shape[0])
        return df   
       
 
    def exec_sql(self, query):
        columns,results = self._exec(query,True)
        if results is None:
            return None
        else: 
            df = pd.DataFrame(np.array(results), columns = columns)
            return df

    def exec_file(self,filename):
        with open(filename, 'r') as f:
            query = f.read()
            columns, results = self._exec(query,True)
            df = pd.DataFrame(np.array(results), columns = columns)
        return df

class Wrapper_FetchData(FetchData):
    def __init__(self, config, SecuCategory = 1):
        super(Wrapper_FetchData,self).__init__(config=config)
        self.SecuCategory = SecuCategory
        
    def set_SecuCategory(self, SecuCategory):
        self.SecuCategory = SecuCategory
        print("Now SecuCategory:",self.SecuCategory)

    def decorator(func):
        def wrapper(self, *args, **kw):
            describe = True
            if not describe:
                return func(*args, **kw)
            else:
                columns,results = func(self, *args, **kw)
                if results is None:
                    return columns, results
                elif("SecuCode" in columns):
                    return columns,results
                elif("InnerCode" in columns):
                    query = 'SELECT InnerCode,SecuCode FROM SecuMain where SecuCategory = %s and (SecuMarket = 83 or SecuMarket = 90);' %(self.SecuCategory)
                    df_main = self.exec_sql(query)
                    df_main = df_main[['InnerCode','SecuCode']]
                    df_main['InnerCode'] = df_main['InnerCode'].astype(np.int64)
                    df = pd.DataFrame(np.array(results), columns = columns)
                    df['InnerCode'] = df['InnerCode'].astype(np.int64)
                    out_df = pd.merge(df_main, df, on = "InnerCode")
#                    pdb.set_trace()
                    return out_df.columns, out_df.values
                elif("CompanyCode" in columns):
                    query = 'SELECT CompanyCode,SecuCode FROM SecuMain where SecuCategory = %s and (SecuMarket = 83 or SecuMarket = 90);' %(self.SecuCategory)
                    df_main = self.exec_sql(query)
                    df_main = df_main[['CompanyCode','SecuCode']]
                    df_main['CompanyCode'] = df_main['CompanyCode'].astype(np.int64)
                    df = pd.DataFrame(np.array(results), columns = columns)
                    df['CompanyCode'] = df['CompanyCode'].astype(np.int64)
                    out_df = pd.merge(df_main, df, on = "CompanyCode")
                    columns = out_df.columns
                    results = out_df.values
                    return out_df.columns, out_df.values
                else:
                    return columns,results
        return wrapper

    @decorator
    def _exec(self, query, describe=True):
#        pdb.set_trace()
        if self.conn is None:
            self.conn = mdb.connect(**self.config)
            self.conn.autocommit(1)

        try:
            cursor = self.conn.cursor()
            count = cursor.execute(query)
#            pdb.set_trace()
            cursor.scroll(0, mode='absolute')
            results = cursor.fetchall()
#            pdb.set_trace()
            description = cursor.description
        except:
            #print(query, ' error')
            #import traceback
            #traceback.print_exc()
            self.conn.rollback()
            results =  None
            description = None
            return None, None
        cursor.close()
        if describe == False:
            return results
        else:
            columns = []
            for x in description:
                columns.append(x[0])
            return columns,results
            

    def get_IndexComponent(self, SecuCode, Date):
        '''
            SecuCode type:str  exmple: '000300' (hs300)
            Date: type:str. exmple '2017-04-10'
        '''
        query = 'SELECT InnerCode, SecuCode FROM SecuMain where SecuCode = {} and SecuCategory = 4 and (SecuMarket = 83 or SecuMarket = 90);'.format(SecuCode)
        df_main = self.exec_sql(query)
        #print(df_main)
        assert len(df_main)==1
        IndexCode = int(df_main['InnerCode'].values)
        Date = pd.to_datetime(Date)
        query2 ='select InnerCode from SA_TradableShare where IndexCode = {} and EndDate = "{}" and DataType = 1 '.format(IndexCode, Date)
        df = self.exec_sql(query2)
#        pdb.set_trace()
        result = pd.DataFrame(df['SecuCode'].unique(),columns = [SecuCode])
        return result

    def get_IndexComponent2(self, SecuCode, Date=None):
        '''
            SecuCode type:str  exmple: '000300' (hs300)
            Date: type:str. exmple '2017-04-10'
        '''
        query = 'SELECT InnerCode, SecuCode FROM SecuMain where SecuCode = {} and SecuCategory = 4 and (SecuMarket = 83 or SecuMarket = 90);'.format(SecuCode)
        df_main = self.exec_sql(query)
        #print(df_main)
        assert len(df_main)==1
        IndexCode = int(df_main['InnerCode'].values[0])
        Date = str(pd.to_datetime(Date))
#        pdb.set_trace()
#        query2 ='select SecuInnerCode as InnerCode from LC_IndexComponent where IndexInnerCode = {} and OutDate is Null'.format(IndexCode)
        query2 = 'select SecuInnerCode as InnerCode from LC_IndexComponent where IndexInnerCode = %s and (OutDate is Null or OutDate >"%s") and InDate <= "%s"'%(IndexCode,Date,Date)
        df = self.exec_sql(query2)
#        pdb.set_trace()
        result = pd.DataFrame(df['SecuCode'].unique(),columns = [SecuCode])
        return result
    
##################################################
secu_loader = Wrapper_FetchData(config = default_config_115)

if __name__ == '__main__':
    #date = pd.datetime(2018,1,5)
    #quote_df = jydb_loader.get_day_data("QT_DailyQuote", "TradingDay", date)
    #a = set(list(secu_loader.get_IndexComponent('000001','2018-06-27')['000001']))
    b = secu_loader.get_IndexComponent2('000905', '2018-04-01')
    print(b)
    #print(b-a)
    #df = secu_loader.exec_sql('select * from LC_QIncomeStatementNew limit 100;')
    #print(df)
    #print(zenan_db.ping('2018-08-17'))
