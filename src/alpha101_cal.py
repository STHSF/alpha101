#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
@version: ??
@author: li
@file: alpha101_cal.py
@time: 2019-05-07 16:43
"""

from src.alpha101_zzh import *
from src.alpha101 import *


class DataPrepare(object):
    def __init__(self):
        pass

    def get_basic_data(self, start_date, end_date, stock_list):
        dependencies = ['close', 'open', 'high', 'low', 'pre_close', 'volume', 'money', 'trade_date', 'symbol']
        sql = "select * from test.alpha101_tmp where trade_date >='{}' and trade_date <='{}' and symbol in {}".format(start_date, end_date, stock_list)
        sql_engine = create_engine("mysql+mysqlconnector://root:1234@10.15.97.128:3306/test")

        sql_data = pd.read_sql(sql, sql_engine)
        # 先转化成multiindex
        res = sql_data.set_index(['trade_date', 'symbol']).to_panel()
        return res


if __name__ == '__main__':
    pd.set_option('display.max_rows', None, 'display.max_columns', None, "display.max_colwidth", 1000, 'display.width', 1000)

    alpha_data = DataPrepare()
    security = ('600519.XSHG', '000858.XSHG', '000799.XSHE', '002304.XSHE', '000860.XSHE', '603369.XSHG', '000568.XSHE')

    stock_panel = alpha_data.get_basic_data("2010-01-01", "2018-12-30", stock_list=security)

    start_time = time.time()
    alpha = Alpha101(stock_panel)
    alpha_ = Alpha101Z26(stock_panel)
    pn = alpha.calculate()
    pn_ = alpha_.calculate()
    res = pn.join(pn_)
    print('spend time %s' % (time.time() - start_time))


    print(res.minor_xs('600519.XSHG'))
    # 数据保存， 用作相关性检验
    res.minor_xs('600519.XSHG').to_csv('600519.csv')

