#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
@version: ??
@author: li
@file: alpha101_tmp_cal.py
@time: 2019-05-07 13:07
"""

import pandas as pd
from sqlalchemy import create_engine
from src.alpha101_tmp import DataPrepare, Alpha101
from src.alpha101_tmp_zzh import Alpha101Z26


if __name__ == '__main__':
    alpha_data = DataPrepare()
    security = ('000001.XSHE', '000002.XSHE', '000004.XSHE', '000006.XSHE', '000007.XSHE')
    # 获取数据
    # stock_panel = alpha_data.get_basic_data("2010-01-01", "2018-12-30", stock_list=security)
    stock_panel = alpha_data.get_data()
    #
    stock_group = stock_panel.groupby('symbol')
    tmp = pd.DataFrame()
    for i, stock in stock_group:
        dependencies = ['close', 'open', 'high', 'low', 'pre_close', 'volume', 'money', 'trade_date', 'symbol',
                        'returns', 'vwap']
        stock = stock[dependencies]
        alpha = Alpha101(stock)
        alpha_ = Alpha101Z26(stock)
        stock['alpha001_1'] = alpha.alpha001()
        stock['alpha002_1'], stock['alpha002_2'] = alpha.alpha002()
        stock['alpha005_1'] = alpha.alpha005()
        stock['alpha006'] = alpha.alpha006()
        stock['alpha007'] = alpha.alpha007()
        stock['alpha008_1'] = alpha.alpha008()
        stock['alpha009'] = alpha.alpha009()
        stock['alpha010'] = alpha.alpha010()
        stock['alpha011_1'], stock['alpha011_2'], stock['alpha011_3'] = alpha.alpha011()
        stock['alpha012'] = alpha.alpha012()
        stock['alpha014_1'], stock['alpha014_2'] = alpha.alpha014()
        stock['alpha017_1'], stock['alpha017_2'], stock['alpha017_3'] = alpha.alpha017()
        stock['alpha018_2'] = alpha.alpha018()
        stock['alpha019_1'], stock['alpha019_2'] = alpha.alpha019()
        stock['alpha020_1'], stock['alpha020_2'], stock['alpha020_3'] = alpha.alpha020()
        stock['alpha021'] = alpha.alpha021()
        stock['alpha022_1'], stock['alpha022_2'] = alpha.alpha022()
        stock['alpha023'] = alpha.alpha023()
        stock['alpha024'] = alpha.alpha024()
        stock['alpha025_1'] = alpha.alpha025()

        stock['alpha028'] = alpha_.alpha028()
        stock['alpha029_1'], stock['alpha029_2'] = alpha_.alpha029()
        stock['alpha030_1'], stock['alpha030_2'] = alpha_.alpha030()
        stock['alpha031_1'], stock['alpha031_2'], stock['alpha031_3'] = alpha_.alpha031()
        stock['alpha032'] = alpha_.alpha032()
        stock['alpha033_1'] = alpha_.alpha033()
        stock['alpha034_1'], stock['alpha034_2'] = alpha_.alpha034()
        stock['alpha035_1'] = alpha_.alpha035()
        stock['alpha036_1'], stock['alpha036_2'], stock['alpha036_3'], stock['alpha036_4'], stock['alpha036_5'] = alpha_.alpha036()
        stock['alpha037_1'], stock['alpha037_2'] = alpha_.alpha037()
        stock['alpha038_1'] = alpha_.alpha038()
        stock['alpha039_1'], stock['alpha039_2'] = alpha_.alpha039()
        stock['alpha040_1'], stock['alpha040_2'] = alpha_.alpha040()
        stock['alpha041'] = alpha_.alpha041()
        stock['alpha042_1'], stock['alpha042_2'] = alpha_.alpha042()
        stock['alpha043_1'], stock['alpha043_2'] = alpha_.alpha043()
        stock['alpha045_1'], stock['alpha045_2'], stock['alpha045_3'] = alpha_.alpha045()
        stock['alpha046'] = alpha_.alpha046()
        stock['alpha047_1'], stock['alpha047_2'], stock['alpha047_3'], stock['alpha047_4'], stock['alpha047_5'] = alpha_.alpha047()
        stock['alpha049'] = alpha_.alpha049()
        stock['alpha051'] = alpha_.alpha051()
        stock['alpha052_1'], stock['alpha052_2'] = alpha_.alpha052()
        stock['alpha053'] = alpha_.alpha053()
        stock['alpha054'] = alpha_.alpha054()
        stock['alpha055_1'] = alpha_.alpha055()
        stock['alpha057_1'], stock['alpha057_2'] = alpha_.alpha057()
        stock['alpha060_1'], stock['alpha060_2'] = alpha_.alpha060()
        stock['alpha061_1'], stock['alpha061_2'] = alpha_.alpha061()
        stock['alpha062_1'], stock['alpha062_2'] = alpha_.alpha062()
        stock['alpha064_1'], stock['alpha064_2'] = alpha_.alpha064()

        stock['alpha065_1'], stock['alpha065_2'] = alpha.alpha065()
        stock['alpha066_1'], stock['alpha066_2'] = alpha.alpha066()
        stock['adv15'], stock['alpha068_1'] = alpha.alpha068()
        stock['alpha071_1'], stock['alpha071_2'] = alpha.alpha071()
        stock['alpha072_1'], stock['alpha072_2'] = alpha.alpha072()
        stock['alpha073_1'], stock['alpha073_2'] = alpha.alpha073()
        stock['alpha074_1'], stock['alpha074_2'] = alpha.alpha074()
        stock['adv50'], stock['alpha075_1'] = alpha.alpha075()
        stock['alpha077_1'], stock['alpha077_2'] = alpha.alpha077()
        stock['alpha078_1'] = alpha.alpha078()
        stock['alpha081_1'] = alpha.alpha081()
        stock['alpha083_1'], stock['alpha083_2'] = alpha.alpha083()
        stock['alpha084_1'], stock['alpha084_2'] = alpha.alpha084()
        stock['alpha085_1'], stock['alpha085_2'] = alpha.alpha085()
        stock['alpha086_1'], stock['alpha086_2'] = alpha.alpha086()
        stock['alpha088_1'] = alpha.alpha088()
        stock['adv30'], stock['alpha092_1'] = alpha.alpha092()
        stock['alpha094_1'], stock['alpha094_2'] = alpha.alpha094()
        stock['alpha095_1'], stock['alpha095_2'] = alpha.alpha095()
        stock['alpha096_1'] = alpha.alpha096()
        stock['adv15'], stock['alpha098_1'] = alpha.alpha098()
        stock['alpha099_1'], stock['alpha099_2'] = alpha.alpha099()
        stock['alpha101'] = alpha.alpha101()

        tmp = tmp.append(stock)
    # 计算结果保存到数据库
    tmp.to_csv('test.csv')
    test_sql_engine = create_engine("mysql+mysqlconnector://root:1234@10.15.97.128:3306/test")
    tmp.to_sql(name='alpha101_tmp', con=test_sql_engine, chunksize=1000, if_exists='replace', index=True)

    with test_sql_engine.connect() as con:
        con.execute('ALTER TABLE `alpha101_tmp_test` ADD PRIMARY KEY (`index`);')
#
