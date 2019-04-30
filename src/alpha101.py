#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
@version: ??
@author: li
@file: alpha.py
@time: 2019-04-16 14:06
"""
import time
from numpy import abs
from numpy import log
from numpy import sign
from sqlalchemy import create_engine
from src.factor_util import *


class Alpha101(object):
    def __init__(self, df_data):
        self.close = df_data['close']
        self.returns = df_data['returns']
        self.volume = df_data['volume']
        self.open = df_data['open']
        self.high = df_data['high']  # 最高价
        self.low = df_data['low']  # 最低价
        self.vwap = df_data['vwap']

        self.alpha001_1 = df_data['alpha001_1']
        self.alpha002_1 = df_data['alpha002_1']
        self.alpha002_2 = df_data['alpha002_1']
        self.alpha005_1= df_data['alpha005_1']
        self.alpha006_ = df_data['alpha006']
        self.alpha007_ = df_data['alpha007']
        self.alpha008_1 = df_data['alpha008_1']
        self.alpha009_ = df_data['alpha009']
        self.alpha010_ = df_data['alpha010']
        self.alpha011_1 = df_data['alpha011_1']
        self.alpha011_2 = df_data['alpha011_2']
        self.alpha011_3 = df_data['alpha011_3']
        self.alpha012_ = df_data['alpha012']
        self.alpha014_1 = df_data['alpha014_1']
        self.alpha014_2 = df_data['alpha014_2']
        self.alpha017_1 = df_data['alpha017_1']
        self.alpha017_2 = df_data['alpha017_2']
        self.alpha017_3 = df_data['alpha017_3']
        self.alpha018_2 = df_data['alpha018_2']
        self.alpha019_1 = df_data['alpha019_1']
        self.alpha019_2 = df_data['alpha019_2']
        self.alpha020_1 = df_data['alpha020_1']
        self.alpha020_2 = df_data['alpha020_2']
        self.alpha020_3 = df_data['alpha020_3']
        self.alpha021_ = df_data['alpha021']
        self.alpha022_1 = df_data['alpha022_1']
        self.alpha022_2 = df_data['alpha022_2']
        self.alpha023_ = df_data['alpha023']
        self.alpha024_ = df_data['alpha024']
        self.alpha025_1 = df_data['alpha025_1']

        self.alpha065_1 = df_data['alpha065_1']
        self.alpha065_2 = df_data['alpha065_2']
        self.alpha066_1 = df_data['alpha066_1']
        self.alpha066_2 = df_data['alpha066_2']
        self.adv15 = df_data['adv15']
        self.alpha068_1 = df_data['alpha068_1']
        self.alpha071_2 = df_data['alpha071_2']
        self.alpha071_1 = df_data['alpha071_1']
        self.alpha072_1 = df_data['alpha072_1']
        self.alpha072_2 = df_data['alpha072_2']
        self.alpha073_1 = df_data['alpha073_1']
        self.alpha073_2 = df_data['alpha073_2']
        self.alpha074_1 = df_data['alpha074_1']
        self.alpha074_2 = df_data['alpha074_2']
        self.alpha075_1 = df_data['alpha075_1']
        self.adv50 = df_data['adv50']
        self.alpha077_1 = df_data['alpha077_1']
        self.alpha077_2 = df_data['alpha077_2']
        self.alpha078_1 = df_data['alpha078_1']
        self.alpha081_1 = df_data['alpha081_1']
        self.alpha083_1 = df_data['alpha083_1']
        self.alpha083_2 = df_data['alpha083_2']
        self.alpha084_1 = df_data['alpha084_1']
        self.alpha084_2 = df_data['alpha084_2']
        self.alpha085_1 = df_data['alpha085_1']
        self.alpha085_2 = df_data['alpha085_2']
        self.alpha086_1 = df_data['alpha086_1']
        self.alpha086_2 = df_data['alpha086_2']
        self.alpha088_1 = df_data['alpha088_1']
        self.adv30 = df_data['adv30']
        self.alpha092_1 = df_data['alpha092_1']
        self.alpha094_1 = df_data['alpha094_1']
        self.alpha094_2 = df_data['alpha094_2']
        self.alpha095_1 = df_data['alpha095_1']
        self.alpha095_2 = df_data['alpha095_2']
        self.alpha096_1 = df_data['alpha096_1']
        self.alpha098_1 = df_data['alpha098_1']
        self.adv15 = df_data['adv15']
        self.alpha099_1 = df_data['alpha099_1']
        self.alpha099_2 = df_data['alpha099_2']
        self.alpha101_= df_data['alpha101']

    def alpha001(self):
        return rank(self.alpha001_1) - 0.5

    def alpha002(self):
        df = -1 * correlation(rank(self.alpha002_1), rank(self.alpha002_2), 6)
        return df.replace([-np.inf, np.inf], 0).fillna(value=0)

    def alpha003(self):
        df = -1 * correlation(rank(self.open), rank(self.volume), 10)
        return df.replace([-np.inf, np.inf], 0).fillna(value=0)

    def alpha004(self):
        return -1 * ts_rank(rank(self.low), 9)

    def alpha005(self):
        return rank(self.alpha005_1) * (-1 * abs(rank((self.close - self.vwap))))

    def alpha006(self):
        return self.alpha006_

    def alpha007(self):
        return self.alpha007_

    def alpha008(self):
        return -1 * (rank(self.alpha008_1))

    def alpha009(self):
        return self.alpha009_

    def alpha010(self):

        return self.alpha010_

    def alpha011(self):
        return (rank(self.alpha011_1) + rank(self.alpha011_2)) * rank(self.alpha011_3)

    def alpha012(self):
        return self.alpha012_

    def alpha013(self):
        return -1 * rank(covariance(rank(self.close), rank(self.volume), 5))

    def alpha014(self):
        return -1 * rank(self.alpha014_2) * self.alpha014_1

    def alpha015(self):
        df = correlation(rank(self.high), rank(self.volume), 3)
        df = df.replace([-np.inf, np.inf], 0).fillna(value=0)
        return -1 * ts_sum(rank(df), 3)

    def alpha016(self):
        return -1 * rank(covariance(rank(self.high), rank(self.volume), 5))

    def alpha017(self):
        return -1 * (rank(self.alpha017_1) * rank(self.alpha017_2) * rank(self.alpha017_3))

    def alpha018(self):
        return -1 * (rank(self.alpha018_2))

    def alpha019(self):
        return self.alpha019_1 * (1 + rank(self.alpha019_2))

    def alpha020(self):
        return -1 * (rank(self.alpha020_1) * rank(self.alpha020_2) * rank(self.alpha020_3))

    def alpha021(self):
        return self.alpha021_

    def alpha022(self):
        return self.alpha022_1 * rank(self.alpha022_2)

    def alpha023(self):
        return self.alpha023_

    def alpha024(self):
        return self.alpha024_

    def alpha025(self):
        return rank(self.alpha025_1)

    def alpha065(self):
        # 中间变量
        return (rank(self.alpha065_1) < rank(self.alpha065_2)) * -1

    def alpha066(self):

        return (rank(self.alpha066_1) + self.alpha066_2) * -1

    # 可能存在问题，我自己的数据测试了很多次值全为0，可能需要调整9,14这些参数？
    def alpha068(self):
        return (ts_rank(correlation(rank(self.high), rank(self.adv15), 9), 14) < rank(self.alpha068_1)) * -1

    def alpha071(self):
        adv180 = sma(self.volume, 180)
        p2 = ts_rank(decay_linear_pn((rank(self.alpha071_2).pow(2)), 16), 4)
        func = lambda x: x[0] if x[0] >= x[1] else x[1]
        pn = pd.Panel({'p1': self.alpha071_1, 'p2': p2})
        re_df = pd.DataFrame()
        for i in pn.minor_axis:
            re_df[i] = pn.minor_xs(i).apply(func, axis=1)
        return re_df
        # 就是按行求p1、p2两个series中最大值问题，max(p1,p2)会报错，有简单写法的请告诉我

    def alpha072(self):
        return rank(self.alpha072_1) / rank(self.alpha072_2)

    def alpha073(self):
        p1 = rank(self.alpha073_1)
        func = lambda x: x[0] if x[0] >= x[1] else x[1]
        pn = pd.Panel({'p1': p1, 'p2': self.alpha073_2})
        re_df = pd.DataFrame()
        for i in pn.minor_axis:
            re_df[i] = pn.minor_xs(i).apply(func, axis=1)
        return -1 * re_df

    def alpha074(self):
        return (rank(self.alpha074_1) < rank(correlation(rank(self.alpha074_2), rank(self.volume), 11))) * -1

    def alpha075(self):
        return rank(self.alpha075_1) < rank(correlation(rank(self.low), rank(self.adv50), 12))

    def alpha077(self):
        p1 = rank(self.alpha077_1)
        p2 = rank(self.alpha077_2)
        func = lambda x: x[0] if x[0] <= x[1] else x[1]
        pn = pd.Panel({'p1': p1, 'p2': p2})
        re_df = pd.DataFrame()
        for i in pn.minor_axis:
            re_df[i] = pn.minor_xs(i).apply(func, axis=1)
        return re_df

    def alpha078(self):
        return rank(self.alpha078_1).pow(rank(correlation(rank(self.vwap), rank(self.volume), 6)))

    def alpha081(self):
        return (rank(log(product(rank((rank(self.alpha081_1).pow(4))), 15))) < rank(correlation(rank(self.vwap), rank(self.volume), 5))) * -1

    def alpha083(self):
        return (rank(self.alpha083_1) * rank(rank(self.volume))) / self.alpha083_2

    def alpha084(self):
        return pow(ts_rank(self.alpha084_1, 21), self.alpha084_2)

    def alpha085(self):
        return rank(self.alpha085_1).pow(rank(correlation(ts_rank(self.alpha085_2, 4), ts_rank(self.volume, 10), 7)))

    # 可能存在问题，我自己的数据测试了很多次值全为0，可能需要调整15，,6,20这些参数？
    def alpha086(self):
        return (self.alpha086_1 < rank(self.alpha086_2)) * -1

    def alpha088(self):
        p1 = rank(decay_linear_pn(((rank(self.open) + rank(self.low)) - (rank(self.high) + rank(self.close))), 8))
        func = lambda x: x[0] if x[0] <= x[1] else x[1]
        pn = pd.Panel({'p1': p1, 'p2': self.alpha088_1})
        re_df = pd.DataFrame()
        for i in pn.minor_axis:
            re_df[i] = pn.minor_xs(i).apply(func, axis=1)
        return re_df

    def alpha092(self):
        alpha092_2 = ts_rank(decay_linear_pn(correlation(rank(self.low), rank(self.adv30), 8), 7), 7)
        func = lambda x: x[0] if x[0] <= x[1] else x[1]
        pn = pd.Panel({'p1': self.alpha092_1, 'p2': alpha092_2})
        re_df = pd.DataFrame()
        for i in pn.minor_axis:
            re_df[i] = pn.minor_xs(i).apply(func, axis=1)
        return re_df

    def alpha094(self):
        return ((rank(self.alpha094_1).pow(self.alpha094_2) * -1))

    def alpha095(self):
        return rank(self.alpha095_1) < ts_rank((rank(self.alpha095_2).pow(5)), 12)

    def alpha096(self):
        p1 = ts_rank(decay_linear_pn(correlation(rank(self.vwap), rank(self.volume), 4), 4), 8)
        func = lambda x: x[0] if x[0] >= x[1] else x[1]
        pn = pd.Panel({'p1': p1, 'p2': self.alpha096_1})
        re_df = pd.DataFrame()
        for i in pn.minor_axis:
            re_df[i] = pn.minor_xs(i).apply(func, axis=1)
        return -1 * re_df

    def alpha098(self):
        return rank(self.alpha098_1) - rank(decay_linear_pn(ts_rank(ts_argmin(correlation(rank(self.open), rank(self.adv15), 21), 9), 7), 8))

    def alpha099(self):
        return (rank(self.alpha099_1) < rank(self.alpha099_2)) * -1

    def alpha101(self):
        return self.alpha101_

    def calculate(self):
        alpha001 = self.alpha001()
        alpha002 = self.alpha002()
        alpha003 = self.alpha003()
        alpha004 = self.alpha004()
        alpha005 = self.alpha005()
        alpha006 = self.alpha006()
        alpha007 = self.alpha007()
        alpha008 = self.alpha008()
        alpha009 = self.alpha009()
        alpha010 = self.alpha010()
        alpha011 = self.alpha011()
        alpha012 = self.alpha012()
        alpha013 = self.alpha013()
        alpha014 = self.alpha014()
        alpha015 = self.alpha015()
        alpha016 = self.alpha016()
        alpha017 = self.alpha017()
        alpha018 = self.alpha018()
        alpha019 = self.alpha019()
        alpha020 = self.alpha020()
        alpha021 = self.alpha021()
        alpha022 = self.alpha022()
        alpha023 = self.alpha023()
        alpha024 = self.alpha024()
        alpha025 = self.alpha025()

        alpha065 = self.alpha065()
        alpha066 = self.alpha066()
        alpha068 = self.alpha068()
        alpha071 = self.alpha071()
        alpha072 = self.alpha072()
        alpha073 = self.alpha073()
        alpha074 = self.alpha074()
        alpha075 = self.alpha075()
        alpha077 = self.alpha077()
        alpha078 = self.alpha078()
        alpha081 = self.alpha081()
        alpha083 = self.alpha083()
        alpha084 = self.alpha084()
        alpha085 = self.alpha085()
        alpha086 = self.alpha086()
        alpha088 = self.alpha088()
        alpha092 = self.alpha092()
        alpha094 = self.alpha094()
        alpha095 = self.alpha095()
        alpha096 = self.alpha096()
        alpha098 = self.alpha098()
        alpha099 = self.alpha099()
        alpha101 = self.alpha101()

        tmp_pn = pd.Panel({'alpha001': alpha001, 'alpha002': alpha002, 'alpha003': alpha003, 'alpha004': alpha004,
                           'alpha005': alpha005, 'alpha006': alpha006, 'alpha007': alpha007, 'alpha008': alpha008,
                           'alpha009': alpha009, 'alpha010': alpha010, 'alpha011': alpha011, 'alpha012': alpha012,
                           'alpha013': alpha013, 'alpha014': alpha014, 'alpha015': alpha015, 'alpha016': alpha016,
                           'alpha017': alpha017, 'alpha018': alpha018, 'alpha019': alpha019, 'alpha020': alpha020,
                           'alpha021': alpha021, 'alpha022': alpha022, 'alpha023': alpha023, 'alpha024': alpha024,
                           'alpha025': alpha025,
                           'alpha065': alpha065, 'alpha066': alpha066, 'alpha068': alpha068, 'alpha071': alpha071,
                           'alpha072': alpha072, 'alpha073': alpha073, 'alpha074': alpha074, 'alpha075': alpha075,
                           'alpha077': alpha077, 'alpha078': alpha078, 'alpha081': alpha081, 'alpha083': alpha083,
                           'alpha084': alpha084, 'alpha085': alpha085, 'alpha086': alpha086, 'alpha088': alpha088,
                           'alpha092': alpha092, 'alpha094': alpha094, 'alpha095': alpha095, 'alpha096': alpha096,
                           'alpha098': alpha098, 'alpha099': alpha099, 'alpha101': alpha101,
                           })

        return tmp_pn


class DataPrepare(object):
    def __init__(self):
        pass

    def get_basic_data(self, start_date, end_date, stock_list):
        dependencies = ['close', 'open', 'high', 'low', 'pre_close', 'volume', 'money', 'trade_date', 'symbol']
        sql = "select * from test.alpha101_tmp_test where trade_date >='{}' and trade_date <='{}' and symbol in {}".format(start_date, end_date, stock_list)
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
    pn = alpha.calculate()
    print('spend time %s' % (time.time() - start_time))
    print(pn.minor_xs('600519.XSHG'))

