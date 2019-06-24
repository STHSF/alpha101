#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
@version: ??
@author: li
@file: alpha.py
@time: 2019-04-16 14:06
"""
import pysnooper
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

    @pysnooper.snoop(prefix='alpha001:')
    def alpha001(self):
        inner = self.close
        inner[self.returns < 0] = stddev(self.returns, 20)
        alpha001_1 = ts_argmax(inner ** 2, 5)
        alpha001_1 = alpha001_1.replace([-np.inf, np.inf], 0).fillna(value=0)
        return alpha001_1

    def alpha002(self):
        alpha002_1 = delta(log(self.volume), 2)
        alpha002_2 = (self.close - self.open) / self.open
        # 自行修改， 如果不添加，数据中则会出现inf和-inf值,
        # 写入数据库时会出错（(mysql.connector.errors.ProgrammingError) 1054 (42S22): Unknown column 'inf' in 'field list'）
        alpha002_1 = alpha002_1.replace([-np.inf, np.inf], 0).fillna(value=0)
        alpha002_2 = alpha002_2.replace([-np.inf, np.inf], 0).fillna(value=0)
        return alpha002_1, alpha002_2

    def alpha003(self):
        df = -1 * correlation(rank(self.open), rank(self.volume), 10)
        return df.replace([-np.inf, np.inf], 0).fillna(value=0)

    def alpha004(self):
        return -1 * ts_rank(rank(self.low), 9)

    def alpha005(self):
        alpha005_1 = (self.open - (sum(self.vwap, 10) / 10))
        alpha005_1 = alpha005_1.replace([-np.inf, np.inf], 0).fillna(value=0)
        return alpha005_1
        # return (rank(alpha005_1) * (-1 * abs(rank((self.close - self.vwap)))))

    def alpha006(self):
        # 直接写数据库
        alpha006_1 = -1 * correlation(self.open, self.volume, 10)
        return alpha006_1.replace([-np.inf, np.inf], 0).fillna(value=0)

    def alpha007(self):
        # 直接写数据库
        adv20 = sma(self.volume, 20)
        alpha007_1 = -1 * ts_rank(abs(delta(self.close, 7)), 60) * sign(delta(self.close, 7))
        alpha007_1[adv20 >= self.volume] = -1
        alpha007_1 = alpha007_1.replace([-np.inf, np.inf], 0).fillna(value=0)
        return alpha007_1

    def alpha008(self):
        alpha008_1 = ((ts_sum(self.open, 5) * ts_sum(self.returns, 5)) - delay(
            (ts_sum(self.open, 5) * ts_sum(self.returns, 5)), 10))

        return alpha008_1.replace([-np.inf, np.inf], 0).fillna(value=0)
        # return -1 * (rank(alpha008_1))

    def alpha009(self):
        # 直接写数据库
        delta_close = delta(self.close, 1)
        cond_1 = ts_min(delta_close, 5) > 0
        cond_2 = ts_max(delta_close, 5) < 0
        alpha = -1 * delta_close
        alpha[cond_1 | cond_2] = delta_close
        return alpha

    def alpha010(self):
        # 直接写数据库
        delta_close = delta(self.close, 1)
        cond_1 = ts_min(delta_close, 4) > 0
        cond_2 = ts_max(delta_close, 4) < 0
        alpha = -1 * delta_close
        alpha[cond_1 | cond_2] = delta_close
        return alpha

    def alpha011(self):
        alpha011_1 = ts_max((self.vwap - self.close), 3).replace([-np.inf, np.inf], 0).fillna(value=0)
        alpha011_2 = ts_min((self.vwap - self.close), 3).replace([-np.inf, np.inf], 0).fillna(value=0)
        alpha011_3 = delta(self.volume, 3).replace([-np.inf, np.inf], 0).fillna(value=0)
        return alpha011_1, alpha011_2, alpha011_3

    def alpha012(self):
        # 直接写数据库
        return sign(delta(self.volume, 1)) * (-1 * delta(self.close, 1))

    def alpha013(self):
        return -1 * rank(covariance(rank(self.close), rank(self.volume), 5))

    def alpha014(self):
        alpha014_1 = correlation(self.open, self.volume, 10)
        alpha014_1 = alpha014_1.replace([-np.inf, np.inf], 0).fillna(value=0)
        alpha014_2 = delta(self.returns, 3).replace([-np.inf, np.inf], 0).fillna(value=0)
        return alpha014_1, alpha014_2

    def alpha015(self):
        df = correlation(rank(self.high), rank(self.volume), 3)
        df = df.replace([-np.inf, np.inf], 0).fillna(value=0)
        return -1 * ts_sum(rank(df), 3)

    def alpha016(self):
        return -1 * rank(covariance(rank(self.high), rank(self.volume), 5))

    def alpha017(self):
        adv20 = sma(self.volume, 20)
        alpha017_1 = ts_rank(self.close, 10).replace([-np.inf, np.inf], 0).fillna(value=0)
        alpha017_2 = delta(delta(self.close, 1), 1).replace([-np.inf, np.inf], 0).fillna(value=0)
        alpha017_3 = ts_rank((self.volume / adv20), 5).replace([-np.inf, np.inf], 0).fillna(value=0)
        return alpha017_1, alpha017_2, alpha017_3

    def alpha018(self):
        alpha018_1 = correlation(self.close, self.open, 10)
        alpha018_1 = alpha018_1.replace([-np.inf, np.inf], 0).fillna(value=0)

        alpha018_2 = (stddev(abs((self.close - self.open)), 5) + (self.close - self.open)) + alpha018_1
        return alpha018_2.replace([-np.inf, np.inf], 0).fillna(value=0)

    def alpha019(self):
        alpha019_1 = (-1 * sign((self.close - delay(self.close, 7)) + delta(self.close, 7))).replace([-np.inf, np.inf], 0).fillna(value=0)
        alpha019_2 = 1 + ts_sum(self.returns, 250)
        return alpha019_1, alpha019_2.replace([-np.inf, np.inf], 0).fillna(value=0)

    def alpha020(self):
        alpha020_1 = self.open - delay(self.high, 1).replace([-np.inf, np.inf], 0).fillna(value=0)
        alpha020_2 = self.open - delay(self.close, 1).replace([-np.inf, np.inf], 0).fillna(value=0)
        alpha020_3 = self.open - delay(self.low, 1).replace([-np.inf, np.inf], 0).fillna(value=0)
        return alpha020_1, alpha020_2, alpha020_3
        # return -1 * (rank(alpha020_1) * rank(alpha020_2) * rank(alpha020_3))

    def alpha021(self):
        # 直接写数据库
        cond_1 = sma(self.close, 8) + stddev(self.close, 8) < sma(self.close, 2)
        cond_2 = sma(self.volume, 20) / self.volume < 1
        alpha = pd.DataFrame(np.ones_like(self.close), index=self.close.index)
        #        alpha = pd.DataFrame(np.ones_like(self.close), index=self.close.index,
        #                             columns=self.close.columns)
        alpha[cond_1 | cond_2] = -1
        return alpha

    def alpha022(self):
        df = correlation(self.high, self.volume, 5)
        df = df.replace([-np.inf, np.inf], 0).fillna(value=0)
        alpha022_1 = -1 * delta(df, 5)
        alpha022_1 = alpha022_1.replace([-np.inf, np.inf], 0).fillna(value=0)
        alpha022_2 = stddev(self.close, 20).replace([-np.inf, np.inf], 0).fillna(value=0)
        return alpha022_1, alpha022_2
        # return alpha022_1 * rank(alpha022_2)

    def alpha023(self):
        # 直接写数据库
        cond = sma(self.high, 20) < self.high
        alpha = pd.DataFrame(np.zeros_like(self.close), index=self.close.index, columns=['close'])
        alpha.at[cond, 'close'] = -1 * delta(self.high, 2).fillna(value=0)
        return alpha

    def alpha024(self):
        # 直接写数据库
        cond = delta(sma(self.close, 100), 100) / delay(self.close, 100) <= 0.05
        alpha = -1 * delta(self.close, 3)
        alpha[cond] = -1 * (self.close - ts_min(self.close, 100))
        return alpha

    def alpha025(self):

        adv20 = sma(self.volume, 20)
        alpha025_1 = ((((-1 * self.returns) * adv20) * self.vwap) * (self.high - self.close))
        return alpha025_1.replace([-np.inf, np.inf], 0).fillna(value=0)

    def alpha065(self):
        # 中间变量
        adv60 = sma(self.volume, 60)
        alpha065_1 = correlation(((self.open * 0.00817205) + (self.vwap * (1 - 0.00817205))), sma(adv60, 9), 6)
        alpha065_1 = alpha065_1.replace([-np.inf, np.inf], 0).fillna(value=0)
        alpha065_2 = (self.open - ts_min(self.open, 14)).replace([-np.inf, np.inf], 0).fillna(value=0)

        return alpha065_1, alpha065_2

    def alpha066(self):

        alpha066_1 = decay_linear(delta(self.vwap, 4).to_frame(), 7).CLOSE
        alpha066_2 = ts_rank(decay_linear(((((self.low * 0.96633) + (self.low * (1 - 0.96633))) - self.vwap) / (self.open - ((self.high + self.low) / 2))).to_frame(), 11).CLOSE, 7)
        alpha066_1 = alpha066_1.replace([-np.inf, np.inf], 0).fillna(value=0)
        alpha066_2 = alpha066_2.replace([-np.inf, np.inf], 0).fillna(value=0)
        return alpha066_1, alpha066_2

    # 可能存在问题，我自己的数据测试了很多次值全为0，可能需要调整9,14这些参数？
    def alpha068(self):
        adv15 = sma(self.volume, 15)
        # 中间变量
        alpha068_1 = delta(((self.close * 0.518371) + (self.low * (1 - 0.518371))), 1.06157)
        alpha068_1 = alpha068_1.replace([-np.inf, np.inf], 0).fillna(value=0)
        return adv15, alpha068_1

    def alpha071(self):

        # 中间变量
        adv180 = sma(self.volume, 180)

        # alpha071_1 = ts_rank(decay_linear_pn(correlation(ts_rank(self.close, 3), ts_rank(adv180, 12), 18), 4), 16)
        alpha071_1 = ts_rank(decay_linear(correlation(ts_rank(self.close, 3), ts_rank(adv180, 12), 18).to_frame(), 4).CLOSE, 16)
        alpha071_2 = ((self.low + self.open) - (self.vwap + self.vwap))
        alpha071_1 = alpha071_1.replace([-np.inf, np.inf], 0).fillna(value=0)
        alpha071_2 = alpha071_2.replace([-np.inf, np.inf], 0).fillna(value=0)
        return alpha071_1, alpha071_2

    def alpha072(self):
        adv40 = sma(self.volume, 40)

        alpha072_1 = decay_linear(correlation(((self.high + self.low) / 2), adv40, 9).to_frame(), 10).CLOSE
        alpha072_2 = decay_linear(correlation(ts_rank(self.vwap, 4), ts_rank(self.volume, 19), 7).to_frame(), 3).CLOSE
        alpha072_1 = alpha072_1.replace([-np.inf, np.inf], 0).fillna(value=0)
        alpha072_2 = alpha072_2.replace([-np.inf, np.inf], 0).fillna(value=0)
        return alpha072_1, alpha072_2

    def alpha073(self):

        alpha073_1 = decay_linear(delta(self.vwap, 5).to_frame(), 3).CLOSE
        alpha073_2 = ts_rank(decay_linear(((delta(((self.open * 0.147155) + (self.low * (1 - 0.147155))), 2) / ((self.open * 0.147155) + (self.low * (1 - 0.147155)))) * -1).to_frame(), 3).CLOSE, 17)
        alpha073_1 = alpha073_1.replace([-np.inf, np.inf], 0).fillna(value=0)
        alpha073_2 = alpha073_2.replace([-np.inf, np.inf], 0).fillna(value=0)
        return alpha073_1, alpha073_2

    def alpha074(self):
        adv30 = sma(self.volume, 30)
        alpha074_1 = correlation(self.close, sma(adv30, 37), 15)
        alpha074_2 = ((self.high * 0.0261661) + (self.vwap * (1 - 0.0261661)))

        alpha074_1 = alpha074_1.replace([-np.inf, np.inf], 0).fillna(value=0)
        alpha074_2 = alpha074_2.replace([-np.inf, np.inf], 0).fillna(value=0)

        return alpha074_1, alpha074_2

    def alpha075(self):
        adv50 = sma(self.volume, 50)

        alpha075_1 = correlation(self.vwap, self.volume, 4)
        alpha075_1 = alpha075_1.replace([-np.inf, np.inf], 0).fillna(value=0)
        return adv50, alpha075_1

    def alpha077(self):
        adv40 = sma(self.volume, 40)
        alpha077_1 = decay_linear(((((self.high + self.low) / 2) + self.high) - (self.vwap + self.high)).to_frame(), 20).CLOSE
        alpha077_2 = decay_linear(correlation(((self.high + self.low) / 2), adv40, 3).to_frame(), 6).CLOSE

        alpha077_1 = alpha077_1.replace([-np.inf, np.inf], 0).fillna(value=0)
        alpha077_2 = alpha077_2.replace([-np.inf, np.inf], 0).fillna(value=0)
        return alpha077_1, alpha077_2

    def alpha078(self):
        adv40 = sma(self.volume, 40)
        alpha078_1 = correlation(ts_sum(((self.low * 0.352233) + (self.vwap * (1 - 0.352233))), 20), ts_sum(adv40, 20), 7)
        alpha078_1 = alpha078_1.replace([-np.inf, np.inf], 0).fillna(value=0)
        return alpha078_1

    def alpha081(self):
        adv10 = sma(self.volume, 10)
        alpha081_1 = correlation(self.vwap, ts_sum(adv10, 50), 8)
        alpha081_1 = alpha081_1.replace([-np.inf, np.inf], 0).fillna(value=0)
        return alpha081_1

    def alpha083(self):

        alpha083_1 = delay(((self.high - self.low) / (ts_sum(self.close, 5) / 5)), 2)
        alpha083_2 = (((self.high - self.low) / (ts_sum(self.close, 5) / 5)) / (self.vwap - self.close))
        alpha083_1 = alpha083_1.replace([-np.inf, np.inf], 0).fillna(value=0)
        alpha083_2 = alpha083_2.replace([-np.inf, np.inf], 0).fillna(value=0)
        return alpha083_1, alpha083_2

    def alpha084(self):

        alpha084_1 = (self.vwap - ts_max(self.vwap, 15))
        alpha084_2 = delta(self.close, 5)
        alpha084_1 = alpha084_1.replace([-np.inf, np.inf], 0).fillna(value=0)
        alpha084_2 = alpha084_2.replace([-np.inf, np.inf], 0).fillna(value=0)
        return alpha084_1, alpha084_2

    def alpha085(self):
        adv30 = sma(self.volume, 30)

        alpha085_1 = correlation(((self.high * 0.876703) + (self.close * (1 - 0.876703))), adv30, 10)
        alpha085_2 = ((self.high + self.low) / 2)
        alpha085_1 = alpha085_1.replace([-np.inf, np.inf], 0).fillna(value=0)
        alpha085_2 = alpha085_2.replace([-np.inf, np.inf], 0).fillna(value=0)
        return alpha085_1, alpha085_2

    # 可能存在问题，我自己的数据测试了很多次值全为0，可能需要调整15，,6,20这些参数？
    def alpha086(self):
        adv20 = sma(self.volume, 20)
        alpha086_1 = ts_rank(correlation(self.close, sma(adv20, 15), 6), 20)
        alpha086_2 = ((self.open + self.close) - (self.vwap + self.open))

        alpha086_1 = alpha086_1.replace([-np.inf, np.inf], 0).fillna(value=0)
        alpha086_2 = alpha086_2.replace([-np.inf, np.inf], 0).fillna(value=0)
        return alpha086_1, alpha086_2


    def alpha088(self):
        adv60 = sma(self.volume, 60)

        alpha088_1 = ts_rank(decay_linear(correlation(ts_rank(self.close, 8), ts_rank(adv60, 21), 8).to_frame(), 7).CLOSE, 3)
        alpha088_1 = alpha088_1.replace([-np.inf, np.inf], 0).fillna(value=0)
        return alpha088_1

    def alpha092(self):
        adv30 = sma(self.volume, 30)
        alpha092_1 = ts_rank(decay_linear(((((self.high + self.low) / 2) + self.close) < (self.low + self.open)).to_frame(), 15).CLOSE, 19)
        alpha092_1 = alpha092_1.replace([-np.inf, np.inf], 0).fillna(value=0)
        return adv30, alpha092_1

    def alpha094(self):
        adv60 = sma(self.volume, 60)
        alpha094_1 = (self.vwap - ts_min(self.vwap, 12))
        alpha094_2 = ts_rank(correlation(ts_rank(self.vwap, 20), ts_rank(adv60, 4), 18), 3)

        alpha094_1 = alpha094_1.replace([-np.inf, np.inf], 0).fillna(value=0)
        alpha094_2 = alpha094_2.replace([-np.inf, np.inf], 0).fillna(value=0)
        return alpha094_1, alpha094_2

    def alpha095(self):
        adv40 = sma(self.volume, 40)
        alpha095_1 = (self.open - ts_min(self.open, 12))
        alpha095_2 = correlation(sma(((self.high + self.low) / 2), 19), sma(adv40, 19), 13)

        alpha095_1 = alpha095_1.replace([-np.inf, np.inf], 0).fillna(value=0)
        alpha095_2 = alpha095_2.replace([-np.inf, np.inf], 0).fillna(value=0)
        return alpha095_1, alpha095_2

    def alpha096(self):
        adv60 = sma(self.volume, 60)
        alpha096_1 = ts_rank(decay_linear(ts_argmax(correlation(ts_rank(self.close, 7), ts_rank(adv60, 4), 4), 13).to_frame(), 14).CLOSE, 13)
        alpha096_1 = alpha096_1.replace([-np.inf, np.inf], 0).fillna(value=0)
        return alpha096_1

    def alpha098(self):
        adv5 = sma(self.volume, 5)
        adv15 = sma(self.volume, 15)
        # alpha098_1 = decay_linear_pn(correlation(self.vwap, sma(adv5, 26), 5), 7)
        alpha098_1 = decay_linear(correlation(self.vwap, sma(adv5, 26), 5).to_frame(), 7).CLOSE
        alpha098_1 = alpha098_1.replace([-np.inf, np.inf], 0).fillna(value=0)
        return adv15, alpha098_1

    def alpha099(self):
        adv60 = sma(self.volume, 60)
        alpha099_1 = correlation(ts_sum(((self.high + self.low) / 2), 20), ts_sum(adv60, 20), 9)
        alpha099_2 = correlation(self.low, self.volume, 6)
        alpha099_1 = alpha099_1.replace([-np.inf, np.inf], 0).fillna(value=0)
        alpha099_2 = alpha099_2.replace([-np.inf, np.inf], 0).fillna(value=0)
        return alpha099_1, alpha099_2

    def alpha101(self):
        # 直接写数据库
        return (self.close - self.open) / ((self.high - self.low) + 0.001)


class DataPrepare(object):
    def __init__(self):
        pass

    def get_basic_data(self, start_date, end_date, stock_list):
        dependencies = ['close', 'open', 'high', 'low', 'pre_close', 'volume', 'money', 'trade_date', 'symbol']
        sql = "select * from vision.sk_daily_price where trade_date >='{}' and trade_date <='{}' and symbol in {}".format(
            start_date, end_date, stock_list)
        sql_engine = create_engine("mysql+mysqlconnector://root:1234@10.15.97.128:3306/vision")

        sql_data = pd.read_sql(sql, sql_engine)
        # 先转化成panel
        res = sql_data[dependencies].set_index(['trade_date', 'symbol']).to_panel()
        res['returns'] = res['close'] / res['pre_close'] - 1
        res['vwap'] = (res['money'] * 1000) / (res['volume'] * 100 + 1)

        return res

    @pysnooper.snoop()
    def get_data(self):
        dependencies = ['close', 'open', 'high', 'low', 'pre_close', 'volume', 'money', 'trade_date', 'symbol']
        sql = "select * from vision.sk_daily_price"
        # sql = "select * from vision.sk_daily_price where symbol in {}".format(stock_list)
        sql_engine = create_engine("mysql+mysqlconnector://root:1234@10.15.97.128:3306/vision")
        sql_data = pd.read_sql(sql, sql_engine)
        # 先转化成multiindex
        sql_data['returns'] = sql_data['close'] / sql_data['pre_close'] - 1
        sql_data['vwap'] = (sql_data['money'] * 1000) / (sql_data['volume'] * 100 + 1)
        return sql_data


if __name__ == '__main__':
    alpha_data = DataPrepare()
    security = ('000001.XSHE', '000002.XSHE', '000004.XSHE','000006.XSHE', '000007.XSHE')
    # 获取数据
    # stock_panel = alpha_data.get_basic_data("2010-01-01", "2018-12-30", stock_list=security)
    stock_panel = alpha_data.get_data()
    #
    stock_group = stock_panel.groupby('symbol')
    tmp = pd.DataFrame()
    for i, stock in stock_group:
        dependencies = ['close', 'open', 'high', 'low', 'pre_close', 'volume', 'money', 'trade_date', 'symbol', 'returns', 'vwap']
        stock = stock[dependencies]
        alpha = Alpha101(stock)
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
    # tmp.to_csv('test.csv')
    test_sql_engine = create_engine("mysql+mysqlconnector://root:1234@10.15.97.128:3306/test")
    tmp.to_sql(name='alpha101_tmp_test', con=test_sql_engine, chunksize=1000, if_exists='replace', index=True)

    with test_sql_engine.connect() as con:
        con.execute('ALTER TABLE `alpha101_tmp_test` ADD PRIMARY KEY (`index`);')

