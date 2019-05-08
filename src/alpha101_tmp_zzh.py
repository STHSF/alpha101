#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
@version: ??
@author: zzh
@file: alpha.py
@time: 2019-05-06 11:06
"""
# import pysnooper
from numpy import abs
from numpy import log
from numpy import sign
from sqlalchemy import create_engine
from src.factor_util import *


class Alpha101Z26(object):
    def __init__(self, df_data):
        self.close = df_data['close']
        self.returns = df_data['returns']
        self.volume = df_data['volume']
        self.open = df_data['open']
        self.high = df_data['high']  # 最高价
        self.low = df_data['low']  # 最低价
        self.vwap = df_data['vwap']

    # Alpha#26	 (-1 * ts_max(correlation(ts_rank(volume, 5), ts_rank(high, 5), 5), 3))
    # 不可预计算
    def alpha026(self):
        return None

    # Alpha#27	 ((0.5 < rank((sum(correlation(rank(volume), rank(vwap), 6), 2) / 2.0))) ? (-1 * 1) : 1)
    # 可能存在问题，我自己的数据测试了很多次值全为1，可能需要调整6,2这些参数？
    # 不可预计算
    def alpha027(self):
        return None

        # Alpha#28	 scale(((correlation(adv20, low, 5) + ((high + low) / 2)) - close))

    def alpha028(self):
        adv20 = sma(self.volume, 20)
        df = correlation(adv20, self.low, 5)
        df = df.replace([-np.inf, np.inf], 0).fillna(value=0)
        return scale(((df + ((self.high + self.low) / 2)) - self.close))

    # Alpha#29	 (min(product(rank(rank(scale(log(sum(ts_min(rank(rank((-1 * rank(delta((close - 1),5))))), 2), 1))))), 1), 5) + ts_rank(delay((-1 * returns), 6), 5))
    def alpha029(self):
        alpha029_1 = delay((-1 * self.returns), 6)
        alpha029_2 = delta((self.close - 1), 5)
        return alpha029_1, alpha029_2

    # Alpha#30	 (((1.0 - rank(((sign((close - delay(close, 1))) + sign((delay(close, 1) - delay(close, 2)))) +sign((delay(close, 2) - delay(close, 3)))))) * sum(volume, 5)) / sum(volume, 20))
    def alpha030(self):
        delta_close = delta(self.close, 1)
        alpha030_1 = sign(delta_close) + sign(delay(delta_close, 1)) + sign(delay(delta_close, 2))
        alpha030_2 = ts_sum(self.volume, 5) / ts_sum(self.volume, 20)
        alpha030_2 = alpha030_2.replace([-np.inf, np.inf], 0).fillna(value=0)
        return alpha030_1, alpha030_2

    # Alpha#31	 ((rank(rank(rank(decay_linear_pn((-1 * rank(rank(delta(close, 10)))), 10)))) + rank((-1 *delta(close, 3)))) + sign(scale(correlation(adv20, low, 12))))
    def alpha031(self):
        alpha031_1 = delta(self.close, 10)
        alpha031_2 = (-1 * delta(self.close, 3))
        adv20 = sma(self.volume, 20)
        df = correlation(adv20, self.low, 12).replace([-np.inf, np.inf], 0).fillna(value=0)
        alpha031_3 = sign(scale(df))
        return alpha031_1, alpha031_2, alpha031_3

    # Alpha#32	 (scale(((sum(close, 7) / 7) - close)) + (20 * scale(correlation(vwap, delay(close, 5),230))))
    def alpha032(self):
        return scale(((sma(self.close, 7) / 7) - self.close)) + (
                20 * scale(correlation(self.vwap, delay(self.close, 5), 230))).replace([-np.inf, np.inf], 0).fillna(
            value=0)

    # Alpha#33	 rank((-1 * ((1 - (open / close))^1)))
    def alpha033(self):
        alpha033_1 = (-1 + (self.open / self.close))
        alpha033_1 = alpha033_1.replace([-np.inf, np.inf], 0).fillna(value=0)
        return alpha033_1

    # Alpha#34	 rank(((1 - rank((stddev(returns, 2) / stddev(returns, 5)))) + (1 - rank(delta(close, 1)))))
    def alpha034(self):
        inner = stddev(self.returns, 2) / stddev(self.returns, 5)
        alpha034_1 = inner.replace([-np.inf, np.inf], 1).fillna(value=1)
        alpha034_2 = delta(self.close, 1)
        return alpha034_1, alpha034_2

    # Alpha#35	 ((Ts_Rank(volume, 32) * (1 - Ts_Rank(((close + high) - low), 16))) * (1 -Ts_Rank(returns, 32)))
    def alpha035(self):
        alpha035_1 = self.close + self.high - self.low
        return alpha035_1

    # Alpha#36	 (((((2.21 * rank(correlation((close - open), delay(volume, 1), 15))) + (0.7 * rank((open- close)))) + (0.73 * rank(Ts_Rank(delay((-1 * returns), 6), 5)))) + rank(abs(correlation(vwap,adv20, 6)))) + (0.6 * rank((((sum(close, 200) / 200) - open) * (close - open)))))
    def alpha036(self):
        adv20 = sma(self.volume, 20)
        alpha036_1 = correlation((self.close - self.open), delay(self.volume, 1), 15).replace([-np.inf, np.inf],
                                                                                              0).fillna(value=0)
        alpha036_2 = self.open - self.close
        alpha036_3 = delay((-1 * self.returns), 6).replace([-np.inf, np.inf], 0).fillna(value=0)
        alpha036_4 = abs(correlation(self.vwap, adv20, 6)).replace([-np.inf, np.inf], 0).fillna(value=0)
        alpha036_5 = ((sma(self.close, 200) / 200) - self.open) * (self.close - self.open).replace([-np.inf, np.inf],
                                                                                                   0).fillna(value=0)
        return alpha036_1, alpha036_2, alpha036_3, alpha036_4, alpha036_5

    # Alpha#37	 (rank(correlation(delay((open - close), 1), close, 200)) + rank((open - close)))
    def alpha037(self):
        alpha037_1 = correlation(delay(self.open - self.close, 1), self.close, 200).replace([-np.inf, np.inf],
                                                                                            0).fillna(value=0)
        alpha037_2 = self.open - self.close
        return alpha037_1, alpha037_2

    # Alpha#38	 ((-1 * rank(Ts_Rank(close, 10))) * rank((close / open)))
    def alpha038(self):
        inner = self.close / self.open
        alpha038_1 = inner.replace([-np.inf, np.inf], 1).fillna(value=1)
        return alpha038_1

    # Alpha#39	 ((-1 * rank((delta(close, 7) * (1 - rank(decay_linear_pn((volume / adv20), 9)))))) * (1 +rank(sum(returns, 250))))
    def alpha039(self):
        adv20 = sma(self.volume, 20)
        alpha039_1 = decay_linear((self.volume / adv20).to_frame(), 9).CLOSE.replace([-np.inf, np.inf], 0).fillna(
            value=0)
        alpha039_2 = sma(self.returns, 250)
        return alpha039_1, alpha039_2

    # Alpha#40	 ((-1 * rank(stddev(high, 10))) * correlation(high, volume, 10))
    def alpha040(self):
        alpha040_1 = stddev(self.high, 10).replace([-np.inf, np.inf], 0).fillna(value=0)
        alpha040_2 = correlation(self.high, self.volume, 10).replace([-np.inf, np.inf], 0).fillna(value=0)
        return alpha040_1, alpha040_2

    # Alpha#41	 (((high * low)^0.5) - vwap)
    def alpha041(self):
        return pow((self.high * self.low), 0.5) - self.vwap

    # Alpha#42	 (rank((vwap - close)) / rank((vwap + close)))
    def alpha042(self):
        alpha042_1 = self.vwap - self.close
        alpha042_2 = self.vwap + self.close
        return alpha042_1, alpha042_2

    # Alpha#43	 (ts_rank((volume / adv20), 20) * ts_rank((-1 * delta(close, 7)), 8))
    def alpha043(self):
        adv20 = sma(self.volume, 20)
        alpha043_1 = self.volume / adv20
        alpha043_1 = alpha043_1.replace([-np.inf, np.inf], 0).fillna(value=0)
        alpha043_2 = (-1 * delta(self.close, 7))
        return alpha043_1, alpha043_2

    # Alpha#44	 (-1 * correlation(high, rank(volume), 5))
    def alpha044(self):
        return None

    # Alpha#45	 (-1 * ((rank((sum(delay(close, 5), 20) / 20)) * correlation(close, volume, 2)) *rank(correlation(sum(close, 5), sum(close, 20), 2))))
    def alpha045(self):
        df = correlation(self.close, self.volume, 2).replace([-np.inf, np.inf], 0).fillna(value=0)
        alpha045_1 = sma(delay(self.close, 5), 20)
        alpha045_2 = df.replace([-np.inf, np.inf], 0).fillna(value=0)
        alpha045_3 = correlation(ts_sum(self.close, 5), ts_sum(self.close, 20), 2).replace([-np.inf, np.inf],
                                                                                           0).fillna(value=0)
        return alpha045_1, alpha045_2, alpha045_3

    # Alpha#46	 ((0.25 < (((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10))) ?(-1 * 1) : (((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < 0) ? 1 :((-1 * 1) * (close - delay(close, 1)))))
    def alpha046(self):
        inner = ((delay(self.close, 20) - delay(self.close, 10)) / 10) - ((delay(self.close, 10) - self.close) / 10)
        alpha = (-1 * delta(self.close))
        alpha[inner < 0] = 1
        alpha[inner > 0.25] = -1
        return alpha

    # Alpha#47	 ((((rank((1 / close)) * volume) / adv20) * ((high * rank((high - close))) / (sum(high, 5) /5))) - rank((vwap - delay(vwap, 5))))
    def alpha047(self):
        alpha047_5 = sma(self.volume, 20).replace([-np.inf, np.inf], 0).fillna(value=0)
        alpha047_1 = (1 / self.close).replace([-np.inf, np.inf], 0).fillna(value=0)
        alpha047_2 = (self.high - self.close).replace([-np.inf, np.inf], 0).fillna(value=0)
        alpha047_3 = (sma(self.high, 5) / 5).replace([-np.inf, np.inf], 0).fillna(value=0)
        alpha047_4 = (self.vwap - delay(self.vwap, 5)).replace([-np.inf, np.inf], 0).fillna(value=0)
        return alpha047_1, alpha047_2, alpha047_3, alpha047_4, alpha047_5

    # Alpha#48	 (indneutralize(((correlation(delta(close, 1), delta(delay(close, 1), 1), 250) *delta(close, 1)) / close), IndClass.subindustry) / sum(((delta(close, 1) / delay(close, 1))^2), 250))
    # 本Alpha使用了美股特有行业数据，indneutralize函数无法构建，无法实现此Alpha

    # Alpha#49	 (((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < (-1 *0.1)) ? 1 : ((-1 * 1) * (close - delay(close, 1))))
    def alpha049(self):
        inner = (((delay(self.close, 20) - delay(self.close, 10)) / 10) - (
                (delay(self.close, 10) - self.close) / 10))
        alpha = (-1 * delta(self.close))
        alpha[inner < -0.1] = 1
        return alpha

    # Alpha#50	 (-1 * ts_max(rank(correlation(rank(volume), rank(vwap), 5)), 5))
    def alpha050(self):
        return None

    # Alpha#51	 (((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < (-1 *0.05)) ? 1 : ((-1 * 1) * (close - delay(close, 1))))
    def alpha051(self):
        inner = (((delay(self.close, 20) - delay(self.close, 10)) / 10) - (
                (delay(self.close, 10) - self.close) / 10))
        alpha = (-1 * delta(self.close))
        alpha[inner < -0.05] = 1
        return alpha

    # Alpha#52	 ((((-1 * ts_min(low, 5)) + delay(ts_min(low, 5), 5)) * rank(((sum(returns, 240) -sum(returns, 20)) / 220))) * ts_rank(volume, 5))
    def alpha052(self):
        alpha052_1 = (-1 * delta(ts_min(self.low, 5), 5))
        alpha052_2 = ((ts_sum(self.returns, 240) - ts_sum(self.returns, 20)) / 220)
        return alpha052_1, alpha052_2

    # Alpha#53	 (-1 * delta((((close - low) - (high - close)) / (close - low)), 9))
    def alpha053(self):
        inner = (self.close - self.low).replace(0, 0.0001)
        return -1 * delta(
            (((self.close - self.low) - (self.high - self.close)) / inner).replace([-np.inf, np.inf], 0).fillna(
                value=0), 9)

    # Alpha#54	 ((-1 * ((low - close) * (open^5))) / ((low - high) * (close^5)))
    def alpha054(self):
        inner = (self.low - self.high).replace(0, -0.0001)
        return -1 * (self.low - self.close) * (self.open ** 5) / (inner * (self.close ** 5))

    # Alpha#55	 (-1 * correlation(rank(((close - ts_min(low, 12)) / (ts_max(high, 12) - ts_min(low,12)))), rank(volume), 6))
    def alpha055(self):
        divisor = (ts_max(self.high, 12) - ts_min(self.low, 12)).replace(0, 0.0001)
        alpha055_1 = ((self.close - ts_min(self.low, 12)) / (divisor)).replace([-np.inf, np.inf], 0).fillna(value=0)
        return alpha055_1

    # Alpha#56	 (0 - (1 * (rank((sum(returns, 10) / sum(sum(returns, 2), 3))) * rank((returns * cap)))))
    # 本Alpha使用了cap|市值，暂未取到该值
    #    def alpha056(self):
    #        return (0 - (1 * (rank((sma(self.returns, 10) / sma(sma(self.returns, 2), 3))) * rank((self.returns * self.cap)))))

    # Alpha#57	 (0 - (1 * ((close - vwap) / decay_linear_pn(rank(ts_argmax(close, 30)), 2))))
    def alpha057(self):
        alpha057_1 = (self.close - self.vwap)
        alpha057_2 = ts_argmax(self.close, 30)
        return alpha057_1, alpha057_2

    # Alpha#58	 (-1 * Ts_Rank(decay_linear_pn(correlation(IndNeutralize(vwap, IndClass.sector), volume,3.92795), 7.89291), 5.50322))
    # 本Alpha使用了美股特有行业数据，indneutralize函数无法构建，无法实现此Alpha
    # Alpha#59	 (-1 * Ts_Rank(decay_linear_pn(correlation(IndNeutralize(((vwap * 0.728317) + (vwap *(1 - 0.728317))), IndClass.industry), volume, 4.25197), 16.2289), 8.19648))
    # 本Alpha使用了美股特有行业数据，indneutralize函数无法构建，无法实现此Alpha

    # Alpha#60	 (0 - (1 * ((2 * scale(rank(((((close - low) - (high - close)) / (high - low)) * volume)))) -scale(rank(ts_argmax(close, 10))))))
    def alpha060(self):
        divisor = (self.high - self.low).replace(0, 0.0001)
        alpha060_1 = (((self.close - self.low) - (self.high - self.close)) * self.volume / divisor).replace(
            [-np.inf, np.inf], 0).fillna(value=0)
        alpha060_2 = ts_argmax(self.close, 10)
        return alpha060_1, alpha060_2

    # Alpha#61	 (rank((vwap - ts_min(vwap, 16.1219))) < rank(correlation(vwap, adv180, 17.9282)))
    def alpha061(self):
        adv180 = sma(self.volume, 180)
        alpha061_1 = (self.vwap - ts_min(self.vwap, 16))
        alpha061_2 = correlation(self.vwap, adv180, 18).replace([-np.inf, np.inf], 0).fillna(value=0)
        return alpha061_1, alpha061_2

    # Alpha#62	 ((rank(correlation(vwap, sum(adv20, 22.4101), 9.91009)) < rank(((rank(open) +rank(open)) < (rank(((high + low) / 2)) + rank(high))))) * -1)
    def alpha062(self):
        adv20 = sma(self.volume, 20)
        alpha062_1 = correlation(self.vwap, sma(adv20, 22), 10).replace([-np.inf, np.inf], 0).fillna(value=0)
        alpha062_2 = ((self.high + self.low) / 2)
        return alpha062_1, alpha062_2

    # Alpha#63	 ((rank(decay_linear_pn(delta(IndNeutralize(close, IndClass.industry), 2.25164), 8.22237))- rank(decay_linear_pn(correlation(((vwap * 0.318108) + (open * (1 - 0.318108))), sum(adv180,37.2467), 13.557), 12.2883))) * -1)
    # 本Alpha使用了美股特有行业数据，indneutralize函数无法构建，无法实现此Alpha

    # Alpha#64	 ((rank(correlation(sum(((open * 0.178404) + (low * (1 - 0.178404))), 12.7054),sum(adv120, 12.7054), 16.6208)) < rank(delta(((((high + low) / 2) * 0.178404) + (vwap * (1 -0.178404))), 3.69741))) * -1)
    def alpha064(self):
        adv120 = sma(self.volume, 120)
        alpha064_1 = correlation(sma(((self.open * 0.178404) + (self.low * (1 - 0.178404))), 13), sma(adv120, 13),
                                 17).replace([-np.inf, np.inf], 0).fillna(value=0)
        alpha064_2 = delta(((((self.high + self.low) / 2) * 0.178404) + (self.vwap * (1 - 0.178404))), 3.69741)
        return alpha064_1, alpha064_2


class DataPrepare(object):
    def __init__(self):
        pass

    def get_basic_data(self, start_date, end_date, stock_list):
        dependencies = ['close', 'open', 'high', 'low', 'pre_close', 'volume', 'money', 'trade_date', 'symbol']
        sql = "select * from vision.sk_daily_price where trade_date >='{}' and trade_date <='{}' and symbol in {}".format(
            start_date, end_date, stock_list)
        sql_engine = create_engine("mysql+mysqlconnector://root:123@10.15.97.128:3306/vision")

        sql_data = pd.read_sql(sql, sql_engine)
        # 先转化成multiindex
        res = sql_data[dependencies].set_index(['trade_date', 'symbol']).to_panel()
        res['returns'] = res['close'] / res['pre_close'] - 1
        res['vwap'] = (res['money'] * 1000) / (res['volume'] * 100 + 1)

        return res

    def get_data(self, stock_list):
        dependencies = ['close', 'open', 'high', 'low', 'pre_close', 'volume', 'money', 'trade_date', 'symbol']
        sql = "select * from vision.sk_daily_price where trade_date>'2010-01-01' and symbol in ('000001.XSHE', '000002.XSHE', '000004.XSHE', '000006.XSHE', '000007.XSHE')"
        # sql = "select * from vision.sk_daily_price where symbol in {}".format(stock_list)
        sql_engine = create_engine("mysql+mysqlconnector://root:1234@10.15.97.128:3306/vision")
        sql_data = pd.read_sql(sql, sql_engine)
        # 先转化成multiindex
        sql_data['returns'] = sql_data['close'] / sql_data['pre_close'] - 1
        sql_data['vwap'] = (sql_data['money'] * 1000) / (sql_data['volume'] * 100 + 1)
        return sql_data


if __name__ == '__main__':
    alpha_data = DataPrepare()
    security = ('000001.XSHE', '000002.XSHE', '000004.XSHE', '000006.XSHE', '000007.XSHE')
    # 获取数据
    # stock_panel = alpha_data.get_basic_data("2010-01-01", "2018-12-30", stock_list=security)
    stock_panel = alpha_data.get_data(stock_list=security)
    #
    stock_group = stock_panel.groupby('symbol')
    tmp = pd.DataFrame()
    for i, stock in stock_group:
        dependencies = ['close', 'open', 'high', 'low', 'pre_close', 'volume', 'money', 'trade_date', 'symbol',
                        'returns', 'vwap']
        stock = stock[dependencies]
        alpha = Alpha101Z26(stock)
        stock['alpha028'] = alpha.alpha028()
        stock['alpha029_1'], stock['alpha029_2'] = alpha.alpha029()
        stock['alpha030_1'], stock['alpha030_2'] = alpha.alpha030()
        stock['alpha031_1'], stock['alpha031_2'], stock['alpha031_3'] = alpha.alpha031()
        stock['alpha032'] = alpha.alpha032()
        stock['alpha033_1'] = alpha.alpha033()
        stock['alpha034_1'], stock['alpha034_2'] = alpha.alpha034()
        stock['alpha035_1'] = alpha.alpha035()
        stock['alpha036_1'], stock['alpha036_2'], stock['alpha036_3'], stock['alpha036_4'], stock[
            'alpha036_5'] = alpha.alpha036()
        stock['alpha037_1'], stock['alpha037_2'] = alpha.alpha037()
        stock['alpha038_1'] = alpha.alpha038()
        stock['alpha039_1'], stock['alpha039_2'] = alpha.alpha039()
        stock['alpha040_1'], stock['alpha040_2'] = alpha.alpha040()
        stock['alpha041'] = alpha.alpha041()
        stock['alpha042_1'], stock['alpha042_2'] = alpha.alpha042()
        stock['alpha043_1'], stock['alpha043_2'] = alpha.alpha043()
        stock['alpha045_1'], stock['alpha045_2'], stock['alpha045_3'] = alpha.alpha045()
        stock['alpha046'] = alpha.alpha046()
        stock['alpha047_1'], stock['alpha047_2'], stock['alpha047_3'], stock['alpha047_4'], stock[
            'alpha047_5'] = alpha.alpha047()
        stock['alpha049'] = alpha.alpha049()
        stock['alpha051'] = alpha.alpha051()
        stock['alpha052_1'], stock['alpha052_2'] = alpha.alpha052()
        stock['alpha053'] = alpha.alpha053()
        stock['alpha054'] = alpha.alpha054()
        stock['alpha055_1'] = alpha.alpha055()
        stock['alpha057_1'], stock['alpha057_2'] = alpha.alpha057()
        stock['alpha060_1'], stock['alpha060_2'] = alpha.alpha060()
        stock['alpha061_1'], stock['alpha061_2'] = alpha.alpha061()
        stock['alpha062_1'], stock['alpha062_2'] = alpha.alpha062()
        stock['alpha064_1'], stock['alpha064_2'] = alpha.alpha064()

        tmp = tmp.append(stock)
    # 计算结果保存到数据库
    # tmp.to_csv('test.csv')
    test_sql_engine = create_engine("mysql+mysqlconnector://root:1234@10.15.97.128:3306/test")
    tmp.to_sql(name='alpha101_tmp_test_z26', con=test_sql_engine, chunksize=1000, if_exists='replace', index=True)

    # with test_sql_engine.connect() as con:
    #     con.execute('ALTER TABLE `alpha101_tmp_test_z26` ADD PRIMARY KEY (`index`);')
