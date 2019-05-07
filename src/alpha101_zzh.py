#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
@version: ??
@author: zzh
@file: alpha.py
@time: 2019-05-06 11:06
"""
import time
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

        # self.alpha026_ = df_data['alpha026']
        # self.alpha027_ = df_data['alpha027']
        self.alpha028_ = df_data['alpha028']
        self.alpha029_1 = df_data['alpha029_1']
        self.alpha029_2 = df_data['alpha029_2']
        self.alpha030_1 = df_data['alpha030_1']
        self.alpha030_2 = df_data['alpha030_2']
        self.alpha031_1 = df_data['alpha031_1']
        self.alpha031_2 = df_data['alpha031_2']
        self.alpha031_3 = df_data['alpha031_3']
        self.alpha032_ = df_data['alpha032']
        self.alpha033_1 = df_data['alpha033_1']
        self.alpha034_1 = df_data['alpha034_1']
        self.alpha034_2 = df_data['alpha034_2']
        self.alpha035_1 = df_data['alpha035_1']
        self.alpha036_1 = df_data['alpha036_1']
        self.alpha036_2 = df_data['alpha036_2']
        self.alpha036_3 = df_data['alpha036_3']
        self.alpha036_4 = df_data['alpha036_4']
        self.alpha036_5 = df_data['alpha036_5']
        self.alpha037_1 = df_data['alpha037_1']
        self.alpha037_2 = df_data['alpha037_2']
        self.alpha038_1 = df_data['alpha038_1']
        self.alpha039_1 = df_data['alpha039_1']
        self.alpha039_2 = df_data['alpha039_2']
        self.alpha040_1 = df_data['alpha040_1']
        self.alpha040_2 = df_data['alpha040_2']
        self.alpha041_ = df_data['alpha041']
        self.alpha042_1 = df_data['alpha042_1']
        self.alpha042_2 = df_data['alpha042_2']
        self.alpha043_1 = df_data['alpha043_1']
        self.alpha043_2 = df_data['alpha043_2']
        # self.alpha044_ = df_data['alpha044']
        self.alpha045_1 = df_data['alpha045_1']
        self.alpha045_2 = df_data['alpha045_2']
        self.alpha045_3 = df_data['alpha045_3']
        self.alpha046_ = df_data['alpha046']
        self.alpha047_1 = df_data['alpha047_1']
        self.alpha047_2 = df_data['alpha047_2']
        self.alpha047_3 = df_data['alpha047_3']
        self.alpha047_4 = df_data['alpha047_4']
        self.alpha047_5 = df_data['alpha047_5']
        # self.alpha048_ = df_data['alpha048']
        self.alpha049_ = df_data['alpha049']
        # self.alpha050_ = df_data['alpha050']
        self.alpha051_ = df_data['alpha051']
        self.alpha052_1 = df_data['alpha052_1']
        self.alpha052_2 = df_data['alpha052_2']
        self.alpha053_ = df_data['alpha053']
        self.alpha054_ = df_data['alpha054']
        self.alpha055_1 = df_data['alpha055_1']
        # self.alpha056_ = df_data['alpha056']
        self.alpha057_1 = df_data['alpha057_1']
        self.alpha057_2 = df_data['alpha057_2']
        # self.alpha058_ = df_data['alpha058']
        # self.alpha059_ = df_data['alpha059']
        self.alpha060_1 = df_data['alpha060_1']
        self.alpha060_2 = df_data['alpha060_2']
        self.alpha061_1 = df_data['alpha061_1']
        self.alpha061_2 = df_data['alpha061_2']
        self.alpha062_1 = df_data['alpha062_1']
        self.alpha062_2 = df_data['alpha062_2']
        # self.alpha063_ = df_data['alpha063']
        self.alpha064_1 = df_data['alpha064_1']
        self.alpha064_2 = df_data['alpha064_2']

    # Alpha#26	 (-1 * ts_max(correlation(ts_rank(volume, 5), ts_rank(high, 5), 5), 3))
    def alpha026(self):
        df = correlation(ts_rank(self.volume, 5), ts_rank(self.high, 5), 5)
        df = df.replace([-np.inf, np.inf], 0).fillna(value=0)
        return -1 * ts_max(df, 3)

    # Alpha#27	 ((0.5 < rank((sum(correlation(rank(volume), rank(vwap), 6), 2) / 2.0))) ? (-1 * 1) : 1)
    # 可能存在问题，我自己的数据测试了很多次值全为1，可能需要调整6,2这些参数？
    def alpha027(self):
        alpha = rank((sma(correlation(rank(self.volume), rank(self.vwap), 6), 2) / 2.0))
        alpha[alpha > 0.5] = -1
        alpha[alpha <= 0.5] = 1
        return alpha

        # Alpha#28	 scale(((correlation(adv20, low, 5) + ((high + low) / 2)) - close))

    def alpha028(self):
        return self.alpha028_

    # Alpha#29	 (min(product(rank(rank(scale(log(sum(ts_min(rank(rank((-1 * rank(delta((close - 1),5))))), 2), 1))))), 1), 5) + ts_rank(delay((-1 * returns), 6), 5))
    def alpha029(self):
        return (ts_min(rank(rank(scale(log(ts_sum(rank(rank(-1 * rank(self.alpha029_2))), 2))))), 5) +
                ts_rank(self.alpha029_1, 5))

    # Alpha#30	 (((1.0 - rank(((sign((close - delay(close, 1))) + sign((delay(close, 1) - delay(close, 2)))) +sign((delay(close, 2) - delay(close, 3)))))) * sum(volume, 5)) / sum(volume, 20))
    def alpha030(self):
        return (1.0 - rank(self.alpha030_1)) * self.alpha030_2

    # Alpha#31	 ((rank(rank(rank(decay_linear_pn_pn((-1 * rank(rank(delta(close, 10)))), 10)))) + rank((-1 *delta(close, 3)))) + sign(scale(correlation(adv20, low, 12))))
    def alpha031(self):
        p1 = rank(rank(rank(decay_linear_pn((-1 * rank(rank(self.alpha031_1))), 10))))
        p2 = rank(self.alpha031_2)

        return p1 + p2 + self.alpha031_3

    # Alpha#32	 (scale(((sum(close, 7) / 7) - close)) + (20 * scale(correlation(vwap, delay(close, 5),230))))
    def alpha032(self):
        return self.alpha032_

    # Alpha#33	 rank((-1 * ((1 - (open / close))^1)))
    def alpha033(self):
        return rank(self.alpha033_1)

    # Alpha#34	 rank(((1 - rank((stddev(returns, 2) / stddev(returns, 5)))) + (1 - rank(delta(close, 1)))))
    def alpha034(self):
        return rank(2 - rank(self.alpha034_1) - rank(self.alpha034_2))

    # Alpha#35	 ((Ts_Rank(volume, 32) * (1 - Ts_Rank(((close + high) - low), 16))) * (1 -Ts_Rank(returns, 32)))
    def alpha035(self):
        return ((ts_rank(self.volume, 32) *
                 (1 - ts_rank(self.alpha035_1, 16))) *
                (1 - ts_rank(self.returns, 32)))

    # Alpha#36	 (((((2.21 * rank(correlation((close - open), delay(volume, 1), 15))) + (0.7 * rank((open- close)))) + (0.73 * rank(Ts_Rank(delay((-1 * returns), 6), 5)))) + rank(abs(correlation(vwap,adv20, 6)))) + (0.6 * rank((((sum(close, 200) / 200) - open) * (close - open)))))
    def alpha036(self):
        adv20 = sma(self.volume, 20)
        return (((((2.21 * rank(self.alpha036_1)) + (
                0.7 * rank(self.alpha036_2))) + (
                          0.73 * rank(ts_rank(self.alpha036_3, 5)))) + rank(
            self.alpha036_4)) + (
                        0.6 * rank(self.alpha036_5)))

    # Alpha#37	 (rank(correlation(delay((open - close), 1), close, 200)) + rank((open - close)))
    def alpha037(self):
        return rank(self.alpha037_1) + rank(self.alpha037_2)

    # Alpha#38	 ((-1 * rank(Ts_Rank(close, 10))) * rank((close / open)))
    def alpha038(self):
        return -1 * rank(ts_rank(self.open, 10)) * rank(self.alpha038_1)

    # Alpha#39	 ((-1 * rank((delta(close, 7) * (1 - rank(decay_linear_pn_pn((volume / adv20), 9)))))) * (1 +rank(sum(returns, 250))))
    def alpha039(self):
        return ((-1 * rank(
            delta(self.close, 7) * (1 - rank(self.alpha039_1)))) *
                (1 + rank(self.alpha039_2)))

    # Alpha#40	 ((-1 * rank(stddev(high, 10))) * correlation(high, volume, 10))
    def alpha040(self):
        return -1 * rank(self.alpha040_1) * self.alpha040_2

    # Alpha#41	 (((high * low)^0.5) - vwap)
    def alpha041(self):
        return self.alpha041_

    # Alpha#42	 (rank((vwap - close)) / rank((vwap + close)))
    def alpha042(self):
        return rank(self.alpha042_1) / rank(self.alpha042_2)

    # Alpha#43	 (ts_rank((volume / adv20), 20) * ts_rank((-1 * delta(close, 7)), 8))
    def alpha043(self):
        return ts_rank(self.alpha043_1, 20) * ts_rank(self.alpha043_2, 8)

    # Alpha#44	 (-1 * correlation(high, rank(volume), 5))
    def alpha044(self):
        df = correlation(self.high, rank(self.volume), 5)
        df = df.replace([-np.inf, np.inf], 0).fillna(value=0)
        return -1 * df

    # Alpha#45	 (-1 * ((rank((sum(delay(close, 5), 20) / 20)) * correlation(close, volume, 2)) *rank(correlation(sum(close, 5), sum(close, 20), 2))))
    def alpha045(self):
        return -1 * (rank(self.alpha045_1) * self.alpha045_2 *
                     rank(self.alpha045_3))

    # Alpha#46	 ((0.25 < (((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10))) ?(-1 * 1) : (((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < 0) ? 1 :((-1 * 1) * (close - delay(close, 1)))))
    def alpha046(self):
        return self.alpha046_

    # Alpha#47	 ((((rank((1 / close)) * volume) / adv20) * ((high * rank((high - close))) / (sum(high, 5) /5))) - rank((vwap - delay(vwap, 5))))
    def alpha047(self):
        return ((((rank(self.alpha047_1) * self.volume) / self.alpha047_5) * (
                (self.high * rank(self.alpha047_2)) / self.alpha047_3)) - rank(self.alpha047_4))

    # Alpha#48	 (indneutralize(((correlation(delta(close, 1), delta(delay(close, 1), 1), 250) *delta(close, 1)) / close), IndClass.subindustry) / sum(((delta(close, 1) / delay(close, 1))^2), 250))
    # 本Alpha使用了美股特有行业数据，indneutralize函数无法构建，无法实现此Alpha

    # Alpha#49	 (((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < (-1 *0.1)) ? 1 : ((-1 * 1) * (close - delay(close, 1))))
    def alpha049(self):
        return self.alpha049_

    # Alpha#50	 (-1 * ts_max(rank(correlation(rank(volume), rank(vwap), 5)), 5))
    def alpha050(self):
        return (-1 * ts_max(rank(correlation(rank(self.volume), rank(self.vwap), 5)), 5))

    # Alpha#51	 (((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < (-1 *0.05)) ? 1 : ((-1 * 1) * (close - delay(close, 1))))
    def alpha051(self):
        return self.alpha051_

    # Alpha#52	 ((((-1 * ts_min(low, 5)) + delay(ts_min(low, 5), 5)) * rank(((sum(returns, 240) -sum(returns, 20)) / 220))) * ts_rank(volume, 5))
    def alpha052(self):
        return ((self.alpha052_1 *
                 rank(self.alpha052_2)) * ts_rank(self.volume, 5))

    # Alpha#53	 (-1 * delta((((close - low) - (high - close)) / (close - low)), 9))
    def alpha053(self):
        return self.alpha053_

    # Alpha#54	 ((-1 * ((low - close) * (open^5))) / ((low - high) * (close^5)))
    def alpha054(self):
        return self.alpha054_

    # Alpha#55	 (-1 * correlation(rank(((close - ts_min(low, 12)) / (ts_max(high, 12) - ts_min(low,12)))), rank(volume), 6))
    def alpha055(self):
        df = correlation(rank(self.alpha055_1), rank(self.volume), 6)
        return -1 * df.replace([-np.inf, np.inf], 0).fillna(value=0)

    # Alpha#56	 (0 - (1 * (rank((sum(returns, 10) / sum(sum(returns, 2), 3))) * rank((returns * cap)))))
    # 本Alpha使用了cap|市值，暂未取到该值
    #    def alpha056(self):
    #        return (0 - (1 * (rank((sma(self.returns, 10) / sma(sma(self.returns, 2), 3))) * rank((self.returns * self.cap)))))

    # Alpha#57	 (0 - (1 * ((close - vwap) / decay_linear_pn_pn(rank(ts_argmax(close, 30)), 2))))
    def alpha057(self):
        alpha057_2 = decay_linear_pn(rank(self.alpha057_2), 2)
        return (0 - (
                1 * (self.alpha057_1 / alpha057_2)))

    # Alpha#58	 (-1 * Ts_Rank(decay_linear_pn_pn(correlation(IndNeutralize(vwap, IndClass.sector), volume,3.92795), 7.89291), 5.50322))
    # 本Alpha使用了美股特有行业数据，indneutralize函数无法构建，无法实现此Alpha
    # Alpha#59	 (-1 * Ts_Rank(decay_linear_pn_pn(correlation(IndNeutralize(((vwap * 0.728317) + (vwap *(1 - 0.728317))), IndClass.industry), volume, 4.25197), 16.2289), 8.19648))
    # 本Alpha使用了美股特有行业数据，indneutralize函数无法构建，无法实现此Alpha

    # Alpha#60	 (0 - (1 * ((2 * scale(rank(((((close - low) - (high - close)) / (high - low)) * volume)))) -scale(rank(ts_argmax(close, 10))))))
    def alpha060(self):
        return - ((2 * scale(rank(self.alpha060_1))) - scale(rank(self.alpha060_2)))

    # Alpha#61	 (rank((vwap - ts_min(vwap, 16.1219))) < rank(correlation(vwap, adv180, 17.9282)))
    def alpha061(self):
        return (rank(self.alpha061_1) < rank(self.alpha061_2))

    # Alpha#62	 ((rank(correlation(vwap, sum(adv20, 22.4101), 9.91009)) < rank(((rank(open) +rank(open)) < (rank(((high + low) / 2)) + rank(high))))) * -1)
    def alpha062(self):
        return ((rank(self.alpha062_1) < rank(
            ((rank(self.open) + rank(self.open)) < (rank(self.alpha062_2) + rank(self.high))))) * -1)

    # Alpha#63	 ((rank(decay_linear_pn_pn(delta(IndNeutralize(close, IndClass.industry), 2.25164), 8.22237))- rank(decay_linear_pn_pn(correlation(((vwap * 0.318108) + (open * (1 - 0.318108))), sum(adv180,37.2467), 13.557), 12.2883))) * -1)
    # 本Alpha使用了美股特有行业数据，indneutralize函数无法构建，无法实现此Alpha

    # Alpha#64	 ((rank(correlation(sum(((open * 0.178404) + (low * (1 - 0.178404))), 12.7054),sum(adv120, 12.7054), 16.6208)) < rank(delta(((((high + low) / 2) * 0.178404) + (vwap * (1 -0.178404))), 3.69741))) * -1)
    def alpha064(self):
        return ((rank(self.alpha064_1) < rank(self.alpha064_2)) * -1)

    def calculate(self):
        alpha026 = self.alpha026()
        alpha027 = self.alpha027()
        alpha028 = self.alpha028()
        alpha029 = self.alpha029()
        alpha030 = self.alpha030()
        alpha031 = self.alpha031()
        alpha032 = self.alpha032()
        alpha033 = self.alpha033()
        alpha034 = self.alpha034()
        alpha035 = self.alpha035()
        alpha036 = self.alpha036()
        alpha037 = self.alpha037()
        alpha038 = self.alpha038()
        alpha039 = self.alpha039()
        alpha040 = self.alpha040()
        alpha041 = self.alpha041()
        alpha042 = self.alpha042()
        alpha043 = self.alpha043()
        alpha044 = self.alpha044()
        alpha045 = self.alpha045()
        alpha046 = self.alpha046()
        alpha047 = self.alpha047()
        # alpha048 = self.alpha048()
        alpha049 = self.alpha049()
        alpha050 = self.alpha050()
        alpha051 = self.alpha051()
        alpha052 = self.alpha052()
        alpha053 = self.alpha053()
        alpha054 = self.alpha054()
        alpha055 = self.alpha055()
        # alpha056 = self.alpha056()
        alpha057 = self.alpha057()
        # alpha058 = self.alpha058()
        # alpha059 = self.alpha059()
        alpha060 = self.alpha060()
        alpha061 = self.alpha061()
        alpha062 = self.alpha062()
        # alpha063 = self.alpha063()
        alpha064 = self.alpha064()

        tmp_pn = pd.Panel({
            'alpha026': alpha026,
            'alpha027': alpha027,
            'alpha028': alpha028,
            'alpha029': alpha029,
            'alpha030': alpha030,
            'alpha031': alpha031,
            'alpha032': alpha032,
            'alpha033': alpha033,
            'alpha034': alpha034,
            'alpha035': alpha035,
            'alpha036': alpha036,
            'alpha037': alpha037,
            'alpha038': alpha038,
            'alpha039': alpha039,
            'alpha040': alpha040,
            'alpha041': alpha041,
            'alpha042': alpha042,
            'alpha043': alpha043,
            'alpha044': alpha044,
            'alpha045': alpha045,
            'alpha046': alpha046,
            'alpha047': alpha047,
            # 'alpha048': alpha048,
            'alpha049': alpha049,
            'alpha050': alpha050,
            'alpha051': alpha051,
            'alpha052': alpha052,
            'alpha053': alpha053,
            'alpha054': alpha054,
            'alpha055': alpha055,
            # 'alpha056': alpha056,
            'alpha057': alpha057,
            # 'alpha058': alpha058,
            # 'alpha059': alpha059,
            'alpha060': alpha060,
            'alpha061': alpha061,
            'alpha062': alpha062,
            # 'alpha063': alpha063,
            'alpha064': alpha064
        })

        return tmp_pn


class DataPrepare(object):
    def __init__(self):
        pass

    def get_basic_data(self, start_date, end_date, stock_list):
        dependencies = ['close', 'open', 'high', 'low', 'pre_close', 'volume', 'money', 'trade_date', 'symbol']
        sql = "select * from test.alpha101_tmp_test_z26 where trade_date >='{}' and trade_date <='{}' and symbol in {}".format(
            start_date, end_date, stock_list)
        sql_engine = create_engine("mysql+mysqlconnector://root:1234@10.15.97.128:3306/test")

        sql_data = pd.read_sql(sql, sql_engine)
        # 先转化成multiindex
        res = sql_data.set_index(['trade_date', 'symbol']).to_panel()
        return res


if __name__ == '__main__':
    pd.set_option('display.max_rows', None, 'display.max_columns', None, "display.max_colwidth", 1000, 'display.width',
                  1000)

    alpha_data = DataPrepare()
    security = ('000001.XSHE', '000002.XSHE', '000004.XSHE', '000006.XSHE', '000007.XSHE')

    stock_panel = alpha_data.get_basic_data("2018-01-01", "2018-12-30", stock_list=security)

    start_time = time.time()
    alpha = Alpha101Z26(stock_panel)
    pn = alpha.calculate()
    print('spend time %s' % (time.time() - start_time))
    print(pn.minor_xs('000001.XSHE'))
