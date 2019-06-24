#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
@version: ??
@author: li
@file: factor_correlation_test.py
@time: 2019-02-11 11:53
"""

import numpy as np
import pandas as pd
from uqer import uqer, DataAPI

client = uqer.Client(token='07b082b1f42b91987660f0c2c19097bc3b10fa4b12f6af3274f82df930185f04')
result = DataAPI.MktStockFactorsDateRangeGet(secID="",ticker="000001",beginDate="20170612",endDate="20170616",field="",pandas="1")

print(result)
# step1 从优矿中读取相关数据
uqer_historical_factor = [np.random.randint(0, 100) for a in range(20)]


# step2 读取数据库中计算的数据
local_historical_factor = [np.random.randint(0, 100) for b in range(20)]


# step3 计算两组数据的相关系数
ab = np.array([uqer_historical_factor, local_historical_factor])

dfab = pd.DataFrame(ab.T, columns=['A', 'B'])
# 协方差
covab = dfab.A.cov(dfab.B)
# 相关系数
corrab = dfab.A.corr(dfab.B)

print("协方差: {}, 相关系数: {}".format(covab, corrab))
