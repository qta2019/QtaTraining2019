# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 21:02:38 2019

@author: PeterHu
"""

# 可以自己import我们平台支持的第三方python模块，比如pandas、numpy等
import numpy as np
import pandas as pd
import talib
import time
import datetime

# 在这个方法中编写任何的初始化逻辑。context对象将会在你的算法策略的任何方法之间做传递
def init(context):
    # context内引入全局变量s1，存储目标合约信息
    context.s1 = 'I99'

    context.n = 1
    context.k1 = 0.3
    context.k2 = 0.3
    context.quantity = 1
    context.stoploss = 0
    context.stoplossday = []

    #初始化时订阅合约行情。订阅之后的合约行情会在handle_bar中进行更新
    subscribe(context.s1)

# 你选择的期货数据更新将会触发此段逻辑，例如日线或分钟线更新
def handle_bar(context, bar_dict):

    if (context.stoploss == 0) :
        if (context.now.strftime('%H:%M') < '14:59'):
            closep = history_bars(context.s1, context.n, '1d', 'close')
            openp = history_bars(context.s1, context.n, '1d', 'open')
            highp = history_bars(context.s1, context.n, '1d', 'high')
            lowp = history_bars(context.s1, context.n, '1d', 'low')

            HH = np.max(highp)
            LL = np.min(lowp)
            HC = np.max(closep)
            LC = np.min(closep)

            ran = max(HH-LC , HC-LL)

            opentoday = history_bars(context.s1, 1, '1d', 'open', include_now=True)[0]

            buyline = opentoday  + context.k1*ran
            sellline = opentoday - context.k2*ran

            if history_bars(context.s1, 1, '5m', 'close' ,include_now=True)[0] > buyline:
                sell_qty = context.portfolio.positions[context.s1].sell_quantity
                # 先判断当前卖方仓位，如果有，则进行平仓操作
                if sell_qty > 0:
                    buy_close(context.s1, sell_qty)
                # 买入开仓
                buy_open(context.s1, context.quantity)

            if history_bars(context.s1, 1, '5m', 'close' ,include_now=True)[0] < sellline:
                buy_qty = context.portfolio.positions[context.s1].buy_quantity
                # 先判断当前卖方仓位，如果有，则进行平仓操作
                if buy_qty > 0:
                    sell_close(context.s1, buy_qty)
                # 买入开仓
                sell_open(context.s1, context.quantity)


            if (history_bars(context.s1, 1, '5m', 'close' ,include_now=True)[0] > 1.2*context.portfolio.positions[context.s1].buy_avg_open_price) |  (history_bars(context.s1, 1, '5m', 'close' ,include_now=True)[0] < 0.98*context.portfolio.positions[context.s1].buy_avg_open_price):
                buy_qty = context.portfolio.positions[context.s1].buy_quantity
                # 先判断当前卖方仓位，如果有，则进行平仓操作
                if buy_qty > 0:
                    context.stoploss = 1
                    context.stoplossday.append(context.now)
                    sell_close(context.s1, buy_qty)

    
            if (history_bars(context.s1, 1, '5m', 'close' ,include_now=True)[0] > 1.02*context.portfolio.positions[context.s1].sell_avg_open_price) |  (history_bars(context.s1, 1, '5m', 'close' ,include_now=True)[0] < 0.8*context.portfolio.positions[context.s1].sell_avg_open_price):
                sell_qty = context.portfolio.positions[context.s1].sell_quantity
                # 先判断当前卖方仓位，如果有，则进行平仓操作
                if sell_qty > 0:
                    context.stoploss = 1
                    context.stoplossday.append(context.now)
                    buy_close(context.s1, sell_qty)
        else:
            buy_qty = context.portfolio.positions[context.s1].buy_quantity
            sell_qty = context.portfolio.positions[context.s1].sell_quantity
            if buy_qty > 0:
                sell_close(context.s1, buy_qty)
            if sell_qty > 0:
                buy_close(context.s1, sell_qty)
    
    if context.stoploss == 1:
        print(context.stoploss)
        if str(context.now)[0:10] == str(get_next_trading_date(get_next_trading_date(get_next_trading_date(get_next_trading_date(get_next_trading_date(context.stoplossday[-1]))))))[0:10]:
            context.stoploss = 0
            