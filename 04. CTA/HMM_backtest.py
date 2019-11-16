# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 21:21:51 2019

@author: Liu Li
基于隐马尔科夫模型的螺纹钢商品期货择时策略
"""
import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM
from matplotlib import cm,pyplot as plt
import matplotlib.dates as dates
from scipy import stats
from sklearn import preprocessing
import datetime



'''
1.下载螺纹钢期货历史数据
'''

from WindPy import *
w.start()

#设置回测区间
start_date = '2010-01-01'
end_date = '2019-10-29'

#下载价量指标，包括开盘价，最高价，最低价，收盘价，成交量
info = w.wsd("RB.SHF", "open,high,low,close,volume", start_date, end_date, "TradingCalendar=SHFE;PriceAdj=B")
#存储数据
rb_info = pd.DataFrame(index = info.Times)
rb_info['open'] = info.Data[0]
rb_info['high'] = info.Data[1]
rb_info['low'] = info.Data[2]
rb_info['close'] = info.Data[3]
rb_info['volume'] = info.Data[4]

rb_info.to_pickle("rb_info")
rb_info.to_excel("rb_info.xls")



'''
2.清洗数据
'''
#读取数据
#rb_info = pd.read_pickle('rb_info')
date = rb_info.index

#统计数据特征
rb_info_statistic = pd.DataFrame(index = ['max','min','avg','std'])
rb_info_statistic['open'] = [np.max(rb_info['open']),np.min(rb_info['open']),np.mean(rb_info['open']),np.sqrt(np.cov(rb_info['open']))]
rb_info_statistic['high'] = [np.max(rb_info['high']),np.min(rb_info['high']),np.mean(rb_info['high']),np.sqrt(np.cov(rb_info['high']))]
rb_info_statistic['low'] = [np.max(rb_info['low']),np.min(rb_info['low']),np.mean(rb_info['low']),np.sqrt(np.cov(rb_info['low']))]
rb_info_statistic['close'] = [np.max(rb_info['close']),np.min(rb_info['close']),np.mean(rb_info['close']),np.sqrt(np.cov(rb_info['close']))]
rb_info_statistic['volume'] = [np.max(rb_info['volume']),np.min(rb_info['volume']),np.mean(rb_info['volume']),np.sqrt(np.cov(rb_info['volume']))]


#画图
plt.plot(date, rb_info['close'], 'b-', label = 'close')
plt.xlabel('date')
plt.ylabel('price')
plt.legend()
plt.title('br_basic_close_price_info')
plt.savefig('br_basic_price_info.jpg')
plt.show()

plt.plot(date, rb_info['volume'], 'r-', label = 'volume')
plt.xlabel('date')
plt.ylabel('volumn')
plt.legend()
plt.title('br_basic_volumn_info')
plt.savefig('br_basic_volumn_info.jpg')
plt.show()


'''
3.构建指标，包括一日对数成交量差，五日对数成交量差，当日对数高低价差，一日对数收盘价差，五日对数收盘价差
'''
#构建新的时间序列，从第六个交易日开始
new_date = pd.to_datetime(date[5:])

#构建指标
#计算一日对数成交量差
log_vol_one = np.array(np.diff(np.log(rb_info['volume'])))
log_vol_one = log_vol_one[4:]#统一时间区间
#计算五日对数成交量差
log_vol_five = np.log(np.array(rb_info['volume'][5:]))-np.log(np.array(rb_info['volume'][:-5]))
#计算当日对数高低价差
delta_log_high_low = np.log(np.array(rb_info['high']))-np.log(np.array(rb_info['low']))
delta_log_high_low = delta_log_high_low[5:]
#计算一日对数收盘价差
log_return_one = np.array(np.diff(np.log(rb_info['close'])))
log_return_one = log_return_one[4:]
#计算五日对数收盘价差
log_return_five = np.log(np.array(rb_info['close'][5:]))-np.log(np.array(rb_info['close'][:-5]))
#计算收盘价
close_price = np.array(rb_info['close'][5:])

#画出5个指标的分布直方图
plt.figure()
ax1 = plt.subplot(511)
ax2 = plt.subplot(512)
ax3 = plt.subplot(513)
ax4 = plt.subplot(514)
ax5 = plt.subplot(515)
plt.tight_layout(pad=0, w_pad=0, h_pad=0.1)
plt.sca(ax1)
plt.hist(log_return_one, 50, normed=1, facecolor='green', alpha=0.75)
plt.sca(ax2)
plt.hist(log_return_five, 50, normed=1, facecolor='green', alpha=0.75)
plt.sca(ax3)
plt.hist(log_vol_one, 50, normed=1, facecolor='green', alpha=0.75)
plt.sca(ax4)
plt.hist(log_vol_five, 50, normed=1, facecolor='green', alpha=0.75)
plt.sca(ax5)
plt.hist(delta_log_high_low, 50, normed=1, facecolor='green', alpha=0.75)
plt.savefig('index_statistic_hist.jpg')
plt.show()

#对指标数据进行标准正态分布和归一处理
box_cox_delta_log_high_low,lambda_ = stats.boxcox(delta_log_high_low)
rescale_delta_log_high_low = preprocessing.scale(box_cox_delta_log_high_low)
rescale_log_return_one = preprocessing.scale(log_return_one)
rescale_log_return_five = preprocessing.scale(log_return_five)
rescale_log_vol_one = preprocessing.scale(log_vol_one)
rescale_log_vol_five = preprocessing.scale(log_vol_five)


#重新画图
plt.figure()
ax1 = plt.subplot(511)
ax2 = plt.subplot(512)
ax3 = plt.subplot(513)
ax4 = plt.subplot(514)
ax5 = plt.subplot(515)
plt.tight_layout(pad=0, w_pad=0, h_pad=0.1)
plt.sca(ax1)
plt.hist(rescale_log_return_one, 50, normed=1, facecolor='green', alpha=0.75)
plt.sca(ax2)
plt.hist(rescale_log_return_five, 50, normed=1, facecolor='green', alpha=0.75)
plt.sca(ax3)
plt.hist(rescale_log_vol_one, 50, normed=1, facecolor='green', alpha=0.75)
plt.sca(ax4)
plt.hist(rescale_log_vol_five, 50, normed=1, facecolor='green', alpha=0.75)
plt.sca(ax5)
plt.hist(rescale_delta_log_high_low, 50, normed=1, facecolor='green', alpha=0.75)
plt.savefig('rescale_index_statistic_hist.jpg')
plt.show()


#定义可观察状态数据集
x = np.column_stack([rescale_log_return_one,rescale_log_return_five,rescale_log_vol_one,rescale_log_vol_five,rescale_delta_log_high_low])

'''
4.将可观察状态数据集带入隐马尔科夫模型，求解隐状态序列
'''

#计算最大回撤函数
def get_max_drawdown(net_list):
    
    drawdown = []
    if net_list[0]<1:
        drawdown.append((1-net_list[0]))
        
    length = len(net_list)
    for i in range(0,length-1):
        start = net_list[i]
        end = np.min(net_list[i+1:])
        if end <start:
            drawdown.append((start-end)/start)
    
    if len(drawdown) ==0:
        return 0
    else:
        return np.max(drawdown)
    
#定义样本内与样本外数据,样本外数据为200个
sample = x[:-200:]
out_sample = x[-200::]

#模型循环100次
state_number = 10
k = 1 #模型参数寻优次数
win_percentage = []#胜率
accumulate_return_rate = [] #累积收益率
u = [] #策略平均正收益率序列
d = [] #策略平均负收益率序列
r = [] #策略平均收益率序列
v = [] #策略平均收益率波动率序列
b = [] #最大回撤序列
while k<100:
    print('第',str(k),'次模拟')
    #模型拟合
    model = GaussianHMM(n_components = state_number, covariance_type = "diag", n_iter = 2000).fit(sample)
    hidden_states = model.predict(sample)
    hidden_states_1 = model.predict(out_sample)
    
    #画出每个隐状态下对应的收盘价价格
    plt.figure(figsize=(15,8))
    for i in range(model.n_components):
        pos = (hidden_states == i)
        plt.plot_date(new_date[:-200][pos],close_price[:-200][pos],'.',label = 'hidden state %d'%i,lw=3)
        plt.legend()
        plt.grid(1)
    plt.savefig('close_price under each hidden state.jpg')
    plt.show()
    
    
    #画出每个隐状态下做多策略收益率数据
    state_log_return = pd.DataFrame(index = new_date[:-200])
    state_log_return['log_return_one'] = log_return_one[:-200]
    state_log_return['state'] = hidden_states
    plt.figure(figsize=(15,8))
    states_sample_end_value = [] #记录每个隐状态下在样本内最终净值大小
    for i in range(model.n_components):
        pos = (hidden_states == i)
        pos = np.append(0,pos[:-1])
        state_log_return['state_return%s'%i] = state_log_return['log_return_one'].multiply(pos)
        plt.plot_date(new_date[:-200],np.exp(state_log_return['state_return%s'%i].cumsum()),'-', label='hidden state %d'%i)
        plt.legend()
        plt.grid(1)
        states_sample_end_value.append(np.exp(state_log_return['state_return%s'%i].cumsum())[-1])
    plt.savefig('net value under each hidden state.jpg')
    plt.show()
    
    #找出样本内数据中收益最高且为正收益的两个隐状态以及收益最低且为负收益的两个隐状态
    contem =  states_sample_end_value
    contem = sorted(contem)
    nagetive_one = [i for i in range(state_number) if states_sample_end_value[i] == contem[0] and contem[0]<1]
    nagetive_two = [i for i in range(state_number) if states_sample_end_value[i] == contem[1] and contem[0]<1]
    positive_one = [i for i in range(state_number) if states_sample_end_value[i] == contem[-1] and contem[0]>1]
    positive_two = [i for i in range(state_number) if states_sample_end_value[i] == contem[-2] and contem[0]>1]
    
    #在样本外数据中，选择收益最高的两个隐状态第二天做多，收益最低的两个隐状态第二天做空
    sig_long = []
    sig_short = []
    for i in hidden_states_1:
        if i==positive_one or i==positive_two:
            sig_long.append(1)
            sig_short.append(0)
        elif i==nagetive_one or i==nagetive_two:
            sig_short.append(1)
            sig_long.append(0)
        else:
            sig_long.append(0)
            sig_short.append(0)       
    sig_long = np.append(0,sig_long[:-1])
    sig_short = np.append(0,sig_short[:-1])
    sig_long = pd.Series(sig_long, index = new_date[-200:])
    sig_short = pd.Series(sig_short,index = new_date[-200:])
    str_return = pd.DataFrame(index = new_date[-200:])
    str_return['log_retrun_one'] = log_return_one[-200:]
    str_return['return'] = str_return['log_retrun_one'].multiply(sig_long,axis=0) - str_return['log_retrun_one'].multiply(sig_short,axis=0)
    
    #画出样本外策略收益净值曲线
    plt.figure(figsize=(15,8))
    plt.plot_date(new_date[-200:],np.exp(str_return['return'].cumsum()),'-', label='out sample strategy net value')
    plt.legend()
    plt.grid(1)
    plt.savefig('out sample strategy net value.jpg')
    plt.show()
    
    #画出样本外隐状态对应收盘价数据
    plt.figure(figsize=(15,8))
    for i in range(model.n_components):
        pos = (hidden_states_1 == i)
        plt.plot_date(new_date[-200:][pos],close_price[-200:][pos],'.',label = 'hidden state %d'%i,lw=3)
        plt.legend()
        plt.grid(1)
    plt.savefig('out sample close price under each hidden state.jpg')
    plt.show()
    
    #计算样本外策略回测指标
    net_value = np.exp(str_return['return'].cumsum())
    positive_return = np.array(str_return['return'])[np.array(str_return['return'])>0]
    negative_return = np.array(str_return['return'])[np.array(str_return['return'])<0]
    u.append(np.mean(positive_return))
    d.append(np.mean(negative_return))
    r.append(np.mean(str_return['return']))
    v.append(np.std(str_return['return']))
    b.append(get_max_drawdown(net_value))
    win_percentage.append(len(positive_return)/(len(positive_return)+len(negative_return)))
    accumulate_return_rate.append(np.exp(str_return['return'].cumsum())[-1]-1)
    
    k=k+1

result_statistic = pd.DataFrame(index = ['min','max','avg'])
result_statistic['average_return'] = [min(r),max(r),np.mean(r)]
result_statistic['average_positive_return'] = [min(u),max(u),np.mean(u)]
result_statistic['average_negative_return'] = [min(d),max(d),np.mean(d)]
result_statistic['win_percentage'] = [min(win_percentage),max(win_percentage),np.mean(win_percentage)]
result_statistic['accumulate_return_rate'] = [min(accumulate_return_rate),max(accumulate_return_rate),np.mean(accumulate_return_rate)]
result_statistic['volotility'] = [min(v),max(v),np.mean(v)]
result_statistic['max_drawback'] = [min(b),max(b),np.mean(b)]

result_statistic.to_excel('result.xls')




