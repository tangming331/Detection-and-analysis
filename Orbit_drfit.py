#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 23 00:26:56 2020

@author: tang
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn import linear_model

from scipy.stats import kendalltau
from mpl_toolkits.basemap import Basemap
from PIL import Image
#数学计算
import math
#nc文件处理 
import netCDF4 as nc
#插值
from scipy import interpolate
#归一分析与还原分析
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
#import sys
#reload(sys)
#sys.setdefaultencoding('utf8')
from mpl_toolkits.mplot3d import Axes3D  # 空间三维画图
from collections import Counter
import time
from matplotlib.lines import Line2D
import seaborn as sns
from scipy.stats import norm
from scipy.stats import t
import os


def ob_list(data,time_ob):
    
    new_data = np.zeros(shape=(time_ob.shape[0]))
    
    for i in range(time_ob.shape[0]):
        new_data[i] = data[int(time_ob[i])]
        
    return new_data


def fit_1D(mat1,time_ob):
    
    y_fit = np.zeros(shape=(mat1.shape[0]))
    valid = ob_list(mat1,time_ob)

    pfit = np.polyfit(time_ob,valid,2)
    y_fun = np.poly1d(pfit)           
    for i in range(mat1.shape[0]):
        if i not in time_ob:
            y_fit[i] = y_fun(i)
        else:
            y_fit[i] = mat1[i]

    return y_fit


cross_list = np.zeros(shape=(396))
cross_time = np.zeros(shape=(396))

cross_list[0] = 14.6
cross_list[12] = 14.97
cross_list[24] = 15.45
cross_list[36] = 16
cross_list[37] = 16.05

time_ob1 = np.array([0,12,24,36,37])

cross_list[38] = 14.4
cross_list[48] = 14.65
cross_list[60] = 14.97
cross_list[72] = 15.6
cross_list[82] = 16.1

time_ob2 = np.array([38,48,60,72,82])
time_ob2[:] -= 38

cross_list[83] = 13.68
cross_list[96] = 13.95
cross_list[108] = 14.3
cross_list[120] = 14.87
cross_list[132] = 15.5
cross_list[144] = 16.33
cross_list[152] = 17.05

time_ob3 = np.array([83,96,108,120,132,144,152])
time_ob3[:] -= 83

cross_list[157] = 13.65
cross_list[168] = 13.8
cross_list[180] = 14.03
cross_list[192] = 14.45
cross_list[204] = 15
cross_list[216] = 15.7
cross_list[227] = 16.47

time_ob4 = np.array([157,168,180,192,204,216,227])
time_ob4[:] -= 157

cross_list[228] = 13.95
cross_list[240] = 13.9
cross_list[252] = 13.98
cross_list[264] = 14.08
cross_list[276] = 14.5
cross_list[287] = 15

time_ob5 = np.array([228,240,252,264,276,287])
time_ob5[:] -= 228

cross_list[288] = 13.88
cross_list[300] = 13.6
cross_list[312] = 13.5
cross_list[324] = 13.6
cross_list[335] = 13.77

time_ob6 = np.array([288,300,312,324,335])
time_ob6[:] -= 288

cross_list[336] = 13.75
cross_list[348] = 13.53
cross_list[360] = 13.5
cross_list[372] = 13.53
cross_list[388] = 13.75
cross_list[395] = 14

time_ob7 = np.array([336,348,360,372,388,395])
time_ob7[:] -= 336

cross_time[0:38] = fit_1D(cross_list[0:38],time_ob1)
cross_time[38:83] = fit_1D(cross_list[38:83],time_ob2)
cross_time[83:153] = fit_1D(cross_list[83:153],time_ob3)
cross_time[157:228] = fit_1D(cross_list[157:228],time_ob4)
cross_time[228:288] = fit_1D(cross_list[228:288],time_ob5)
cross_time[288:336] = fit_1D(cross_list[288:336],time_ob6)
cross_time[336:] = fit_1D(cross_list[336:],time_ob7)

cross_time[153] = cross_time[152]
cross_time[154] = cross_time[152]
cross_time[155] = cross_time[152]
cross_time[156] = cross_time[157]


# Model 3 Amplitude

os.chdir('/Volumes/NO NAME 1/output/EOF_output')
norm_PCData = np.load('norm_PCData.npy')

plt.figure(figsize=(10,12))

x = cross_time
y = (norm_PCData[:,2])
seris = np.linspace(min(x[:,]),max(x[:,]),100)

pfit = np.polyfit(x,y,3)
y_fun = np.poly1d(pfit)
plt.plot(seris,y_fun(seris),color = 'black',linewidth = 5, label = 'regression curve') 

# 绘制办公用品的气泡图
plt.scatter(x[0:38],y[0:38],color = 'steelblue',s=45, label = 'NOAA-7',marker='v', alpha = 0.6)
plt.scatter(x[38:83],y[38:83],color = 'indianred',s=45, label = 'NOAA-9',marker='<', alpha = 0.6)
plt.scatter(x[83:153],y[83:153],color = 'grey',s=45, label = 'NOAA-11',marker='>', alpha = 0.6)
plt.scatter(x[157:228],y[157:228],color = 'orange',s=45, label = 'NOAA-14',marker='8', alpha = 0.6)
plt.scatter(x[228:288],y[228:288],color = 'brown',s=45, label = 'NOAA-16',marker='s', alpha = 0.6)
plt.scatter(x[288:336],y[288:336],color = 'purple',s=45, label = 'NOAA-18',marker='8', alpha = 0.6)
plt.scatter(x[336:],y[336:],color = 'green',s=45, label = 'NOAA-19',marker='p', alpha = 0.6)

plt.xticks([13.5,14.0,14.5,15,15.5,16,16.5,17],['13:30', '14:00', '14:30', '15:00', '15:30','16:00','16:30','17:00'], fontsize=18)
plt.yticks([-40,-20,0,20,40,60],['-40', '-20', '0', '20', '40','60'], fontsize=18)
# 添加x轴和y轴标签
plt.xlabel('Equatorial Crossing Time(ECT)',fontsize=24)
plt.ylabel('Mode 3 Amplitude',fontsize=24)
# 添加标题
plt.title('(a) Analysis of afternoon drift',fontsize=28)
# 添加图例

plt.legend(prop={'size':18})
plt.grid()
# 显示图形
os.chdir('/Volumes/NO NAME 1/output/Figures')
plt.savefig('Figure S1(1)')
plt.show()



mode_time = np.linspace(0,395,396)
syn_line = np.zeros(shape = (396))

for i in range(396):
    syn_line[i] = y_fun(cross_time[i])
    
plt.figure(figsize=(18,7))

plt.plot(mode_time,norm_PCData[:,2],color = 'steelblue',linewidth = 3, label = 'Original Mode 3') 
plt.plot(mode_time,syn_line,color = 'black',linewidth = 4, label = 'Synthetic Mode 3') 

plt.xticks([0,36,72,108,144,180,216,252,288,324,360],['1982', '1985', '1988', '1991', '1994','1997','2000','2003', '2006','2009','2012'], fontsize=18)
plt.yticks([-40,-20,0,20,40,60],['-40', '-20', '0', '20', '40','60'], fontsize=18)

plt.xlabel('Year',fontsize=24)
plt.ylabel('Amplitude',fontsize=24)
# 添加标题
plt.title('(b) Analysis of Satellite Transition - REOF Mode 3',fontsize=24)
# 添加图例
plt.legend(prop={'size':18})
plt.grid()
# 显示图形
os.chdir('/Volumes/NO NAME 1/output/Figures')
plt.savefig('Figure S1(2)')
plt.show()
