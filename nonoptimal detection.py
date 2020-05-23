#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 21 23:29:55 2020

@author: tang
"""

import xlrd

import csv
from scipy.interpolate import griddata
import xlwt
import xlsxwriter
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn import linear_model

from scipy.stats import kendalltau
from mpl_toolkits.basemap import Basemap
from PIL import Image

import math

import netCDF4 as nc

from scipy import interpolate

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
#import sys
#reload(sys)
#sys.setdefaultencoding('utf8')
from mpl_toolkits.mplot3d import Axes3D 
from collections import Counter
import time
from matplotlib.lines import Line2D
import seaborn as sns
from scipy.stats import norm
from scipy.stats import t

from scipy.stats import spearmanr

import datetime

"""
Index:

Executive code for Figures and Tables
    
Figure 2   -   line 63 - 112
Figure 3(a)-   line 117 - 182
Table S4  -    line 188 - 275

"""

"""
# Figure 2

address = '/Volumes/NO NAME 1/output/xlsx_files'
title_list = ['AF','AO','AS','EU','NA','SA','ML','TR','LH']

test_xlsx_v_trend = np.zeros(shape=(len(title_list)))
test_xlsx_pi_trend = np.zeros(shape=(len(title_list),461))

for m in range(len(title_list)):
    
    #picontrol 部分

    xlsx_name = 'pi_' + title_list[m] + '_list_1_5.xlsx'
        
    test_xlsx_pi = read_xlrd(xlsx_name,address)
        
    for i in range(test_xlsx_pi_trend.shape[1]):
    
        test_xlsx_pi_trend[m,i] = trend_1d(test_xlsx_pi[i,:])
    
    #cci 部分
    
    xlsx_name = 'cfc_' + title_list[m] + '.xlsx'
    
    test_xlsx_v = read_xlrd(xlsx_name,address)    
    
    test_xlsx_v_trend[m] = trend_1d(test_xlsx_v)

x_label = 'Trends of cloud fraction(%)'
y_label = 'Number of occurrences'
address = '/Volumes/NO NAME 1/output/Figures'
savepicture = 1
t_data = test_xlsx_pi_trend
o_data = test_xlsx_v_trend
range_group = [2,2,2,2,2,2,1,0.75,0.5]
bins_group = [16,16,16,16,16,16,16,12,10]
title_list = ['AF','AO','AS','EU','NA','SA','ML','TR','LH']
ticks_num = [9,9,9,9,9,9,5,7,5]
ticks_label = [['-2.0','-1.5','-1.0','-0.5','0','0.5','1.0','1.5','2.0'],
               ['-2.0','-1.5','-1.0','-0.5','0','0.5','1.0','1.5','2.0'],
               ['-2.0','-1.5','-1.0','-0.5','0','0.5','1.0','1.5','2.0'],
               ['-2.0','-1.5','-1.0','-0.5','0','0.5','1.0','1.5','2.0'],
               ['-2.0','-1.5','-1.0','-0.5','0','0.5','1.0','1.5','2.0'],
               ['-2.0','-1.5','-1.0','-0.5','0','0.5','1.0','1.5','2.0'],
               ['-1.0','-0.5','0','0.5','1.0'],
               ['-0.75','-0.50','-0.25','0','0.25','0.50','0.75'],
               ['-0.5','-0.25','0','0.25','0.5']]
title_height = 20

distribution_trend(t_data,o_data,x_label,y_label,address,savepicture,range_group,bins_group,ticks_num,title_list,title_height)

"""

'''
# Figure 3(a)

# NAT/GHG/AER forcings

address = '/Volumes/NO NAME 1/output/xlsx_files' 
title_list = ['AF','AO','AS','EU','NA','SA','ML','TR','LH']
forcing_list = ['nat','ghg','aer']

trend_list = np.zeros(shape=(3,9))
std_list = np.zeros(shape=(3,9))

for n in range(len(forcing_list)):
    for m in range(len(title_list)):
            
        xlsx_name = forcing_list[n] + '_' + title_list[m] + '.xlsx'
            
        test_xlsx_v = read_xlrd(xlsx_name,address)
    
        if xlsx_name[:3] == 'ghg' or xlsx_name[:3] == 'aer':
            row_num = [10,5,4,10,3]
            test_xlsx = matrix_row_plus(test_xlsx_v,row_num)
            test_trend = np.zeros(shape=(5))
        elif xlsx_name[:3] == 'nat':
            row_num = [10,1,5,4,10,3]
            test_xlsx = matrix_row_plus(test_xlsx_v,row_num)
            test_trend = np.zeros(shape=(6))  
        for i in range(test_trend.shape[0]):
            test_trend[i] = trend_1d(test_xlsx[i,:])
        std_list[n,m] = np.std(test_trend)
        trend_list[n,m] = trend_1d(np.mean(test_xlsx,axis = 0))

point_value = np.zeros(shape=(1,t_data.shape[0]))
dy = np.zeros(shape=(1,t_data.shape[0]))
for i in range(point_value.shape[0]):
    for j in range(point_value.shape[1]):
        point_value[i,j] = 100
        
t_data = test_xlsx_pi_trend
o_data = test_xlsx_v_trend

t_std = np.zeros(shape=(t_data.shape[0]))

for i in range(t_data.shape[0]):
    m_data = t_data[i,:]
    t_std[i] = np.std(m_data)
    dy[0,i] = abs(100*(t_std[i]/o_data[i]))
        
point_value_2 = np.zeros(shape=(4,t_data.shape[0]))
dy_2 = np.zeros(shape=(4,t_data.shape[0]))
point_value_2[0,:] =  point_value[0,:]
dy_2[0,:] = dy[0,:]

for i in range(len(forcing_list)):
    for j in range(len(title_list)):
        point_value_2[i+1,j] = abs(100*(trend_list[i,j]/o_data[j]))
        dy_2[i+1,j] = abs(100*(std_list[i,j]/o_data[j]))
        
        
row_col = [1,1]
subtitle = ''
var_name = np.array(['F1','F2 NAT','F2 GHG','F2 AER'])
xticks = np.array(['AF','AO','AS','EU','NA','SA','ML','TR','LH'])
color_list = ['b','g','r','y']
savepictures = 1      
        
percent_range_compare(point_value_2,dy_2,row_col,subtitle,var_name,xticks,color_list,savepictures)

'''


'''
# Table S4

address = '/Volumes/NO NAME 1/output/xlsx_files'
title_list = ['AF','AO','AS','EU','NA','SA','ML','TR','LH']

"""
#pi_control
"""

test_xlsx_pi_trend = np.zeros(shape=(len(title_list),461,33))

for m in range(len(title_list)):
    
    """
    #picontrol 部分
    """

    xlsx_name = 'pi_' + title_list[m] + '_list_1_5.xlsx'
        
    test_xlsx_pi = read_xlrd(xlsx_name,address)
        
    test_xlsx_pi_trend[m,:,:] = test_xlsx_pi[:,:]

test_xlsx_pi_trend_mean = np.mean(test_xlsx_pi_trend,axis = 1)
   
"""
#observation
"""     
test_xlsx_v_trend = np.zeros(shape=(len(title_list)))

test_xlsx_v_trend = np.zeros(shape=(len(title_list),33))

for m in range(len(title_list)):
    
    xlsx_name = 'cfc_' + title_list[m] + '.xlsx'
    
    test_xlsx_v = read_xlrd(xlsx_name,address)    
    
    test_xlsx_v_trend[m,:] = test_xlsx_v[:]       
        
        
#相关性平均        

vec1 = test_xlsx_pi_trend

vec2 = test_xlsx_v_trend

r = np.zeros(shape=(9,461))
p = np.zeros(shape=(9,461))
rt = np.zeros(shape=(9))
pt = np.zeros(shape=(9))

for i in range(9):
    for j in range(461):
        r[i,j], p[i,j] = stats.pearsonr(vec1[i,j,:], vec2[i,:])
    rt[i] = np.mean(r[i,:])
    pt[i] = np.mean(p[i,:])        

# r,p
    
vec1 = test_xlsx_pi_trend_mean

vec2 = test_xlsx_v_trend

r = np.zeros(shape=(9))
p = np.zeros(shape=(9))

for i in range(9):
    r[i], p[i] = stats.pearsonr(vec1[i,:], vec2[i,:])
    
# signal_to_noise

data_std = np.zeros(shape=(9))

for i in range(9):
    data_std[i] = np.std(t_data[i,:])

signal_to_noise = o_data/data_std

save_address = '/Volumes/NO NAME 1/output/Tables/list_cor_pi.xlsx'
    
list_cor_pi = np.zeros(shape=(9,3))
list_cor_pi[:,0] = np.round(r,3)
list_cor_pi[:,1] = np.round(p,3)
list_cor_pi[:,2] = np.abs(np.round(signal_to_noise,3))

# save data
creat_xlsx_2d(list_cor_pi,save_address)

'''

"""
#计算picontrol和cci趋势
"""

def trend_1d(m_data):

    x = np.linspace(1,m_data.shape[0],m_data.shape[0])
    
    pfit = np.polyfit(x,m_data,1)
    y_fun = np.poly1d(pfit)
    out_map = (y_fun(x)[-1] - y_fun(x)[0])   
            
    return out_map


def distribution_trend(t_data,o_data,x_label,y_label,address,savepicture,range_group,bins_group,ticks_num,title_list,title_height):

    plt.figure(figsize=(28,20))
    
    i = -1
    
    for m in range(3):
        for n in range(3):
            
            i += 1
            
            m_data = t_data[i,:]
            
            map_location = int(str(3) + str(3) + str(i + 1))
            
            plt.subplot(map_location)
            font_title = {'family' : 'Times New Roman',
                'color'  : 'black',
                'weight' : 'semibold',
                'size'   : 40,
            }
            font = {'family' : 'Times New Roman',
            'color'  : 'black',
            'weight' : 'semibold',
            'size'   : 30,
            }
            plt.title(title_list[i],fontdict=font_title, pad= title_height)
            
            plt.hist(x = m_data, # 指定绘图数据
                     range=(-range_group[i],range_group[i]),bins = bins_group[i], # 指定直方图中条块的个数
                     color = 'steelblue', # 指定直方图的填充色
                     edgecolor = 'black', # 指定直方图的边框色
                     align = 'mid',
                     label = str(round(np.std(m_data),2))
                     )
            
            plt.axvline(o_data[i],color = 'black',linestyle = '--',linewidth = 4,label = str(round(o_data[i],2)))
            plt.yticks(fontsize=25)
            x_ticks = np.linspace(-range_group[i],range_group[i],ticks_num[i]) 
            plt.xticks(np.array(x_ticks),(ticks_label[i]))
            plt.xticks(fontsize=25)
            
            if m == 2:
                plt.xlabel(x_label,fontdict=font)
            if n == 0:
                plt.ylabel(y_label,fontdict=font) 
                
            plt.legend(fontsize=25,frameon = True,shadow = True)
            
            plt.tight_layout()
            
    if savepicture:
        os.chdir(address)
        #savename = input("输入保存名（histogram pictures）：\n")
        savename = 'Figure 2'       
        plt.savefig(savename)

    
def percent_range_compare(point_value,dy,row_col,subtitle,var_name,xticks,color_list,savepictures):
    
    x = np.zeros(shape=(point_value.shape[0]+1,point_value.shape[1]))
    for i in range(point_value.shape[0]+1):
        for j in range(point_value.shape[1]):
            x[i,j] = i*2+j*10
            
    plt.figure(figsize=(14, 7))
    font_legend = {'family' : 'Times New Roman',
            'style'  : 'oblique',
            'weight' : 'semibold',
            'size'   : 24,
    }
    for m in range(1):
    
        map_location = int(str(row_col[0]) + str(row_col[1]) + str(m+1))
        
        plt.subplot(map_location) #确定第一个图的位置
        
        plt.style.use('seaborn-whitegrid')
         
        #fmt是一种控制线条和点的外观的代码格式,语法与plt.plot的缩写代码一样.
        #plt.axhline(0,color = 'black',linestyle = '-',linewidth = 2.5)
        plt.axhline(100,color = 'black',linestyle = '--',linewidth = 4)
        plt.ylabel('Observed trend due to external forcing(%)',fontsize=24) 
        plt.title(subtitle,fontproperties='SimHei',fontsize = 24) 
        
        for i in range(point_value.shape[0]):
            if i == 0:
                plt.errorbar(x[i,:],point_value[i,:] ,yerr=dy[i,:],label = var_name[i],fmt='o',ecolor=color_list[i],color='black',elinewidth=10,capsize=6)
            else:
                plt.errorbar(x[i,:],point_value[i,:] ,yerr=dy[i,:],label = var_name[i],fmt='o',ecolor=color_list[i],color='black',elinewidth=4,capsize=6)
        
        #for j in range(point_value.shape[1]):
        #    plt.axvline(x[4,j],color = 'black',linestyle = 'dotted',linewidth = 3)

        #plt.axvline(17,color = 'black',linestyle = '--',linewidth = 2.5)
        #plt.axvline(27,color = 'grey',linestyle = '--',linewidth = 2.5)
        plt.ylim(-50,200)
        """
        设置纵坐标范围
        """
        plt.xticks(x[0,:]+3,xticks,fontsize = 24)
        plt.yticks(fontsize = 20)
        plt.legend(prop=font_legend,frameon = True,ncol=4,loc = 'upper right',shadow = True)
        #plt.grid()        #删除网格线
        plt.tight_layout()
    #plt.suptitle('Title',fontproperties='SimHei',fontsize = 28)
    if savepictures:
        os.chdir('/Volumes/NO NAME 1/output/Figures')
        plt.savefig('Figure 3(a)')    
    

def matrix_row_plus(data,row_num):#矩阵列向相加
    
    out_data = np.zeros(shape=(len(row_num),data.shape[1]))
    i_start = 0
    
    for i in range(len(row_num)):
        out_data[i,:] = np.mean(data[int(i_start):int(i_start+row_num[i]),:],axis=0)
        i_start += row_num[i]
    
    return out_data
    


