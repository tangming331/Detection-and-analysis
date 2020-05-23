#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 21 23:08:15 2020

@author: tang
"""

import xlrd
import random
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
    
Table S5   -   line 140 - 218
Figure 1   -   line 227 - 287
Figure S3  -   line 290 - 305
Figure 4   -   line 311 - 473
Figure S4  -   line 476 - 525
Figure 3(b)/Figure 3(c)   -   line 530 - 627

"""

"""
area：

#SA: 
SWpoint_SA = [60,0]
NEpoint_SA = [120,150]

#NA:
SWpoint_NA = [120,0]
NEpoint_NA = [180,150]

#AF:
SWpoint_AF = [60,150]
NEpoint_AF = [120,240]    

#EU:
SWpoint_EU = [120,150]
NEpoint_EU = [180,240]  

#AO:
SWpoint_AO = [60,240]
NEpoint_AO = [120,360]  

#AS:
SWpoint_AS = [120,240]
NEpoint_AS = [180,360] 

#ML:
SWpoint_ML = [120,0]
NEpoint_ML = [180,360] 

#TR
SWpoint_TR = [60,0]
NEpoint_TR = [120,360] 

#LH
SWpoint_LH = [60,0]
NEpoint_LH = [180,360] 


"""

"""

cci observation:
cci_annual_origdata_r

cmip data：
clt_nat_origdata
clt_ghg_origdata
clt_aer_origdata

"""

#============================================================================================
# clisccp process

"""
clisccp_SA_list = np.zeros(shape=(33,7,6))
clisccp_NA_list = np.zeros(shape=(33,7,6))
clisccp_AS_list = np.zeros(shape=(33,7,6))
clisccp_AO_list = np.zeros(shape=(33,7,6))
clisccp_AF_list = np.zeros(shape=(33,7,6))
clisccp_EU_list = np.zeros(shape=(33,7,6))

"""



"""
# Table S5 code
# read obs data

os.chdir('/Volumes/NO NAME 1/output/Base_data/Cloud_cci')
ci_annual_origdata_r = np.load('ci_annual_origdata_r.npy')
os.chdir('/Volumes/NO NAME 1/output/Base_data/cmip_clt')
clt_aer_origdata = np.load('clt_aer_origdata.npy')
clt_ghg_origdata = np.load('clt_ghg_origdata.npy')
clt_nat_origdata = np.load('clt_nat_origdata.npy')

cci_list_group = np.zeros(shape= (9,33))

for i in range(len(SWpoint_ensemble)):
    cci_list_group[i,:] = extract_list_3d(cci_annual_origdata_r,SWpoint_ensemble[i],NEpoint_ensemble[i],mean_num)*100

# 模拟平均值
    
#cmip_trend = clt_ghg_origdata
#row_num = [10,5,4,10,3]
#row_num = [10,1,5,4,10,3]
    
    
ghg_list_group = np.zeros(shape= (9,33))
aer_list_group = np.zeros(shape= (9,33))
nat_list_group = np.zeros(shape= (9,33))

# 云量平均

def temp_map_trend(cmip_trend,row_num,SWpoint,NEpoint,mean_num):#临时函数，存在重复

    num = 0
    clt_total = 0
    for i in range(len(row_num)):
        clt_valid = 0
        clt_s = np.zeros(shape= (33))
        for j in range(row_num[i]):
            clt_valid = valid_map(cmip_trend[num,:,:,:])
            clt_s += extract_list_3d(clt_valid,SWpoint,NEpoint,mean_num)
            num += 1
        clt_s = clt_s/row_num[i]
        clt_total += clt_s
    
    clt_total = clt_total/len(row_num)
    
    return clt_total
    
    
for i in range(ghg_list_group.shape[0]):
    ghg_list_group[i,:] = temp_map_trend(clt_ghg_origdata,[10,5,4,10,3],SWpoint_ensemble[i],NEpoint_ensemble[i],mean_num)
for i in range(aer_list_group.shape[0]):
    aer_list_group[i,:] = temp_map_trend(clt_aer_origdata,[10,5,4,10,3],SWpoint_ensemble[i],NEpoint_ensemble[i],mean_num)
for i in range(nat_list_group.shape[0]):
    nat_list_group[i,:] = temp_map_trend(clt_nat_origdata,[10,1,5,4,10,3],SWpoint_ensemble[i],NEpoint_ensemble[i],mean_num)


# 平均值相关性
    
list_cor_ghg = np.zeros(shape=(9,2))
list_cor_aer = np.zeros(shape=(9,2))
list_cor_nat = np.zeros(shape=(9,2))


for i in range(9):
    list_cor_ghg[i,:] = stats.pearsonr(cci_list_group[i,:], ghg_list_group[i,:])
    list_cor_aer[i,:] = stats.pearsonr(cci_list_group[i,:], aer_list_group[i,:])
    list_cor_nat[i,:] = stats.pearsonr(cci_list_group[i,:], nat_list_group[i,:])

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
    
save_address = '/Volumes/NO NAME 1/output/Tables/list_cor_cmip.xlsx'
    
list_cor_cmip = np.zeros(shape=(9,6))
list_cor_cmip[:,:2] = np.round(list_cor_aer,3)
list_cor_cmip[:,2:4] = np.round(list_cor_ghg,3)
list_cor_cmip[:,4:] = np.round(list_cor_nat,3)

# save data
creat_xlsx_2d(list_cor_cmip,save_address)
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
"""


"""
# Figure 1 

obs_data = np.zeros(shape=(9,33))

data = np.zeros(shape=(9,17,33))

address_ensemble = ['/Volumes/NO NAME 1/output/xlsx_files/cfc','/Volumes/NO NAME 1/output/xlsx_files/nat','/Volumes/NO NAME 1/output/xlsx_files/ghg','/Volumes/NO NAME 1/output/xlsx_files/aer']

sim_nat_data = np.zeros(shape=(9,6,33))
sim_aer_data = np.zeros(shape=(9,5,33))
sim_ghg_data = np.zeros(shape=(9,5,33))

for n in range(len(address_ensemble)):
    
    address = address_ensemble[n]
    ff = ff_open(address)
    
    for m in range(len(ff)):
        
        count = 0
        xlsx_name = ff[m]
        test_xlsx_v = read_xlrd(xlsx_name,address)
        
        if ff[m][:3] == 'aer':
            row_num = [10,5,4,10,3]
            acc = 0
            for i in range(len(row_num)):
                sim_aer_data[m,count,:] = np.mean(test_xlsx_v[acc:int(acc+row_num[i]),:],axis = 0)
                count += 1
                acc += row_num[i]
        elif ff[m][:3] == 'ghg':
            row_num = [10,5,4,10,3]
            acc = 0
            for i in range(len(row_num)):
                sim_ghg_data[m,count,:] = np.mean(test_xlsx_v[acc:int(acc+row_num[i]),:],axis = 0)
                count += 1
                acc += row_num[i]                      
        elif ff[m][:3] == 'nat':
            row_num = [10,1,5,4,10,3]
            acc = 0
            for i in range(len(row_num)):
                sim_nat_data[m,count,:] = np.mean(test_xlsx_v[acc:int(acc+row_num[i]),:],axis = 0)
                count += 1
                acc += row_num[i]      
        else:
            obs_data[m,:] = test_xlsx_v
            
data[:,0,:] = obs_data[:,:]        
data[:,1:7,:] = sim_nat_data[:,:,:]
data[:,7:12,:] = sim_ghg_data[:,:,:]
data[:,12:,:] = sim_aer_data[:,:,:]
         
grid_scale = 33

initial_num = [6,5,5]
color_list = ['black','blue','red','lightgreen']
label_list = ['OBS:','NAT','GHG','AER']
title_list = ['AF','AO','AS','EU','LH','ML','NA','SA','TR']
map_list = [1,2,3,4,9,7,5,6,8]
savepictures = 1

# depict

variety_multiple_forcing(data,initial_num,grid_scale,color_list,label_list,title_list,map_list,savepictures)

# Figure S3

data = np.zeros(shape=(9,4,11))

address_ensemble = ['/Volumes/NO NAME 1/output/DA_xlsx/cfc','/Volumes/NO NAME 1/output/DA_xlsx/nat','/Volumes/NO NAME 1/output/DA_xlsx/ghg','/Volumes/NO NAME 1/output/DA_xlsx/aer']

for n in range(len(address_ensemble)):
    
    address = address_ensemble[n]
    ff = ff_open(address)
    
    for m in range(len(ff)):
        xlsx_name = ff[m]
        data[m,n,:] = read_xlrd(xlsx_name,address)

scale_variety_multiple(data,grid_scale,color_list,label_list,title_list,map_list,savepictures)

"""

"""

# Figure 4

# obs 

clisccp_LH_trend = np.zeros(shape = (7,6))
clisccp_LH_trend = histogram_trend(clisccp_LH_list*100)

clisccp_ML_trend = np.zeros(shape = (7,6))
clisccp_ML_trend = histogram_trend(clisccp_ML_list)

clisccp_TR_trend = np.zeros(shape = (7,6))
clisccp_TR_trend = histogram_trend(clisccp_TR_list)

# simulation

time_start = '198201'
time_end = '201412'
yearormonth = 'year'
valid_delete = 1 #是否删除空白月份
SWpoint_ensemble = [[120,0],[60,0],[60,0]]
NEpoint_ensemble = [[180,360],[180,360],[120,360]]
mean_num = 1
front_word = 'cli'

address_aer = '/Volumes/TOSHIBA EXT/Work（2020.1.15—2020.2.08）/CMIP-clisccp 数据全集/aer'
address_ghg = '/Volumes/TOSHIBA EXT/Work（2020.1.15—2020.2.08）/CMIP-clisccp 数据全集/ghg'
address_nat = '/Volumes/TOSHIBA EXT/Work（2020.1.15—2020.2.08）/CMIP-clisccp 数据全集/nat'

clisccp_aer_ML_mean_list = clisccp_set(address_aer,front_word,time_start,time_end,yearormonth,mean_num,valid_delete,SWpoint_ensemble[0],NEpoint_ensemble[0])
clisccp_nat_ML_mean_list = clisccp_set(address_nat,front_word,time_start,time_end,yearormonth,mean_num,valid_delete,SWpoint_ensemble[0],NEpoint_ensemble[0])
clisccp_ghg_ML_mean_list = clisccp_set(address_ghg,front_word,time_start,time_end,yearormonth,mean_num,valid_delete,SWpoint_ensemble[0],NEpoint_ensemble[0])
clisccp_aer_LH_mean_list = clisccp_set(address_aer,front_word,time_start,time_end,yearormonth,mean_num,valid_delete,SWpoint_ensemble[1],NEpoint_ensemble[1])
clisccp_nat_LH_mean_list = clisccp_set(address_nat,front_word,time_start,time_end,yearormonth,mean_num,valid_delete,SWpoint_ensemble[1],NEpoint_ensemble[1])
clisccp_ghg_LH_mean_list = clisccp_set(address_ghg,front_word,time_start,time_end,yearormonth,mean_num,valid_delete,SWpoint_ensemble[1],NEpoint_ensemble[1])
clisccp_aer_TR_mean_list = clisccp_set(address_aer,front_word,time_start,time_end,yearormonth,mean_num,valid_delete,SWpoint_ensemble[2],NEpoint_ensemble[2])
clisccp_nat_TR_mean_list = clisccp_set(address_nat,front_word,time_start,time_end,yearormonth,mean_num,valid_delete,SWpoint_ensemble[2],NEpoint_ensemble[2])
clisccp_ghg_TR_mean_list = clisccp_set(address_ghg,front_word,time_start,time_end,yearormonth,mean_num,valid_delete,SWpoint_ensemble[2],NEpoint_ensemble[2])


clisccp_aer_ML_list = np.zeros(shape=(33,7,6))
clisccp_nat_ML_list = np.zeros(shape=(33,7,6))
clisccp_ghg_ML_list = np.zeros(shape=(33,7,6))
clisccp_aer_LH_list = np.zeros(shape=(33,7,6))
clisccp_nat_LH_list = np.zeros(shape=(33,7,6))
clisccp_ghg_LH_list = np.zeros(shape=(33,7,6))
clisccp_aer_TR_list = np.zeros(shape=(33,7,6))
clisccp_nat_TR_list = np.zeros(shape=(33,7,6))
clisccp_ghg_TR_list = np.zeros(shape=(33,7,6))


clisccp_aer_ML_list = (np.mean(clisccp_aer_ML_mean_list[:3,:,:,:,:],axis = 0)[0] + np.mean(clisccp_aer_ML_mean_list[3:,:,:,:,:],axis = 0)[0])/2
clisccp_nat_ML_list = (np.mean(clisccp_nat_ML_mean_list[:3,:,:,:,:],axis = 0)[0] + np.mean(clisccp_nat_ML_mean_list[3:,:,:,:,:],axis = 0)[0])/2
clisccp_ghg_ML_list = (np.mean(clisccp_ghg_ML_mean_list[:3,:,:,:,:],axis = 0)[0] + np.mean(clisccp_ghg_ML_mean_list[3:,:,:,:,:],axis = 0)[0])/2
clisccp_aer_LH_list = (np.mean(clisccp_aer_LH_mean_list[:3,:,:,:,:],axis = 0)[0] + np.mean(clisccp_aer_LH_mean_list[3:,:,:,:,:],axis = 0)[0])/2
clisccp_nat_LH_list = (np.mean(clisccp_nat_LH_mean_list[:3,:,:,:,:],axis = 0)[0] + np.mean(clisccp_nat_LH_mean_list[3:,:,:,:,:],axis = 0)[0])/2
clisccp_ghg_LH_list = (np.mean(clisccp_ghg_LH_mean_list[:3,:,:,:,:],axis = 0)[0] + np.mean(clisccp_ghg_LH_mean_list[3:,:,:,:,:],axis = 0)[0])/2
clisccp_aer_TR_list = (np.mean(clisccp_aer_TR_mean_list[:3,:,:,:,:],axis = 0)[0] + np.mean(clisccp_aer_TR_mean_list[3:,:,:,:,:],axis = 0)[0])/2
clisccp_nat_TR_list = (np.mean(clisccp_nat_TR_mean_list[:3,:,:,:,:],axis = 0)[0] + np.mean(clisccp_nat_TR_mean_list[3:,:,:,:,:],axis = 0)[0])/2
clisccp_ghg_TR_list = (np.mean(clisccp_ghg_TR_mean_list[:3,:,:,:,:],axis = 0)[0] + np.mean(clisccp_ghg_TR_mean_list[3:,:,:,:,:],axis = 0)[0])/2


clisccp_aer_ML_mean = np.mean(clisccp_aer_ML_list[:,:,:],axis = 0)
clisccp_nat_ML_mean = np.mean(clisccp_nat_ML_list[:,:,:],axis = 0)
clisccp_ghg_ML_mean = np.mean(clisccp_ghg_ML_list[:,:,:],axis = 0)
clisccp_aer_LH_mean = np.mean(clisccp_aer_LH_list[:,:,:],axis = 0)
clisccp_nat_LH_mean = np.mean(clisccp_nat_LH_list[:,:,:],axis = 0)
clisccp_ghg_LH_mean = np.mean(clisccp_ghg_LH_list[:,:,:],axis = 0)
clisccp_aer_TR_mean = np.mean(clisccp_aer_TR_list[:,:,:],axis = 0)
clisccp_nat_TR_mean = np.mean(clisccp_nat_TR_list[:,:,:],axis = 0)
clisccp_ghg_TR_mean = np.mean(clisccp_ghg_TR_list[:,:,:],axis = 0)


clisccp_aer_ML_mean_trend = np.zeros(shape = (7,6))
clisccp_aer_ML_mean_trend[:,:] = histogram_trend(clisccp_aer_ML_list[:,:,:])
clisccp_nat_ML_mean_trend = np.zeros(shape = (7,6))
clisccp_nat_ML_mean_trend[:,:] = histogram_trend(clisccp_nat_ML_list[:,:,:])
clisccp_ghg_ML_mean_trend = np.zeros(shape = (7,6))
clisccp_ghg_ML_mean_trend[:,:] = histogram_trend(clisccp_ghg_ML_list[:,:,:])
clisccp_aer_LH_mean_trend = np.zeros(shape = (7,6))
clisccp_aer_LH_mean_trend[:,:] = histogram_trend(clisccp_aer_LH_list[:,:,:])
clisccp_nat_LH_mean_trend = np.zeros(shape = (7,6))
clisccp_nat_LH_mean_trend[:,:] = histogram_trend(clisccp_nat_LH_list[:,:,:])
clisccp_ghg_LH_mean_trend = np.zeros(shape = (7,6))
clisccp_ghg_LH_mean_trend[:,:] = histogram_trend(clisccp_ghg_LH_list[:,:,:])
clisccp_aer_TR_mean_trend = np.zeros(shape = (7,6))
clisccp_aer_TR_mean_trend[:,:] = histogram_trend(clisccp_aer_TR_list[:,:,:])
clisccp_nat_TR_mean_trend = np.zeros(shape = (7,6))
clisccp_nat_TR_mean_trend[:,:] = histogram_trend(clisccp_nat_TR_list[:,:,:])
clisccp_ghg_TR_mean_trend = np.zeros(shape = (7,6))
clisccp_ghg_TR_mean_trend[:,:] = histogram_trend(clisccp_ghg_TR_list[:,:,:])


histogram_group = np.zeros(shape = (3,4,7,6))

histogram_group[0,0,:,:] = clisccp_LH_trend
histogram_group[0,1,:,:] = clisccp_nat_LH_mean_trend[::-1]
histogram_group[0,2,:,:] = clisccp_ghg_LH_mean_trend[::-1]
histogram_group[0,3,:,:] = clisccp_aer_LH_mean_trend[::-1]

histogram_group[1,0,:,:] = clisccp_ML_trend
histogram_group[1,1,:,:] = clisccp_nat_ML_mean_trend[::-1]
histogram_group[1,2,:,:] = clisccp_ghg_ML_mean_trend[::-1]
histogram_group[1,3,:,:] = clisccp_aer_ML_mean_trend[::-1]

histogram_group[2,0,:,:] = clisccp_TR_trend
histogram_group[2,1,:,:] = clisccp_nat_TR_mean_trend[::-1]
histogram_group[2,2,:,:] = clisccp_ghg_TR_mean_trend[::-1]
histogram_group[2,3,:,:] = clisccp_aer_TR_mean_trend[::-1]


R_LH_nat = map_pearson(histogram_group[0,0,:,:],histogram_group[0,1,:,:])
R_LH_ghg = map_pearson(histogram_group[0,0,:,:],histogram_group[0,2,:,:])
R_LH_aer = map_pearson(histogram_group[0,0,:,:],histogram_group[0,3,:,:])
R_ML_nat = map_pearson(histogram_group[1,0,:,:],histogram_group[1,1,:,:])
R_ML_ghg = map_pearson(histogram_group[1,0,:,:],histogram_group[1,2,:,:])
R_ML_aer = map_pearson(histogram_group[1,0,:,:],histogram_group[1,3,:,:])
R_TR_nat = map_pearson(histogram_group[2,0,:,:],histogram_group[2,1,:,:])
R_TR_ghg = map_pearson(histogram_group[2,0,:,:],histogram_group[2,2,:,:])
R_TR_aer = map_pearson(histogram_group[2,0,:,:],histogram_group[2,3,:,:])


histogram_correlation = np.zeros(shape = (3,4))

histogram_correlation[0,1] = R_LH_nat[0]
histogram_correlation[0,2] = R_LH_ghg[0]
histogram_correlation[0,3] = R_LH_aer[0]
histogram_correlation[1,1] = R_ML_nat[0]
histogram_correlation[1,2] = R_ML_ghg[0]
histogram_correlation[1,3] = R_ML_aer[0] 
histogram_correlation[2,1] = R_TR_nat[0]
histogram_correlation[2,2] = R_TR_ghg[0]
histogram_correlation[2,3] = R_TR_aer[0]

histogram_group_3 = np.zeros(shape = (3,4,3,3))

for i in range(3):
    for j in range(4):
        for k in range(3):
            for m in range(3):
                if k == 0:
                    histogram_group_3[i,j,k,m] = np.sum(histogram_group[i,j,:3,m*2:m*2+2])
                else:
                    histogram_group_3[i,j,k,m] = np.sum(histogram_group[i,j,1+k*2:3+k*2,m*2:m*2+2])                   


address = '/Volumes/NO NAME 1/output/Figures'
x_title = 'cloud optical thickness (τ)'
y_title = 'cloud top pressure (hPa)'
title_height = 20
colorbar_name = 'Trend / Cloud fraction(%)'
textornot = 1
histogram_correlation.shape
savepicture = 1
t_data = histogram_group
name_group 
color_limit = 0.8 # 色标限制比例
name_group = np.array([['LH Cloud_cci','LH NAT','LH GHG','LH AER'],['ML Cloud_cci','ML NAT','ML GHG','ML AER'],['TR Cloud_cci','TR NAT','TR GHG','TR AER']])

x_label3 = ['0','3.6','23','100']
y_label3 = ['>800','680','440','0']
t_data3 = histogram_group_3

histogram_compact_trend_mapping(t_data3,x_label3,y_label3,x_title,y_title,title_height,colorbar_name,textornot,name_group,histogram_correlation,savepicture,address,color_limit)


# Figure S4

histogram_group = np.zeros(shape = (3,4,7,6))

histogram_group[0,0,:,:] = clisccp_LH_mean
histogram_group[0,1,:,:] = (clisccp_nat_LH_mean)[::-1]
histogram_group[0,2,:,:] = (clisccp_ghg_LH_mean)[::-1]
histogram_group[0,3,:,:] = (clisccp_aer_LH_mean)[::-1]

histogram_group[1,0,:,:] = clisccp_ML_mean
histogram_group[1,1,:,:] = (clisccp_nat_ML_mean)[::-1]
histogram_group[1,2,:,:] = (clisccp_ghg_ML_mean)[::-1]
histogram_group[1,3,:,:] = (clisccp_aer_ML_mean)[::-1]

histogram_group[2,0,:,:] = clisccp_TR_mean
histogram_group[2,1,:,:] = (clisccp_nat_TR_mean)[::-1]
histogram_group[2,2,:,:] = (clisccp_ghg_TR_mean)[::-1]
histogram_group[2,3,:,:] = (clisccp_aer_TR_mean)[::-1]

histogram_cloud = np.zeros(shape = (3,4))

histogram_cloud[0,0] = np.sum(histogram_group[0,0,:,:])
histogram_cloud[0,1] = np.sum(histogram_group[0,1,:,:])
histogram_cloud[0,2] = np.sum(histogram_group[0,2,:,:])
histogram_cloud[0,3] = np.sum(histogram_group[0,3,:,:])

histogram_cloud[1,0] = np.sum(histogram_group[1,0,:,:])
histogram_cloud[1,1] = np.sum(histogram_group[1,1,:,:])
histogram_cloud[1,2] = np.sum(histogram_group[1,2,:,:])
histogram_cloud[1,3] = np.sum(histogram_group[1,3,:,:])

histogram_cloud[2,0] = np.sum(histogram_group[2,0,:,:])
histogram_cloud[2,1] = np.sum(histogram_group[2,1,:,:])
histogram_cloud[2,2] = np.sum(histogram_group[2,2,:,:])
histogram_cloud[2,3] = np.sum(histogram_group[2,3,:,:])

address = '/Volumes/NO NAME 1/output/Figures'
x_label = ['0','0.6','3.6','9.4','23','60','100']
y_label = ['>800','800','680','560','440','310','180','0']
x_title = 'cloud optical thickness (τ)'
y_title = 'cloud top pressure (hPa)'
title_height = 20
colorbar_name = 'Cloud fraction(%)'
textornot = 1
histogram_cloud.shape
savepicture = 1
t_data = histogram_group
name_group = np.array([['LH Cloud_cci','LH NAT','LH GHG','LH AER'],['ML Cloud_cci','ML NAT','ML GHG','ML AER'],['TR Cloud_cci','TR NAT','TR GHG','TR AER']])

histogram_compact_mapping(t_data,x_label,y_label,x_title,y_title,title_height,colorbar_name,textornot,name_group,histogram_cloud,savepicture,address)

"""

"""
# Figure 3(b)/Figure 3(c)

# D&A data for R studio code

address = '/Volumes/NO NAME 1/output/xlsx_files'

mean_num = 3
ff = os.listdir(address)
ff.sort()
ff = remove_abnormal_file(ff)


for m in range(len(ff)):

    xlsx_name = ff[m]
        
    test_xlsx_v = read_xlrd(xlsx_name,address)

    if ff[m][:3] == 'ghg' or ff[m][:3] == 'aer':
        row_num = [10,5,4,10,3]
        test_xlsx = matrix_row_plus(test_xlsx_v,row_num)
    elif ff[m][:3] == 'nat':
        row_num = [10,1,5,4,10,3]
        test_xlsx = matrix_row_plus(test_xlsx_v,row_num)        
    else:
        test_xlsx = test_xlsx_v
        
    outdata = np.zeros(shape=test_xlsx.shape)
    
    if len(test_xlsx.shape) > 1:
        for i in range(test_xlsx.shape[0]):
            outdata[i,:] =  scaling_mean_1d(test_xlsx[i,:])
    else:
        outdata =  scaling_mean_1d(test_xlsx)
        
    if ff[m][:3] == 'ghg' or ff[m][:3] == 'aer' or ff[m][:3] == 'nat':
        outdata = np.mean(outdata,axis = 0)
        
    meandata = mean_list(outdata,mean_num)
            
    save_address = '/Volumes/NO NAME 1/output/DA_xlsx/'+ xlsx_name[:-5] +'_scale_mean.xlsx'
    if len(meandata.shape) > 1:
        creat_xlsx_2d(meandata,save_address = save_address)  
    else:
        creat_xlsx_1d(meandata,save_address = save_address)       

#------------------------------------------------------------------------------

# Scaling factors 图像绘制

row_col = [1,1]
subtitle = ''
var_name = np.array(['NAT','AER','GHG'])
xticks1 = np.array(['AF','AO','AS','EU','NA','SA'])
xticks2 = np.array(['ML','TR','LH','ML+TR','6-region'])
color_list = ['b','g','r']
savepictures = 1

address = '/Volumes/NO NAME 1/output/ols_tls'
ff = os.listdir(address)
ff.sort()
ff = remove_abnormal_file(ff)

savename = 'Figure 3(b)'

point_value = np.zeros(shape=(3,6))
dy = np.zeros(shape=(3,6))

var_list = np.array(['nat','aer','ghg'])
area_list = np.array(['AF','AO','AS','EU','NA','SA'])

for i in range(len(var_list)):
    for j in range(len(area_list)):
        filename = var_list[i] + '_' + area_list[j] + '_scale_mean_tls.csv'                 
        test_m = read_csv(address,filename)
        point_value[i,j] = float(test_m[-1,3])
        dy[i,j] = float(test_m[-1,4])-float(test_m[-1,3])   

scaling_factors_compare(point_value,dy,row_col,subtitle,var_name,xticks1,color_list,savepictures,savename)

savename = 'Figure 3(c)'

point_value = np.zeros(shape=(3,5))
dy = np.zeros(shape=(3,5))

var_list = np.array(['nat','aer','ghg'])
area_list = np.array(['ML','TR','LH','ML+TR','NA-AO'])

for i in range(len(var_list)):
    for j in range(len(area_list)):
        filename = var_list[i] + '_' + area_list[j] + '_scale_mean_tls.csv'  
        if j == len(area_list)-1:
            filename = var_list[i] + '_' + area_list[j] + '_scale_mean_ols.csv'
        test_m = read_csv(address,filename)
        point_value[i,j] = float(test_m[-1,3])
        dy[i,j] = float(test_m[-1,4])-float(test_m[-1,3])   

scaling_factors_compare(point_value,dy,row_col,subtitle,var_name,xticks2,color_list,savepictures,savename)

"""

def cci_clisccp_mean_3d(data,SWpoint,NEpoint,mean_num,valid_delete):#valid_delete 
    
    out_list = np.zeros(shape=(int(data.shape[0]/mean_num),data.shape[3],data.shape[4]))
    
    M = data.shape[1]*180/(NEpoint[0]-SWpoint[0])
    SWsys = (SWpoint[0]/180)*M
    
    for i in range(data.shape[0]):
        dif_area = np.zeros(shape=(data.shape[3],data.shape[4]))
        cul_area = 0
        for j in range(data.shape[1]):
            scale = 1*math.sin(((j+ SWsys)/M)*math.pi)
            for k in range(data.shape[2]):
                if valid_delete:
                    if np.mean(data[i,j,k,:,:]) > 0:
                        dif_area[:,:] += scale*data[i,j,k,:,:]
                        cul_area += scale
                else:
                    dif_area[:,:] += scale*data[i,j,k,:,:]
                    cul_area += scale
                
        out_list[i,:,:] =  dif_area[:,:]/cul_area
        
    return out_list



#============================================================================================
def pro_mean_3d(data,SWpoint,NEpoint,mean_num):#
    
    out_list = np.zeros(shape=(int(data.shape[0]/mean_num)))
    ori_list = np.zeros(shape=(data.shape[0]))
    
    M = data.shape[1]*180/(NEpoint[0]-SWpoint[0])
    SWsys = (SWpoint[0]/180)*M
    
    for i in range(data.shape[0]):
        dif_area = 0
        cul_area = 0
        for j in range(data.shape[1]):
            scale = 1*math.sin(((j+ SWsys)/M)*math.pi)
            for k in range(data.shape[2]):
                if data[i,j,k] > 0:
                    dif_area += scale*data[i,j,k]
                    cul_area += scale                    
                
        ori_list[i] =  dif_area/cul_area
        
    for i in range(int(data.shape[0]/mean_num)):
        for j in range(mean_num):
            out_list[i] += ori_list[int(mean_num*i+j)]
        out_list[i] = out_list[i]/mean_num
    
    return out_list

def extract_orig_map_3d(origdata,SWpoint,NEpoint):#
    
    clip_data = np.zeros(shape=(origdata.shape[0],int(origdata.shape[1]),int(origdata.shape[2]))) #缩小裁剪范围
    
    data = valid_map(origdata)
    
    lats = NEpoint[0] - SWpoint[0]
    lons = NEpoint[1] - SWpoint[1]
    
    cutdata = np.zeros(shape = (origdata.shape[0],int((lats/180)*data.shape[1]),int((lons/360)*data.shape[2])))
    
    for m in range(data.shape[0]):        
        for i in range(int((lats/180)*data.shape[1])):
            for j in range(int((lons/360)*data.shape[2])):
                cutdata[m,i,j] = data[m,i+int((SWpoint[0]/180)*data.shape[1]),j + int((SWpoint[1]/360)*data.shape[2])]
    
    clip_data[:,:cutdata.shape[1],:cutdata.shape[2]] = cutdata[:,:,:]
    
    return clip_data

def valid_map(data):
    
    valid_data = np.zeros(shape=(data.shape[0],len(np.nonzero(data[0,:,1])[0]),len(np.nonzero(data[0,1,:])[0])))
    
    valid_data[:,:,:] =  data[:,:len(np.nonzero(data[0,:,1])[0]),:len(np.nonzero(data[0,1,:])[0])]
    
    return valid_data

def paramove_3d_valid_map(x): #
    
    xp = np.zeros(shape=(x.shape[0],x.shape[1],x.shape[2]))
    valid_x = valid_map(x)
    valid_xp = np.zeros(shape=(valid_x.shape[0],valid_x.shape[1],valid_x.shape[2]))
    
    for i in range(x.shape[0]):
        valid_xp[i,:,:int(valid_x.shape[2]/2)] = valid_x[i,:,int(valid_x.shape[2]/2):]
        valid_xp[i,:,int(valid_x.shape[2]/2):] = valid_x[i,:,:int(valid_x.shape[2]/2)]

    xp[:,:valid_x.shape[1],:valid_x.shape[2]] = valid_xp[:,:,:]       

    return xp


def creat_xls_3p(*data,save_address):
    
    #------------------------------
    #3p - 3 part - shape.[0] or [0]
    #------------------------------
    #
    #ex:
    #creat_xls_3p(aer_value[1],clt_aer_dataname,clt_aer_modelnum,save_address = save_address)

    if len(data) > 1:
        data1 = data[0]
        data2 = data[1]
        data3 = data[2]
    else:
        data1 = data.shape[0]
        data2 = data.shape[1]
        data3 = data.shape[2]        
    
    book = xlwt.Workbook(encoding='utf-8',style_compression=0)
    sheetname1 = book.add_sheet('data',cell_overwrite_ok=True)
    sheetname2 = book.add_sheet('data_num',cell_overwrite_ok=True)
    sheetname3 = book.add_sheet('model_num',cell_overwrite_ok=True)
    for i in range(data1.shape[0]):
        for j in range(data1.shape[1]):
            sheetname1.write(i,j,data1[i,j])
    for i in range(len(data2)):
        for j in range(len(data2[i])):
            sheetname2.write(i,j,data2[i][j])
    for i in range(len(data3)):
        sheetname3.write(0,i,data3[i])
            
    book.save(save_address)

def mapping(data):
    
    lons_csm = data.shape[1]
    lats_csm = data.shape[0]    
        
    xx= np.arange(-180 + (360/lons_csm)/2, 180 + (360/lons_csm)/2, 360.0/lons_csm)
    yy= np.arange(-90 + (180/lats_csm)/2, 90 + (180/lats_csm)/2, 180.0/lats_csm)

    lons_csm = np.ma.masked_array(xx)
    lats_csm = np.ma.masked_array(yy)

    lon_0 = lons_csm.mean() # list -> array -> .mean()
    lat_0 = lats_csm.mean()
    m = Basemap(lat_0=lat_0, lon_0=lon_0)
    lon, lat = np.meshgrid(lons_csm, lats_csm)
    xi_csm, yi_csm = m(lon, lat)

    plt.figure(figsize=(12, 7))
    #clts_0 = clts[0:1:, ::, ::]
    cs = m.pcolor(xi_csm, yi_csm, data, cmap=plt.cm.coolwarm)
    
    # Add Grid Lines
    m.drawparallels(np.arange(-90., 91., 20.), labels=[1,0,0,0], fontsize=10)
    m.drawmeridians(np.arange(-180., 181., 40.), labels=[0,0,0,1], fontsize=10)
    
    # Add Coastlines, States, and Country Boundaries
    m.drawcoastlines()
    #m.drawstates()
    #m.drawcountries()
    # Add Colorbar
    cbar = m.colorbar(cs, location='bottom', pad="10%")
    cbar.set_label('test',fontproperties='SimHei',fontsize = 18)
    
    # Add Title
    plt.title('test',fontproperties='SimHei',fontsize = 24)
    
    """
    N=np.array([0.5,1])
    
    #plt.contour(xi_csm, yi_csm,sst_time[0,::,::],N,linewidth=2,colors = 'black')#画出等高线图，cmap表示颜色的图层。
    xx = np.arange(1,360,10)
    
    m.scatter(xx, yy,color = 'black')
    """
    plt.show()

def creat_xlsx_3p(*data,save_address):
    
    #------------------------------
    #3p - 3 part - shape.[0] or [0]
    #------------------------------
    #
    #ex:
    #creat_xls_3p(aer_value[1],clt_aer_dataname,clt_aer_modelnum,save_address = save_address)

    if len(data) > 1:
        data1 = data[0]
        data2 = data[1]
        data3 = data[2]
    else:
        data1 = data.shape[0]
        data2 = data.shape[1]
        data3 = data.shape[2]        
    
    print(save_address)
    
    book = xlsxwriter.Workbook(save_address)
    sheetname1 = book.add_worksheet('data')
    sheetname2 = book.add_worksheet('data_num')
    sheetname3 = book.add_worksheet('model_num')
    for i in range(data1.shape[0]):
        for j in range(data1.shape[1]):
            sheetname1.write(i,j,data1[i,j])
    for i in range(len(data2)):
        for j in range(len(data2[i])):
            sheetname2.write(i,j,data2[i][j])
    for i in range(len(data3)):
        sheetname3.write(0,i,data3[i])
            
    book.close()



def creat_xlsx_2d(data,save_address):
    
    # save_address
    
    print(save_address)
    
    book = xlsxwriter.Workbook(save_address)
    sheetname1 = book.add_worksheet('data')
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            sheetname1.write(i,j,data[i,j])
            
    book.close()

def creat_xlsx_1d(data,save_address):
    
    # save_address 
    
    print(save_address)
    
    book = xlsxwriter.Workbook(save_address)
    sheetname1 = book.add_worksheet('data')
    for i in range(data.shape[0]):
        sheetname1.write(0,i,data[i])
            
    book.close()
    


#==============================================================================
#==============================================================================
# additional function

def piControl_read_set(address,monthoryear,full_period,time_start,time_end,interpolateornot):

    #------------------------------
    # * 
    #variables:
    #count
    #class_num
    #data_fullname_set
    #class_list
    
    #------------------------------
    #open and sort
    ff = os.listdir(address)
    ff.sort()
    os.chdir(address)
    print('location : ')
    os.getcwd()
    #------------------------------
    #valid value
    valid_ff = len(ff)*['']    
    count = 0
    for i in range(len(ff)):
        if ff[i][:3] == 'clt':
            valid_ff[count] = ff[i]
            count += 1
        else:
            count = count
    #------------------------------    
    #classify
    class_num = 0
    same_or_not = ['']
    for i in range(count):
        if valid_ff[i][:-16] != same_or_not:
            class_num += 1
        else:
            class_num = class_num
        same_or_not = valid_ff[i][:-16]

    #------------------------------    
    #classified_num
    ens_num = class_num*['']
    th = -1
    same_or_not = ['']
    for i in range(count):
        if valid_ff[i][:-16] != same_or_not:
            th += 1
            ens_num[th] += valid_ff[i]
        else:
            ens_num[th] += ' ' + valid_ff[i]
        same_or_not = valid_ff[i][:-16]
    #------------------------------    
    #assign
    period_set = class_num*['']
    data_fullname_set = class_num*['']
    year_period_set = class_num*[''] #
    startyear = class_num*['']  
    
    th = -1
    same_or_not = [''] 
    for i in range(class_num):
        data_fullname_set[i] = ens_num[i].split()
        for j in range(len(data_fullname_set[i])):
            period_set[i] += ' ' + data_fullname_set[i][j][-16:-3]      
            year_period_set[i] += ' ' + str(int(data_fullname_set[i][j][-9:-5]) - int(data_fullname_set[i][j][-16:-12]) + 1)
            startyear[i] = data_fullname_set[i][0][-16:-12]
        period_set[i] = period_set[i].split()
        year_period_set[i] = year_period_set[i].split()        
    #------------------------------    
    #start and end
    start_end_set = class_num*['']
    front_name_set = class_num*[''] 
    
    for i in range(class_num):
        start_end_set[i] = period_set[i][0][:6]+'-'+ period_set[i][-1][-6:]
        front_name_set[i] = data_fullname_set[i][0][:-16]
        
    #------------------------------    
    #classified_class and filename
    class_list = np.zeros(shape=(class_num))
    #------------------------------    
    #final set    
    data_set = np.zeros(shape=(500,33,360,720))   # origdata 
    ems_front = 0
    ems_back = 0
    for i in range(class_num):
        #///////////////////////////////
        #time begin   
        begin=datetime.datetime.now()
        print(begin)
        
        pi_model,begin_point = piControl(address,front_name_set[i],start_end_set[i],year_period_set[i],full_period)
        #///////////////////////////////
        print(datetime.datetime.now(),front_name_set[i],' piControl() done!')
        
        ems_back += pi_model.shape[0]
        class_list[i] = pi_model.shape[0]
        pi_model_y = scandal_piControl(pi_model,int(startyear[i])+begin_point,monthoryear,interpolateornot)
        data_set[ems_front:ems_back,:,:,:] = pi_model_y
        ems_front += pi_model.shape[0]
        
        #///////////////////////////////
        #time end 
        end=datetime.datetime.now()
        dur=end-begin
        print(end,front_name_set[i],' scandal_piControl() done!')
        print(dur,front_name_set[i],' done!')

        
    return data_set,data_fullname_set,class_list



def piControl(address,front_name,start_end,year_period,full_period):
       
    period_num = len(year_period)
    full_list = int(start_end[7:11])-int(start_end[0:4])+1    
    filename = len(year_period)*['']
    #------------------------------
    #open and sort
    ff = os.listdir(address)
    ff.sort()
    #------------------------------
    #valid value
    count = 0
    for i in range(len(ff)):
        if ff[i][:-16] == front_name:
            filename[count] = ff[i]
            count += 1
        else:
            count = count
    fh = nc.Dataset(filename[0],'r')
    lat = len(fh.variables['lat'][:])
    lon = len(fh.variables['lon'][:])
    fh.close()    
    fulldata = np.zeros(shape=(full_list*12,lat,lon))
    begin_point = 0
    
    data_var = ['']# 
    for i in range(len(filename[0])):
        if i == 0:
            data_var = data_var[0] + filename[0][0]
        else:
            if filename[0][i] != '_':
                data_var = data_var[:i] + filename[0][i]
            else:
                break  
            
    for i in range(32):
        if abs(int(start_end[:4]) + i -1982)%33 == 0:
            begin_point = i
        else:
            continue

    begin_list = 0
    begin_year = 0
    sum_year = 0
    for i in range(32):
        sum_year += int(year_period[i])
        if begin_point < sum_year:
            begin_year = begin_point - (sum_year-int(year_period[i]))
            break
        else:
            begin_list += 1

    front_num = 0

    for i in range(period_num):  
        
        if i == 0:
            fh = nc.Dataset(filename[i+begin_list],'r')#
            fulldata[front_num:front_num + int(year_period[i+begin_list])*12 - begin_year*12,:,:] = fh.variables['clt'][begin_year*12:,:,:]            
            front_num += int(year_period[i+begin_list])*12 - begin_year*12
            fh.close
        else:
            try:        
                filepath = filename[i+begin_list]
                fh = nc.Dataset(filepath,'r')#
                fulldata[front_num:int(front_num + int(year_period[i+begin_list])*12),:,:] = fh.variables['clt'][:,:,:]
                front_num += int(year_period[i+begin_list])*12
                fh.close
            except:
                print("Error: data missing")
                
    fulldata2 = np.zeros(shape=(int((full_list-begin_point)/full_period),396,lat,lon))
    
    for i in range(int((full_list-begin_point)/full_period)):        
        
        fulldata2[i,:,:,:] = fulldata[i*396:i*396 + 396,:,:]
    
    return fulldata2,begin_point





def scandal_piControl(data,startyear,monthoryear,interpolateornot):
    
    if monthoryear == 'year':
    
        out_data = np.zeros(shape=(data.shape[0],33,360,720))
        
    else:
        
        out_data = np.zeros(shape=(data.shape[0],396,360,720))
    
    for f in range(data.shape[0]):
        
        cfc = np.zeros(shape=(data.shape[1],data.shape[2],data.shape[3]))
        cfc[:,:,:] = data[f,:,:,:]
        '''
        for i in range(data.shape[1]):
            for j in range(data.shape[2]):
                for k in range(data.shape[3]):                
                    if data[f,i,j,k] < 100 and data[f,i,j,k] > 0:
                        cfc[i,j,k] = data[f,i,j,k]
                    else:
                        cfc[i,j,k] = 0.0     
        '''
        if monthoryear == 'year':
            
            cfc_y = np.zeros(shape=(33,cfc.shape[1],cfc.shape[2])) 
            for i in range(33):
                for j in range(cfc.shape[1]):
                    for k in range(cfc.shape[2]):
                        if float(startyear+i)%4 == 0:
                            cfc_y[i,j,k] = float(cfc[12*i,j,k]*31+cfc[12*i+1,j,k]*29+cfc[12*i+2,j,k]*31+cfc[12*i+3,j,k]*30+cfc[12*i+4,j,k]*31+cfc[12*i+5,j,k]*30+cfc[12*i+6,j,k]*31+cfc[12*i+7,j,k]*31+cfc[12*i+8,j,k]*30+cfc[12*i+9,j,k]*31+cfc[12*i+10,j,k]*30+cfc[12*i+11,j,k]*31)
                            if cfc_y[i,j,k] != 0:
                                cfc_y[i,j,k] = cfc_y[i,j,k]/366
                        else:                 
                            cfc_y[i,j,k] = float(cfc[12*i,j,k]*31+cfc[12*i+1,j,k]*28+cfc[12*i+2,j,k]*31+cfc[12*i+3,j,k]*30+cfc[12*i+4,j,k]*31+cfc[12*i+5,j,k]*30+cfc[12*i+6,j,k]*31+cfc[12*i+7,j,k]*31+cfc[12*i+8,j,k]*30+cfc[12*i+9,j,k]*31+cfc[12*i+10,j,k]*30+cfc[12*i+11,j,k]*31)
                            if cfc_y[i,j,k] != 0:
                                cfc_y[i,j,k] = cfc_y[i,j,k]/365
    
            if interpolateornot != 0:
            
                for i in range(33):
                    if cfc_y[i,0,0]!= 0:
                        out_data[f,i,:data.shape[2],:data.shape[3]] = griddata_interpolate(cfc_y[i,:,:],'linear',['72','144'])
            else:
                
                out_data[f,:,:data.shape[2],:data.shape[3]] = cfc_y[:,:,:]  
                
        if monthoryear == 'month': 
    
            if interpolateornot != 0:
            
                for i in range(396):
                    if cfc[i,0,0]!= 0:
                        out_data[f,i,:data.shape[2],:data.shape[3]] = griddata_interpolate(cfc[i,:,:],'linear',['72','144'])
                    
            else:
                out_data[f,:,:data.shape[2],:data.shape[3]] = cfc[:,:,:] 
                
    return out_data



def griddata_interpolate(data,method,scale):

    rows = data.shape[0]
    cols = data.shape[1]
    
    points = np.zeros(shape=(rows*cols,2))
    values = np.zeros(shape=rows*cols)
    
    for i in range(rows):
        for j in range(cols):
            values[i*cols+j] = data[i,j]
            points[i*cols+j,0] = i
            points[i*cols+j,1] = j

    if float(data.shape[0])/float(data.shape[1]) > 0.2:
    
        row_step = complex(0,scale[0]+1)
        col_step = complex(0,scale[1]+2)
        
        grid_x, grid_y = np.mgrid[1:rows:row_step, 1:cols:col_step]
        
        if method == 'nearest':
            grid_z = griddata(points, values, (grid_x, grid_y), method='nearest')

        if method == 'linear':
            grid_z = griddata(points, values, (grid_x, grid_y), method='linear')
        if method == 'cubic':
            grid_z = griddata(points, values, (grid_x, grid_y), method='cubic')
            
        grid_z = grid_z[:-1,:-2]
        
    else:
        
        row_step = complex(0,scale[0]+1)
        col_step = complex(0,scale[1]+1)
        
        grid_x, grid_y = np.mgrid[1:rows:row_step, 1:cols:col_step]
        
        if method == 'nearest':
            grid_z = griddata(points, values, (grid_x, grid_y), method='nearest')
        if method == 'linear':
            grid_z = griddata(points, values, (grid_x, grid_y), method='linear')
        if method == 'cubic':
            grid_z = griddata(points, values, (grid_x, grid_y), method='cubic')
            
        grid_z = grid_z[:-1,:-1]
        
    return grid_z

def basic_mask_map(allmaporpart,tiff_data,target_data,SWpoint,NEpoint,depictornot):

    tiff_data_p = check_data_slice(tiff_data,SWpoint,NEpoint,depictornot)
    
    if allmaporpart == 'allmap':
        target_data_p = check_data_slice(target_data,SWpoint,NEpoint,depictornot) 
    else:
        target_data_p = target_data
    
    if target_data.shape != tiff_data.shape:
        scale = np.array(target_data.shape)
        int_tiff_data_p = griddata_interpolate(tiff_data_p,'linear',scale)
        masked_part = np.logical_not(int_tiff_data_p<0.95)
    else:
        masked_part = np.logical_not(tiff_data_p<0.95)
        
    masked_result = np.zeros(shape = target_data_p.shape)
    
    for i in range(target_data_p.shape[0]):
        for j in range(target_data_p.shape[1]):
            if masked_part[i,j]:
                masked_result[i,j] = 0
            else:
                masked_result[i,j] = target_data_p[i,j]
                
    #extract_data_slice(masked_result,SWpoint,NEpoint,depictornot)
    
    return masked_result


def check_data_slice(data,SWpoint,NEpoint,depictornot):
    
    lats = NEpoint[0] - SWpoint[0]
    lons = NEpoint[1] - SWpoint[1]
    
    cutdata = np.zeros(shape = (int((lats/180)*data.shape[0]),int((lons/360)*data.shape[1])))
    
    for i in range(int((lats/180)*data.shape[0])):
        for j in range(int((lons/360)*data.shape[1])):
            cutdata[i,j] = data[i+int((SWpoint[0]/180)*data.shape[0]),j + int((SWpoint[1]/360)*data.shape[1])]

    if depictornot == 1:

        lons_csm = data.shape[1]
        lats_csm = data.shape[0]
        
        xx= np.arange(-180 + SWpoint[1] + (360/lons_csm)/2, -180 + NEpoint[1] + (360/lons_csm)/2, 360/lons_csm)
        yy= np.arange(-90 + SWpoint[0] + (180/lats_csm)/2, -90 + NEpoint[0] + (180/lats_csm)/2, 180/lats_csm)
        
        lons_csm = np.ma.masked_array(xx)
        lats_csm = np.ma.masked_array(yy)
        """
        lon_0 = lons_csm.mean() # list -> array -> .mean()
        lat_0 = lats_csm.mean()
        """
        m = Basemap(llcrnrlon=-180 +SWpoint[1],llcrnrlat = -90 +SWpoint[0], urcrnrlon = -180 +NEpoint[1],urcrnrlat = -90 +NEpoint[0])
        """
        Warning:ZeroDivisionError: float division by zero
        """
        lon, lat = np.meshgrid(lons_csm, lats_csm)
        xi_csm, yi_csm = m(lon, lat)
    
        plt.figure(figsize=(12, 7))
        #clts_0 = clts[0:1:, ::, ::]
        cs = m.pcolor(xi_csm, yi_csm, cutdata, cmap=plt.cm.coolwarm)
        
        # Add Grid Lines
        m.drawparallels(np.arange(float(-90+SWpoint[0]),float(-90+NEpoint[0]+1),30), labels=[1,0,0,0], fontsize=10)
        m.drawmeridians(np.arange(float(-180+SWpoint[1]),float(-180+NEpoint[1]+1),30), labels=[0,0,0,1], fontsize=10)
        """
        m.drawparallels(np.arange(-90 ,90 ,30), labels=[1,0,0,0], fontsize=10)
        m.drawmeridians(np.arange(-180 ,180 ,40), labels=[0,0,0,1], fontsize=10)
        """
        # Add Coastlines, States, and Country Boundaries
        m.drawcoastlines()
        #m.drawstates()
        #m.drawcountries()
        # Add Colorbar
        cbar = m.colorbar(cs, location='bottom', pad="10%")
        cbar.set_label('test',fontproperties='SimHei',fontsize = 18)
        
        # Add Title
        plt.title('test',fontproperties='SimHei',fontsize = 24)
        
        """
        N=np.array([0.5,1])
        
        #plt.contour(xi_csm, yi_csm,sst_time[0,::,::],N,linewidth=2,colors = 'black')
        xx = np.arange(1,360,10)
        
        m.scatter(xx, yy,color = 'black')
        """
        plt.show()
    
    return cutdata
        

def extract_data_slice(cutdata,SWpoint,NEpoint,depictornot):
    
    data_shape0 = cutdata.shape[0]*180/(NEpoint[0]-SWpoint[0])
    data_shape1 = cutdata.shape[1]*360/(NEpoint[1]-SWpoint[1])

    if depictornot == 1:

        lons_csm = data_shape1
        lats_csm = data_shape0
        
        xx= np.arange(-180 + SWpoint[1] + (360/lons_csm)/2, -180 + NEpoint[1] + (360/lons_csm)/2, 360/lons_csm)
        yy= np.arange(-90 + SWpoint[0] + (180/lats_csm)/2, -90 + NEpoint[0] + (180/lats_csm)/2, 180/lats_csm)
        
        lons_csm = np.ma.masked_array(xx)
        lats_csm = np.ma.masked_array(yy)
        """

        lon_0 = lons_csm.mean() # list -> array -> .mean()
        lat_0 = lats_csm.mean()
        """
        m = Basemap(llcrnrlon=-180 +SWpoint[1],llcrnrlat = -90 +SWpoint[0], urcrnrlon = -180 +NEpoint[1],urcrnrlat = -90 +NEpoint[0])
        """
        Warning:ZeroDivisionError: float division by zero
        """
        lon, lat = np.meshgrid(lons_csm, lats_csm)
        xi_csm, yi_csm = m(lon, lat)
    
        plt.figure(figsize=(12, 7))
        #clts_0 = clts[0:1:, ::, ::]
        cs = m.pcolor(xi_csm, yi_csm, cutdata, cmap=plt.cm.coolwarm)
        
        # Add Grid Lines
        # 绘制经纬线
        m.drawparallels(np.arange(float(-90+SWpoint[0]),float(-90+NEpoint[0]+1),30), labels=[1,0,0,0], fontsize=10)
        m.drawmeridians(np.arange(float(-180+SWpoint[1]),float(-180+NEpoint[1]+1),30), labels=[0,0,0,1], fontsize=10)
        """
        m.drawparallels(np.arange(-90 ,90 ,30), labels=[1,0,0,0], fontsize=10)
        m.drawmeridians(np.arange(-180 ,180 ,40), labels=[0,0,0,1], fontsize=10)
        """
        # Add Coastlines, States, and Country Boundaries
        m.drawcoastlines()
        #m.drawstates()
        #m.drawcountries()
        # Add Colorbar
        cbar = m.colorbar(cs, location='bottom', pad="10%")
        cbar.set_label('test',fontproperties='SimHei',fontsize = 18)
        
        # Add Title
        plt.title('test',fontproperties='SimHei',fontsize = 24)
        
        """
        N=np.array([0.5,1])
        
        #plt.contour(xi_csm, yi_csm,sst_time[0,::,::],N,linewidth=2,colors = 'black')
        xx = np.arange(1,360,10)
        
        m.scatter(xx, yy,color = 'black')
        """
        plt.show()
        
        

        
def read_csv(address,csv_name):

    os.chdir(address)
    
    csv_file=csv.reader(open(csv_name,'r'))
    print(csv_file) 
    
    tt = 0
    
    for line in csv_file:
        if tt == 0:
            matrix_content = line

            tt += 1
        else:
            matrix_content = np.row_stack((matrix_content,line))
        #content.append(line)
        
    return matrix_content


def scaling_factors(point_value,dy,row_col,boundary,number,subtitle,saveornot,address):

    plt.figure(figsize=(14, 7))
    
    for i in range(point_value.shape[0]):
    
        map_location = int(str(row_col[0]) + str(row_col[1]) + str(i+1))
        
        plt.subplot(map_location) 
        
        plt.style.use('bmh')
        
        x = np.linspace(boundary[0],boundary[1],number)
         
        plt.errorbar(x,point_value[i] ,yerr=dy[i],fmt='o',ecolor='r',color='b',elinewidth=2,capsize=4)
        plt.axhline(0,color = 'black',linestyle = '-',linewidth = 1.5)
        plt.axhline(1,color = 'black',linestyle = '--',linewidth = 1)
        plt.title(subtitle,fontproperties='SimHei',fontsize = 24)
        plt.ylabel('scaling factors',fontsize=18) 
    
    plt.suptitle('',fontproperties='SimHei',fontsize = 28)
    if saveornot:
        os.chdir(address)
        plt.savefig(subtitle + '  scaling_factors')
    
    
def residual_consistency_test(address,point_value,down_line,up_line,row_col,boundary,number,subtitle,label_name,saveornot):

    plt.figure(figsize=(14, 7))
    
    for i in range(point_value.shape[0]):
    
        map_location = int(str(row_col[0]) + str(row_col[1]) + str(i+1))
            
        plt.subplot(map_location) 
            
        plt.style.use('bmh')
            
        x = np.linspace(boundary[0],boundary[1],number)
       
        #plt.ylim(1e-3, 8e0)
        plt.semilogy()

        plt.plot(x,up_line[i],color = 'b',label = label_name[0],linestyle = '--')
        plt.plot(x,down_line[i],color = 'g',label = label_name[1],linestyle = '--')
        plt.scatter(x,point_value[i], color = 'black',marker ='o',s=50)
        plt.title(subtitle,fontproperties='SimHei',fontsize = 24)
        plt.ylabel('scaling factors',fontsize=18) 
        plt.yticks(fontsize = 20)
        plt.xticks(fontsize = 20)
        plt.legend(prop={'size':18}, shadow=True) 
        
    plt.suptitle('',fontproperties='SimHei',fontsize = 28)
    if saveornot:
        os.chdir(address)
        plt.savefig(subtitle + '  residual_consistency_test')    
    
    

#============================================================================================
#============================================================================================


def cmip_clisccp_mean_3d(data,SWpoint,NEpoint,mean_num,valid_delete):#valid_delete 是否考虑零值
    
    out_list = np.zeros(shape=(int(data.shape[0]/mean_num),data.shape[1],data.shape[2]))
    
    row_range = int(data.shape[3]*(NEpoint[0]-SWpoint[0])/180)
    col_range = int(data.shape[4]*(NEpoint[1]-SWpoint[1])/360)
    row_start = int(data.shape[3]*SWpoint[0]/180)
    col_start = int(data.shape[4]*SWpoint[1]/360)
    
    for i in range(data.shape[0]):
        dif_area = np.zeros(shape=(data.shape[1],data.shape[2]))
        cul_area = 0
        for j in range(row_range):
            scale = 1*math.sin(((j+ row_start)/data.shape[3])*math.pi)
            for k in range(col_range):
                if valid_delete:
                    if np.mean(data[i,:,:,j+row_start,k+col_start]) > 0:
                        dif_area[:,:] += scale*data[i,:,:,j+row_start,k+col_start]
                        cul_area += scale
                else:
                    dif_area[:,:] += scale*data[i,:,:,j+row_start,k+col_start]
                    cul_area += scale
                
        out_list[i,:,:] =  dif_area[:,:]/cul_area
        
    return out_list

def copy_mask_map(copy_data,target_data):
    
    cut_data = np.zeros(shape=target_data.shape)
    
    for i in range(copy_data.shape[0]):
        for j in range(copy_data.shape[1]):
            cut_data[i,j] = (target_data[i,j] if copy_data[i,j] > 0 else 0) #一句型语法一定要else
            
    return cut_data


def read_xlrd(excelFile,address):
    
    os.chdir(address)
    data = xlrd.open_workbook(excelFile)
    table = data.sheet_by_index(0)

    for rowNum in range(table.nrows):
        rowVale = table.row_values(rowNum)
        if rowNum == 0:
            list_total = rowVale
        else:
            list_total = np.row_stack((list_total,rowVale))
    """
        for colNum in range(table.ncols):
            if rowNum > 0 and colNum == 0:
                print(int(rowVale[0]))
            else:
                print(rowVale[colNum])
        print("---------------")
    """
    list_total = np.array(list_total)
    return list_total

def scaling_mean_1d(data):
    
    data_max = np.max(data)
    data_min = np.min(data)
    data_mean = np.mean(data)
    
    out_data = np.zeros(shape = data.shape)
    
    for i in range(len(out_data)):
        out_data[i] = (data[i] - data_mean)/(data_max-data_min)
        
    return out_data

def scaling_1d(data):#
    
    data_max = np.max(data)
    data_min = np.min(data)
    
    out_data = np.zeros(shape = data.shape)
    
    for i in range(len(out_data)):
        out_data[i] = (data[i] - data_min)/(data_max-data_min)
        
    return out_data

def normalize_1d(data):
    
    data_var = np.var(data)
    data_mean = np.mean(data)
    
    out_data = np.zeros(shape = data.shape)

    for i in range(len(out_data)):
        out_data[i] = (data[i] - data_mean)/data_var
        
    return out_data

     
    

def mean_list(data,mean_num):
    num = 0
    if len(data.shape) > 1:
        outdata = np.zeros(shape=(data.shape[0],int(data.shape[1]/mean_num)))
        for i in range(data.shape[1]):
            if i%mean_num == 0 and i != 0:
                num += 1
                outdata[:,num] += data[:,i]
            else:
                outdata[:,num] += data[:,i]
        for i in range(int(data.shape[1]/mean_num)): 
            outdata[:,i] = outdata[:,i]/mean_num
    else:
        outdata = np.zeros(shape=(int(data.shape[0]/mean_num)))
        for i in range(data.shape[0]):
            if i%mean_num == 0 and i != 0:
                num += 1
                outdata[num] += data[i]
            else:
                outdata[num] += data[i]
        for i in range(int(data.shape[0]/mean_num)): 
            outdata[i] = outdata[i]/mean_num 
            
    return outdata

def remove_abnormal_file(ff):
    
    out_ff = []
    for i in range(len(ff)):
        if '.' in ff[i] and ff[i][0] != '.':
            out_ff.append(ff[i])
            
    return out_ff   
 

    
def matrix_row_plus(data,row_num):
    
    out_data = np.zeros(shape=(len(row_num),data.shape[1]))
    i_start = 0
    
    for i in range(len(row_num)):
        out_data[i,:] = np.mean(data[int(i_start):int(i_start+row_num[i]),:],axis=0)
        i_start += row_num[i]
    
    return out_data
    
def extract_obs_clisccp(area,SA_row_range,SA_col_range):
    
    row_num = SA_row_range[1] - SA_row_range[0]
    col_num = SA_col_range[1] - SA_col_range[0]
    os.chdir('/Volumes/NO NAME/Work/npy文件')
    data_a=np.load('cci_clisccp_annual1.npy')    
    clisccp_SA_1 = large_to_small_his(data_a,SA_row_range,SA_col_range)
    del(data_a)
    data_a=np.load('cci_clisccp_annual2.npy')    
    clisccp_SA_2 = large_to_small_his(data_a,SA_row_range,SA_col_range)
    del(data_a)
    data_a=np.load('cci_clisccp_annual3.npy')    
    clisccp_SA_3 = large_to_small_his(data_a,SA_row_range,SA_col_range)
    del(data_a)
    data_a=np.load('cci_clisccp_annual4.npy')    
    clisccp_SA_4 = large_to_small_his(data_a,SA_row_range,SA_col_range)
    del(data_a)
    data_a=np.load('cci_clisccp_annual5.npy')    
    clisccp_SA_5 = large_to_small_his(data_a,SA_row_range,SA_col_range)
    del(data_a)
    clisccp_SA = np.zeros(shape=(33,row_num,col_num,7,6))
    clisccp_SA[:8,:,:,:,:] = clisccp_SA_1[:,:,:,:,:]
    clisccp_SA[8:13,:,:,:,:] = clisccp_SA_2[:,:,:,:,:]
    clisccp_SA[13:22,:,:,:,:] = clisccp_SA_3[:,:,:,:,:]
    clisccp_SA[22:31,:,:,:,:] = clisccp_SA_4[:,:,:,:,:]
    clisccp_SA[31:33,:,:,:,:] = clisccp_SA_5[:,:,:,:,:]    
    np.save('clisccp_' + area + '.npy',clisccp_SA)
    
def large_to_small_his(clisccp_data,row_range,col_range):

    row_num = row_range[-1] - row_range[0]
    col_num = col_range[-1] - col_range[0]
        
    hist_stand = np.zeros(shape = (clisccp_data.shape[0],row_num,col_num,7,6))
    for n in range(clisccp_data.shape[0]):
        for i in range(row_num):
            for j in range(col_num):
                a = np.zeros(shape=(clisccp_data.shape[3],clisccp_data.shape[4]))    
                a = clisccp_data[n,i+row_range[0],j+col_range[0],0:clisccp_data.shape[3],0:clisccp_data.shape[4]] #没有m，k变量往上提
                for k in range(7):
                    for m in range(6):
                        if k != 0 and m != 0:
                            hist_stand[n,i,j,k,m] = a[2*k+1,2*m+1]+0.5*(a[2*k,2*m+1]+a[2*k+1,2*m]+a[2*k+1,2*m+2]+a[2*k+2,2*m+1])+0.25*(a[2*k,2*m]+a[2*k+2,2*m]+a[2*k,2*m+2]+a[2*k+2,2*m+2])
                        if k != 0 and m == 0:
                            hist_stand[n,i,j,k,m] = a[2*k+1,2*m+1]+0.5*(a[2*k,2*m+1]+a[2*k+1,2*m]+a[2*k+1,2*m+2]+a[2*k+2,2*m+1])+0.25*(a[2*k,2*m]+a[2*k+2,2*m]+a[2*k,2*m+2]+a[2*k+2,2*m+2])+0.5*(a[2*k+1,2*m])+0.25*(a[2*k,2*m]+a[2*k+2,2*m])
                        if k == 0 and m != 0:
                            hist_stand[n,i,j,k,m] = a[2*k+1,2*m+1]+0.5*(a[2*k,2*m+1]+a[2*k+1,2*m]+a[2*k+1,2*m+2]+a[2*k+2,2*m+1])+0.25*(a[2*k,2*m]+a[2*k+2,2*m]+a[2*k,2*m+2]+a[2*k+2,2*m+2])+0.5*(a[2*k,2*m+1])+0.25*(a[2*k,2*m+2]+a[2*k,2*m])
                        if k == 7 and m != 6:
                            hist_stand[n,i,j,k,m] = a[2*k+1,2*m+1]+0.5*(a[2*k,2*m+1]+a[2*k+1,2*m]+a[2*k+1,2*m+2]+a[2*k+2,2*m+1])+0.25*(a[2*k,2*m]+a[2*k+2,2*m]+a[2*k,2*m+2]+a[2*k+2,2*m+2])+0.5*(a[2*k+2,2*m+1])+0.25*(a[2*k+2,2*m+2]+a[2*k+2,2*m])
                        if k == 7 and m != 6:
                            hist_stand[n,i,j,k,m] = a[2*k+1,2*m+1]+0.5*(a[2*k,2*m+1]+a[2*k+1,2*m]+a[2*k+1,2*m+2]+a[2*k+2,2*m+1])+0.25*(a[2*k,2*m]+a[2*k+2,2*m]+a[2*k,2*m+2]+a[2*k+2,2*m+2])+0.5*(a[2*k+1,2*m+2])+0.25*(a[2*k,2*m+2]+a[2*k+2,2*m+2])
                        if k == 7 and m == 6:
                            hist_stand[n,i,j,k,m] = a[2*k+1,2*m+1]+0.5*(a[2*k,2*m+1]+a[2*k+1,2*m]+a[2*k+1,2*m+2]+a[2*k+2,2*m+1])+0.25*(a[2*k,2*m]+a[2*k+2,2*m]+a[2*k,2*m+2]+a[2*k+2,2*m+2])+0.5*(a[2*k+1,2*m+2]+a[2*k+2,2*m+1])+0.25*(a[2*k,2*m+2]+a[2*k+2,2*m])+0.75*a[14,12]
                        if k == 0 and m == 6:
                            hist_stand[n,i,j,k,m] = a[2*k+1,2*m+1]+0.5*(a[2*k,2*m+1]+a[2*k+1,2*m]+a[2*k+1,2*m+2]+a[2*k+2,2*m+1])+0.25*(a[2*k,2*m]+a[2*k+2,2*m]+a[2*k,2*m+2]+a[2*k+2,2*m+2])+0.5*(a[2*k+1,2*m+2]+a[2*k,2*m+1])+0.25*(a[2*k,2*m]+a[2*k+2,2*m+2])+0.75*a[0,12]
                        if k == 7 and m == 0:
                            hist_stand[n,i,j,k,m] = a[2*k+1,2*m+1]+0.5*(a[2*k,2*m+1]+a[2*k+1,2*m]+a[2*k+1,2*m+2]+a[2*k+2,2*m+1])+0.25*(a[2*k,2*m]+a[2*k+2,2*m]+a[2*k,2*m+2]+a[2*k+2,2*m+2])+0.5*(a[2*k+1,2*m]+a[2*k+2,2*m+1])+0.25*(a[2*k+2,2*m+2]+a[2*k,2*m])+0.75*a[14,0]
                        if k == 0 and m == 0:
                            hist_stand[n,i,j,k,m] = a[2*k+1,2*m+1]+0.5*(a[2*k,2*m+1]+a[2*k+1,2*m]+a[2*k+1,2*m+2]+a[2*k+2,2*m+1])+0.25*(a[2*k,2*m]+a[2*k+2,2*m]+a[2*k,2*m+2]+a[2*k+2,2*m+2])+0.5*(a[1,0]+a[0,1])+0.25*(a[0,2]+a[2,0])+0.75*a[0,0]

    return hist_stand
   
    

def histogram_compact_mapping(t_data,x_label,y_label,x_title,y_title,title_height,colorbar_name,textornot,name_group,histogram_cloud,savepicture,address):

    fig,axs = plt.subplots(nrows = t_data.shape[0], ncols = t_data.shape[1],figsize = (t_data.shape[1]*7,t_data.shape[0]*7))
    #figsize = (width，hight)
    extent = (0,1,0,1)
    #将x周和y轴都归一化。方便指定label的位置(0,1)相对比例，否则指定label位置时不好指定
    #求最小值
    mins = []
    for i in range(t_data.shape[0]):
        for j in range(t_data.shape[1]):
            mins.append(np.min(t_data[i,j,:,:]))
    vmin = np.min(mins)
    #求最大值
    maxes = []
    for i in range(t_data.shape[0]):
        for j in range(t_data.shape[1]):
            maxes.append(np.max(t_data[i,j,:,:]))
    vmax = np.max(maxes)
    
    norm = Normalize(vmin = vmin,vmax = vmax)
    #Normalize()跟归一化没有任何关系，函数的作用是将颜色映射到vmin-vmax上，即让接下来的颜色表/颜色柱的起始和终止分别取值vmin和vmax
    #axs[0,0].imshow(data1,extent = extent,vmin = vmin,vmax = vmax,cmap = 'jet_r')
    font_title = {'family' : 'Times New Roman',
                  'color'  : 'black',
                  'weight' : 'medium',
                  'size'   : 38,
                  }
    font_colorbar = {'family' : 'Times New Roman',
                    'color'  : 'black',
                    'weight' : 'medium',
                    'size'   : 42,
                    }
    font_xytitle = {'family' : 'Times New Roman',
                  'color'  : 'black',
                  'weight' : 'semibold',
                  'size'   : 36,
                  }
    font_ticks = {'family' : 'Times New Roman',
                  'color'  : 'black',
                  'weight' : 'medium',
                  'size'   : 32,
                  }
    font_text_1 = {'family' : 'Times New Roman',
                 'color'  : 'black',
                 'weight' : 'bold',
                 'size'   : 29,
                 }           
    font_text_2 = {'family' : 'Times New Roman',
                 'color'  : 'white',
                 'weight' : 'bold',
                 'size'   : 29,
                 } 
    for m in range(t_data.shape[0]):
        for n in range(t_data.shape[1]):
            
            data1 = t_data[m,n,:,:]
            
            #imshow
            if m == t_data.shape[0]-1 and n == t_data.shape[1]-1:
                sc = axs[m,n].imshow(data1,extent = extent,vmin = vmin,vmax = vmax,cmap = plt.cm.Reds) #添加colorbar标记
            else:
                axs[m,n].imshow(data1,extent = extent,norm = norm,cmap =plt.cm.Reds)
            
            #添加子标题
            axs[m,n].set_title(name_group[m,n] + '   ' + str(round(histogram_cloud[m,n],1)) + '%',fontdict=font_title, pad= title_height)
            #边缘设置标题    
            if m == t_data.shape[0]-1:
                axs[m,n].set_xlabel(x_title,fontdict = font_xytitle)
            if n == 0:
                axs[m,n].set_ylabel(y_title,fontdict = font_xytitle)
                
            #边缘设置坐标刻度
            axs[m,n].set_xticks(np.linspace(0,1,len(x_label)))
            if m == t_data.shape[0]-1:
                if n == 0:
                    axs[m,n].set_xticklabels(x_label,fontdict = font_ticks)
                else:
                    axs[m,n].set_xticklabels(['0','1.3','3.6','9.4','23','60','100'],fontdict = font_ticks)
            else:
                axs[m,n].set_xticklabels([''])
            axs[m,n].set_yticks(np.linspace(0,1,len(y_label)))         
            if n == 0:
                axs[m,n].set_yticklabels(y_label,fontdict = font_ticks)
            else:
                axs[m,n].set_yticklabels([''])
            

            #显示数值
            if textornot:
                for i in range(data1.shape[1]):
                    for j in range(data1.shape[0]):
                        '''
                        if data1[j,i] > vmax*0.05 and data1[j,i] < vmax*0.5:
                        '''
                        if data1[j,i] > 1 and data1[j,i] < vmax*0.5:
                            axs[m,n].text((i+0.5-len(str(round(data1[j,i],1)))*0.1)*(1/data1.shape[1]), 1-(j+0.6)*(1/data1.shape[0]), str(round(data1[j,i],1)), fontdict=font_text_1, wrap=True)
                        elif data1[j,i] > vmax*0.5:
                            axs[m,n].text((i+0.5-len(str(round(data1[j,i],1)))*0.1)*(1/data1.shape[1]), 1-(j+0.6)*(1/data1.shape[0]), str(round(data1[j,i],1)), fontdict=font_text_2, wrap=True)

            #显示数值
              
    #weights=['light','normal','medium','semibold','bold','heavy','black']
    #norm = plt.colors.Normalize(vmin=vmin, vmax=vmax)
    #前面三个子图的总宽度 为 全部宽度的 0.9；剩下的0.1用来放置colorbar
    
    #紧凑图像
    plt.tight_layout()
    
    fig.subplots_adjust(right=0.9)
    #colorbar 左 下 宽 高 
    l = 0.92
    b = 0.2
    w = 0.02
    h = 0.7
    #对应 l,b,w,h；设置colorbar位置；
    rect = [l,b,w,h] 
    cbar_ax = fig.add_axes(rect) 
    cb = plt.colorbar(sc, cax=cbar_ax)
    cb.set_label(colorbar_name,fontdict=font_colorbar)
    cb.ax.tick_params(labelsize=32)  #设置色标刻度字体大小。 
    if savepicture:
        os.chdir(address)
        #savename = input("输入保存名（histogram pictures）：\n")
        savename = 'Figure S4'
        plt.savefig(savename)
    plt.show()
    

def histogram_compact_trend_mapping(t_data,x_label,y_label,x_title,y_title,title_height,colorbar_name,textornot,name_group,histogram_correlation,savepicture,address,color_limit):

    fig,axs = plt.subplots(nrows = t_data.shape[0], ncols = t_data.shape[1],figsize = (t_data.shape[1]*7,t_data.shape[0]*7))
    #figsize = (width，hight)
    extent = (0,1,0,1)
    #将x周和y轴都归一化。方便指定label的位置(0,1)相对比例，否则指定label位置时不好指定
    #求最小值
    mins = []
    for i in range(t_data.shape[0]):
        for j in range(t_data.shape[1]):
            mins.append(np.min(t_data[i,j,:,:]))
    vmin = np.min(mins)
    #求最大值
    maxes = []
    for i in range(t_data.shape[0]):
        for j in range(t_data.shape[1]):
            maxes.append(np.max(t_data[i,j,:,:]))
    vmax = np.max(maxes)
    if color_limit:
        vmin1 = vmin
        vmax1 = vmax
        if vmax > np.abs(vmin):
            up_limit = 1
            vmin = vmin1*color_limit
            vmax = -vmin1*color_limit
        else:
            up_limit = 0
            vmin = -vmax1*color_limit
            vmax = vmax1*color_limit
    norm = Normalize(vmin = vmin,vmax = vmax) #不能在内部修改，需要提前修改
    #Normalize()跟归一化没有任何关系，函数的作用是将颜色映射到vmin-vmax上，即让接下来的颜色表/颜色柱的起始和终止分别取值vmin和vmax
    #axs[0,0].imshow(data1,extent = extent,vmin = vmin,vmax = vmax,cmap = 'jet_r')
    font_title = {'family' : 'Times New Roman',
                  'color'  : 'black',
                  'weight' : 'medium',
                  'size'   : 38,
                  }
    font_colorbar = {'family' : 'Times New Roman',
                    'color'  : 'black',
                    'weight' : 'medium',
                    'size'   : 42,
                    }
    font_xytitle = {'family' : 'Times New Roman',
                  'color'  : 'black',
                  'weight' : 'semibold',
                  'size'   : 36,
                  }
    font_ticks = {'family' : 'Times New Roman',
                  'color'  : 'black',
                  'weight' : 'medium',
                  'size'   : 32,
                  }
    font_text_1 = {'family' : 'Times New Roman',
                 'color'  : 'black',
                 'weight' : 'bold',
                 'size'   : 35,
                 }           
    font_text_2 = {'family' : 'Times New Roman',
                 'color'  : 'white',
                 'weight' : 'bold',
                 'size'   : 35,
                 } 
    for m in range(t_data.shape[0]):
        for n in range(t_data.shape[1]):
            
            data1 = t_data[m,n,:,:]
            
            #imshow
            if m == t_data.shape[0]-1 and n == t_data.shape[1]-1:
                sc = axs[m,n].imshow(data1,extent = extent,norm = norm,vmin = vmin,vmax = vmax,cmap = plt.cm.coolwarm) #添加colorbar标记
            else:
                axs[m,n].imshow(data1,extent = extent,norm = norm,cmap =plt.cm.coolwarm)
            
            #添加子标题
            if histogram_correlation[m,n] != 0:
                axs[m,n].set_title(name_group[m,n] + '   ' + str(round(histogram_correlation[m,n],2)),fontdict=font_title, pad= title_height)
            else:
                axs[m,n].set_title(name_group[m,n],fontdict=font_title, pad= title_height)                
            #边缘设置标题    
            if m == t_data.shape[0]-1:
                axs[m,n].set_xlabel(x_title,fontdict = font_xytitle)
            if n == 0:
                axs[m,n].set_ylabel(y_title,fontdict = font_xytitle)
                
            #边缘设置坐标刻度
            axs[m,n].set_xticks(np.linspace(0,1,len(x_label)))
            if m == t_data.shape[0]-1:
                if n == 0:
                    axs[m,n].set_xticklabels(x_label,fontdict = font_ticks)
                else:
                    '''
                    axs[m,n].set_xticklabels(['0','1.3','3.6','9.4','23','60','100'],fontdict = font_ticks)
                    '''
                    axs[m,n].set_xticklabels(['0','3.6','23','100'],fontdict = font_ticks)
                    
            else:
                axs[m,n].set_xticklabels([''])
            axs[m,n].set_yticks(np.linspace(0,1,len(y_label)))         
            if n == 0:
                axs[m,n].set_yticklabels(y_label,fontdict = font_ticks)
            else:
                axs[m,n].set_yticklabels([''])
            """
            #显示数值
            if textornot:
                for i in range(data1.shape[1]):
                    for j in range(data1.shape[0]):
                        if abs(data1[j,i]) > vmax*0.03 and abs(data1[j,i]) < vmax*0.5:
                            axs[m,n].text((i+0.5-len(str(round(data1[j,i],1)))*0.1)*(1/data1.shape[1]), 1-(j+0.6)*(1/data1.shape[0]), str(round(data1[j,i],2)), fontdict=font_text_1, wrap=True)
                        elif abs(data1[j,i]) > vmax*0.5:
                            axs[m,n].text((i+0.5-len(str(round(data1[j,i],1)))*0.1)*(1/data1.shape[1]), 1-(j+0.6)*(1/data1.shape[0]), str(round(data1[j,i],2)), fontdict=font_text_2, wrap=True)
            """
            #显示数值
            if textornot:
                for i in range(data1.shape[1]):
                    for j in range(data1.shape[0]):
                        if abs(data1[j,i]) > vmax*0.03 and abs(data1[j,i]) < vmax*0.5:
                            axs[m,n].text((i+0.5-len(str(round(data1[j,i],1)))*0.1)*(1/data1.shape[1]), 1-(j+0.6)*(1/data1.shape[0]), str(round(data1[j,i],2)), fontdict=font_text_1, wrap=True)
                        elif abs(data1[j,i]) > vmax*0.5:
                            axs[m,n].text((i+0.5-len(str(round(data1[j,i],1)))*0.1)*(1/data1.shape[1]), 1-(j+0.6)*(1/data1.shape[0]), str(round(data1[j,i],2)), fontdict=font_text_2, wrap=True)                                  
                            
    plt.tight_layout()
    
    fig.subplots_adjust(right=0.85)
    #colorbar 左 下 宽 高 
    l = 0.87
    b = 0.2
    w = 0.02
    h = 0.7
    #对应 l,b,w,h；设置colorbar位置；
    rect = [l,b,w,h] 
    cbar_ax = fig.add_axes(rect) 
    cb = plt.colorbar(sc, cax=cbar_ax)
    cb.set_label(colorbar_name,fontdict=font_colorbar)
    #cb.set_clim(-100,100)
    cb.set_ticks(np.linspace(vmin,vmax,7))
    if color_limit:
        ctick = list(np.linspace(vmin,vmax,7))
        for i in range(len(ctick)):
            ctick[i] = round(ctick[i],2)
        #cb.set_ticklabels((str(ctick[0]),str(ctick[1]),str(ctick[2]),str(ctick[3]),str(ctick[4]),str(ctick[5]),'>'+str(ctick[6])))  
        cb.set_ticklabels(('<'+str(ctick[0]),str(ctick[1]),str(ctick[2]),str(ctick[3]),str(ctick[4]),str(ctick[5]),'>'+str(ctick[6])))  
            
    cb.ax.tick_params(labelsize=32)  #设置色标刻度字体大小。 
    if savepicture:
        os.chdir(address)
        #savename = input("输入保存名（histogram pictures）：\n")
        savename = 'Figure 4'
        plt.savefig(savename)
    plt.show()
    
    
    
def creat_obs(onlyland,cfc_fulldata,SWpoint_SA,NEpoint_SA,mean_num,area):
    
    rows = int(cfc_fulldata.shape[1]*(NEpoint_SA[0]-SWpoint_SA[0])/180)
    cols = int(cfc_fulldata.shape[2]*(NEpoint_SA[1]-SWpoint_SA[1])/360)
    
    cut_data = np.zeros(shape=(cfc_fulldata.shape[0],rows,cols))
    if onlyland:
        valid_data = np.zeros(shape=(cfc_fulldata.shape[0],rows,cols))
    for i in range(cfc_fulldata.shape[0]):
        cut_data[i,:,:] = check_data_slice(cfc_fulldata[i,:,:],SWpoint_SA,NEpoint_SA,0)
        if onlyland:
            valid_data[i,:,:] = basic_mask_map('part',tiff_data,cut_data[i,:,:],SWpoint_SA,NEpoint_SA,0)
        else:
            valid_data = cut_data
        
    valid_data_list = np.zeros(shape=(cfc_fulldata.shape[0]))     
    
    valid_data_list[:] = pro_mean_3d(valid_data,SWpoint_SA,NEpoint_SA,mean_num)
    
    valid_data_list = valid_data_list*100
    
    if onlyland:
        save_address = '/Volumes/NO NAME/Work/xlsx文件/cfc_land_'+ area +'.xlsx'
        creat_xlsx_1d(valid_data_list,save_address = save_address)  
    else:
        save_address = '/Volumes/NO NAME/Work/xlsx文件/cfc_'+ area +'.xlsx'
        creat_xlsx_1d(valid_data_list,save_address = save_address)          

    
def creat_clt(onlyland,clt_nat_fulldata,clt_ghg_fulldata,clt_aer_fulldata,SWpoint_SA,NEpoint_SA,mean_num,area):
    
    clt_nat_SA = np.zeros(shape=(clt_nat_fulldata.shape[0],33,200,400))
    clt_ghg_SA = np.zeros(shape=(clt_ghg_fulldata.shape[0],33,200,400))
    clt_aer_SA = np.zeros(shape=(clt_aer_fulldata.shape[0],33,200,400))
    
    for i in range(clt_nat_fulldata.shape[0]):
        clt_nat_SA[i,:,:,:] = extract_orig_map_3d(clt_nat_fulldata[i,:,:,:],SWpoint_SA,NEpoint_SA)
        
    for i in range(clt_ghg_fulldata.shape[0]):
        clt_ghg_SA[i,:,:,:] = extract_orig_map_3d(clt_ghg_fulldata[i,:,:,:],SWpoint_SA,NEpoint_SA)    
        
    for i in range(clt_aer_fulldata.shape[0]):
        clt_aer_SA[i,:,:,:] = extract_orig_map_3d(clt_aer_fulldata[i,:,:,:],SWpoint_SA,NEpoint_SA)
    #============================================================================================
    # 检验全局地图
    # mapping((valid_map(clt_nat_SA[0,:,:,:]))[0,:,:])
    
    #============================================================================================
    # 生成序列
    
    nat_SA_list = np.zeros(shape=(clt_nat_fulldata.shape[0],33))     
    
    for i in range(clt_nat_fulldata.shape[0]): #模型数量
        
        temp_data = valid_map(clt_nat_SA[i,:,:,:])
        if onlyland:
            for j in range(temp_data.shape[0]): #年份数量
                if not j:
                    temp_data[j,:,:] = basic_mask_map('part',tiff_data,temp_data[j,:,:],SWpoint_SA,NEpoint_SA,0)
                else:
                    temp_data[j,:,:] = copy_mask_map(temp_data[0,:,:],temp_data[j,:,:])
        nat_SA_list[i,:] = pro_mean_3d(temp_data,SWpoint_SA,NEpoint_SA,mean_num)
                
    ghg_SA_list = np.zeros(shape=(clt_ghg_fulldata.shape[0],33))     
    
    for i in range(clt_ghg_fulldata.shape[0]):
        
        temp_data = valid_map(clt_ghg_SA[i,:,:,:])
        if onlyland:
            for j in range(temp_data.shape[0]):
                if not j:
                    temp_data[j,:,:] = basic_mask_map('part',tiff_data,temp_data[j,:,:],SWpoint_SA,NEpoint_SA,0)
                else:
                    temp_data[j,:,:] = copy_mask_map(temp_data[0,:,:],temp_data[j,:,:])
        ghg_SA_list[i,:] = pro_mean_3d(temp_data,SWpoint_SA,NEpoint_SA,mean_num)
                
    aer_SA_list = np.zeros(shape=(clt_aer_fulldata.shape[0],33))     
    
    for i in range(clt_aer_fulldata.shape[0]):
        
        temp_data = valid_map(clt_aer_SA[i,:,:,:])
        if onlyland:
            for j in range(temp_data.shape[0]):
                if not j:
                    temp_data[j,:,:] = basic_mask_map('part',tiff_data,temp_data[j,:,:],SWpoint_SA,NEpoint_SA,0)
                else:
                    temp_data[j,:,:] = copy_mask_map(temp_data[0,:,:],temp_data[j,:,:])
        aer_SA_list[i,:] = pro_mean_3d(temp_data,SWpoint_SA,NEpoint_SA,mean_num)
        
    if onlyland:
        save_address = '/Volumes/NO NAME/output/xlsx_files/aer_land_' + area + '.xlsx'
        creat_xlsx_2d(aer_SA_list,save_address = save_address)
        save_address = '/Volumes/NO NAME/output/xlsx_files/ghg_land_' + area + '.xlsx'
        creat_xlsx_2d(ghg_SA_list,save_address = save_address)
        save_address = '/Volumes/NO NAME/output/xlsx_files/nat_land_' + area + '.xlsx'
        creat_xlsx_2d(nat_SA_list,save_address = save_address)
    else:
        save_address = '/Volumes/NO NAME/output/xlsx_files/aer_' + area + '.xlsx'
        creat_xlsx_2d(aer_SA_list,save_address = save_address)
        save_address = '/Volumes/NO NAME/output/xlsx_files/ghg_' + area + '.xlsx'
        creat_xlsx_2d(ghg_SA_list,save_address = save_address)
        save_address = '/Volumes/NO NAME/output/xlsx_files/nat_' + area + '.xlsx'
        creat_xlsx_2d(nat_SA_list,save_address = save_address)
        
    return nat_SA_list,ghg_SA_list,aer_SA_list
    


def creat_pi(onlyland,SWpoint_SA,NEpoint_SA,mean_num,area,pi_no,monthoryear,full_period,time_start,time_end,interpolateornot):
    
    for i in range(len(pi_no)):
    
        address_1 = '/Volumes/NO NAME/Work/piControl_' + pi_no[i]
        
        pi_group_1,pi_fullname_set_1,pi_class_list_1 = piControl_read_set(address_1,monthoryear,full_period,time_start,time_end,interpolateornot) 
        pi_valid_time = len(np.nonzero(pi_group_1[:,0,0,0])[0]) #提取有效年份
        
        pi_fulldata_1 = np.zeros(shape=(pi_valid_time,33,360,720)) 
            
        for m in range(pi_valid_time):
            pi_fulldata_1[m,:,:,:] = paramove_3d_valid_map(pi_group_1[m,:,:,:])
                
        #============================================================================================
        # 生成部分地图
        pi_SA_data_1 = np.zeros(shape=(pi_valid_time,33,360,720))
                
        for m in range(pi_valid_time):
            pi_SA_data_1[m,:,:,:] = extract_orig_map_3d(pi_fulldata_1[m,:,:,:],SWpoint_SA,NEpoint_SA)
                
        #============================================================================================
        # 生成序列
            
        pi_SA_list_3 = np.zeros(shape=(pi_valid_time,33))     
            
        for m in range(pi_valid_time):
                
            temp_data = valid_map(pi_SA_data_1[m,:,:,:])
            if onlyland:
                for j in range(temp_data.shape[0]):
                    if not j:
                        temp_data[j,:,:] = basic_mask_map('part',tiff_data,temp_data[j,:,:],SWpoint_SA,NEpoint_SA,0)
                    else:
                        temp_data[j,:,:] = copy_mask_map(temp_data[0,:,:],temp_data[j,:,:])
            pi_SA_list_3[m,:] = pro_mean_3d(temp_data,SWpoint_SA,NEpoint_SA,mean_num)
                
        #============================================================================================
        # save
        #os.chdir('/Volumes/NO NAME/Work/npy文件')
        #np.save('pi_'+ area +'_list_' + pi_no[i] + '.npy',pi_SA_list_3)
        #save_address = '/Volumes/NO NAME/Work/xlsx文件/pi_'+ area +'_list_' + pi_no[i]+ '_e_t.xlsx'
        #creat_xlsx_3p(pi_SA_list_3,pi_fullname_set_1,pi_class_list_1,save_address = save_address)
            
        if i == 0:
            front_list = pi_SA_list_3
        else:
            back_list = pi_SA_list_3
            pi_SA_list_1_5 = np.row_stack((front_list,back_list))
            front_list = pi_SA_list_1_5

        del(pi_SA_data_1)
        del(pi_fulldata_1)
        del(pi_group_1)
        
    return pi_SA_list_1_5
        



def cci_clisccp_data(ff,address,var,plus_var,SWpoint,NEpoint,period):

    #提取数据
    #ff = os.listdir(address)
    #ff.sort()
    
    time_l = int(period[:-1])
    year_or_month = period[-1]
    
    #标准数据范围
    lats = NEpoint[0] - SWpoint[0]
    lons = NEpoint[1] - SWpoint[1]
    #原始数据范围
    data_lats = int((lats/180)*360) 
    data_lons = int((lons/360)*720)
    
    #基础数据框
    if year_or_month == 'y':
        hist_stand = np.zeros(shape=(len(ff),data_lats,data_lons,15,13))
        #年份数据框
        hist_stand_y = np.zeros(shape=(time_l,data_lats,data_lons,15,13))
        
    else:
        hist_stand = np.zeros(shape=(len(ff),data_lats,data_lons,15,13))
    
    year_equal = ''
    year_count = -1
    month_num = np.zeros(shape=(time_l,data_lats,data_lons))

    for n in range(len(ff)):
        
        print(n)
        print(datetime.datetime.now())
        
        #附加数据
        if len(plus_var) > 1:        
            plus_var_data = nc_var(address,ff[n],plus_var)[0] #读取云量月数据    
        
        #读取主要数据
        var_data = nc_var(address,ff[n],var)
        
        #年份序列
        if year_or_month == 'y':        
            #同一年
            if ff[n][:4] == year_equal:
    
                for i in range(data_lats):
                    for j in range(data_lons):  
                        hist_stand[n,i,j,:,:] = var_data[0,0,0:15,0:13,i,j] + var_data[0,1,0:15,0:13,i,j]       #没有m，k变量往上提
                        if np.mean(hist_stand[n,i,j,:,:]) > 0.0 and plus_var_data[i,j] > 0.0:
                            month_num[year_count,i,j] += 1              
                            hist_stand[n,i,j,:,:] = (hist_stand[n,i,j,:,:]/sum(sum(hist_stand[n,i,j,::,::])))*plus_var_data[i,j]#融入云量变量
                            #整体赋值比循环体更快！！！
                            hist_stand_y[year_count,i,j,:,:] += hist_stand[n,i,j,:,:]                        
                        else:
                            hist_stand[n,i,j,:,:] = 0
                            month_num[year_count,i,j] = month_num[year_count,i,j]              
                    
                if n == len(ff)-1:#结束汇总成年数据
                    
                    for i in range(data_lats):
                        for j in range(data_lons):
                            if np.mean(hist_stand_y[year_count,i,j,:,:]) > 0.0:
                                hist_stand_y[year_count,i,j,:,:] = hist_stand_y[year_count,i,j,:,:]/month_num[year_count,i,j]   
                
            else:
                year_count += 1
                
                if year_count > 0:#数据年份改变年数据

                    for i in range(data_lats):
                        for j in range(data_lons):
                            if np.mean(hist_stand_y[year_count-1,i,j,:,:]) > 0.0:
                                hist_stand_y[year_count-1,i,j,:,:] = hist_stand_y[year_count-1,i,j,:,:]/month_num[year_count-1,i,j]   
    
                for i in range(data_lats):
                    for j in range(data_lons):  
                        hist_stand[n,i,j,:,:] = var_data[0,0,0:15,0:13,i,j] + var_data[0,1,0:15,0:13,i,j]       #没有m，k变量往上提
                        if np.mean(hist_stand[n,i,j,:,:]) > 0.0 and plus_var_data[i,j] > 0.0:
                            month_num[year_count,i,j] += 1              
                            hist_stand[n,i,j,:,:] = (hist_stand[n,i,j,:,:]/sum(sum(hist_stand[n,i,j,::,::])))*plus_var_data[i,j]#融入云量变量
                            hist_stand_y[year_count,i,j,:,:] += hist_stand[n,i,j,:,:]                        
                        else:
                            hist_stand[n,i,j,:,:] = 0
                            month_num[year_count,i,j] = month_num[year_count,i,j] 
                
            year_equal = ff[n][:4]
            
        else:
            for i in range(data_lats):
                for j in range(data_lons):
                    hist_stand[n,i,j,:,:] = var_data[0,0,0:15,0:13,i,j] + var_data[0,1,0:15,0:13,i,j]   
                    if np.mean(hist_stand[n,i,j,:,:]) > 0.0 and plus_var_data[i,j] > 0.0:
                        hist_stand[n,i,j,:,:] = (hist_stand[n,i,j,:,:]/sum(sum(hist_stand[n,i,j,::,::])))*plus_var_data[i,j]#融入云量变量
                    else:
                        hist_stand[n,i,j,:,:] = 0
                        
    if year_or_month == 'y':
        
        return hist_stand_y
    
    else:
    
        return hist_stand

def nc_var(address,filename,var):
    
    os.chdir(address)
    fh = nc.Dataset(filename)
    data = fh.variables[var][:]
    #print(fh.variables[var])
    fh.close()
    return data

def scaling_factors_compare(point_value,dy,row_col,subtitle,var_name,xticks,color_list,savepictures,savename):
    
    x = np.zeros(shape=(point_value.shape[0],point_value.shape[1]))
    for i in range(point_value.shape[0]):
        for j in range(point_value.shape[1]):
            x[i,j] = i*2+j*10
            
    plt.figure(figsize=(14, 7))
    
    for m in range(1):
    
        map_location = int(str(row_col[0]) + str(row_col[1]) + str(m+1))
        
        plt.subplot(map_location) #确定第一个图的位置
        
        plt.style.use('seaborn-whitegrid')
         
        #fmt是一种控制线条和点的外观的代码格式,语法与plt.plot的缩写代码一样.
        plt.axhline(0,color = 'black',linestyle = '-',linewidth = 2.5)
        plt.axhline(1,color = 'grey',linestyle = '--',linewidth = 2)
        plt.ylabel('scaling factors',fontsize=24) 
        plt.title(subtitle,fontproperties='SimHei',fontsize = 24) 
        
        for i in range(point_value.shape[0]):
            plt.errorbar(x[i,:],point_value[i,:] ,yerr=dy[i,:],label = var_name[i],fmt='o',ecolor=color_list[i],color='black',elinewidth=4,capsize=6)

        #plt.axvline(17,color = 'black',linestyle = '--',linewidth = 2.5)
        #plt.axvline(27,color = 'grey',linestyle = '--',linewidth = 2.5)
        
        plt.xticks(x[1,:],xticks,fontsize = 24)
        plt.yticks(fontsize = 20)
        plt.legend(prop={'size':24},frameon = True,ncol=3,loc = 'lower left',shadow = True)
        plt.grid(None)        #删除网格线
        plt.tight_layout()
    #plt.suptitle('Title',fontproperties='SimHei',fontsize = 28)
    if savepictures:
        os.chdir('/Volumes/NO NAME 1/output/Figures')
        plt.savefig(savename)
        
 
def confidence_interval_2d(data,color,depict):
    
    X = range(0,data.shape[-1])
    out_data = np.zeros(shape=(2,data.shape[-1]))
    
    for i in range(data.shape[-1]):
        out_data[0,i] = np.percentile(data[:,i], 15.9)
        out_data[1,i] = np.percentile(data[:,i], 84.1)

    if depict:
        plt.fill_between(X,out_data[0,:],out_data[1,:],color=color,alpha=.6)
        plt.show()
        
    return out_data


def randomcolor():
    colorArr = ['1','2','3','4','5','6','7','8','9','A','B','C','D','E','F']
    color = ""
    for i in range(6):
        color += colorArr[random.randint(0,14)]
    return "#"+color

    
def variety_multiple_forcing(data,initial_num,grid_scale,color_list,label_list,title_list,map_list,savepictures): #图画规格，图画数量
    
    plt.figure(figsize=(20,14))
    x = np.linspace(1,data.shape[2],data.shape[2])
    #plt.figure(grid_scale)
    plt.style.use('seaborn-whitegrid')
    X = range(1,data.shape[-1]+1)
    

    trend_group = np.zeros(shape = data.shape[:-1])
    sim_trend_group = np.zeros(shape = (9,3))    
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            trend_group[i,j] = round(trend_1d(data[i,j,:]),3)
            if j == 1:
                sim_trend_group[i,0] = round(trend_1d(np.mean(data[i,j:j+6,:],axis = 0)),3)
            elif j == 7:
                sim_trend_group[i,1] = round(trend_1d(np.mean(data[i,j:j+5,:],axis = 0)),3)
            elif j == 12:
                sim_trend_group[i,2] = round(trend_1d(np.mean(data[i,j:j+5,:],axis = 0)),3)                
    for i in range(data.shape[0]):
        subplot_num = int(map_list[i]) + grid_scale*10
        plt.subplot(subplot_num)
        mean_anomaly = np.mean(data[i,0,:])
        if i == 3:
            plt.ylabel('Anomaly (%)',fontsize=30) 
        if i == 8:
            plt.xlabel('Period (year)',fontsize=30) 
        for j in range(data.shape[1]):
            if j == 0:
                plt.plot(x,data[i,j,:]-mean_anomaly,color_list[0],label= label_list[0] + '   ' + str(trend_group[i,j]),linewidth = 3.5,linestyle = '-')
                plt.axhline(0,color = 'black',linestyle = '--',linewidth = 3.5)
            elif j>0 and j<7:
                if j == 1:
                    confi = confidence_interval_2d(data[i,j:j+6,:]-mean_anomaly,color_list[1],0)
                    plt.fill_between(X,confi[0,:],confi[1,:],color=color_list[1],alpha=.5)
                    #t1, = plt.plot(x,np.mean(data[i,j:j+6,:],axis = 0)-mean_anomaly,color_list[1],linewidth = 3.5,linestyle = '--')                     
                    if i == 4:
                        p1, = plt.plot(x,np.mean(data[i,j:j+6,:],axis = 0)-mean_anomaly,color_list[1],linewidth = 3.5,linestyle = '--') 
                        plt.plot(x,np.mean(data[i,j:j+6,:],axis = 0)-mean_anomaly,color_list[1],label= label_list[1] + '   ' + str(sim_trend_group[i,0]),linewidth = 3.5,linestyle = '--')                                             
                    else:
                        plt.plot(x,np.mean(data[i,j:j+6,:],axis = 0)-mean_anomaly,color_list[1],label= label_list[1] + '   ' + str(sim_trend_group[i,0]),linewidth = 3.5,linestyle = '--')                         
                #plt.plot(x,data[i,j,:]-mean_anomaly,color_list[1],linewidth = 1.5,linestyle = '--') 
                    
            elif j>6 and j<12:
                if j == 7:
                    confi = confidence_interval_2d(data[i,j:j+5,:]-mean_anomaly,color_list[2],0)
                    plt.fill_between(X,confi[0,:],confi[1,:],color=color_list[2],alpha=.5)
                    #t2, = plt.plot(x,np.mean(data[i,j:j+5,:],axis = 0)-mean_anomaly,color_list[2],linewidth = 3.5,linestyle = '--')                                         
                    if i == 4:                    
                        p2, = plt.plot(x,np.mean(data[i,j:j+5,:],axis = 0)-mean_anomaly,color_list[2],linewidth = 3.5,linestyle = '--') 
                        plt.plot(x,np.mean(data[i,j:j+5,:],axis = 0)-mean_anomaly,color_list[2],label= label_list[2] + '   ' + str(sim_trend_group[i,1]),linewidth = 3.5,linestyle = '--')                                             
                    else:
                        plt.plot(x,np.mean(data[i,j:j+5,:],axis = 0)-mean_anomaly,color_list[2],label= label_list[2] + '   ' + str(sim_trend_group[i,1]),linewidth = 3.5,linestyle = '--')                         
                #plt.plot(x,data[i,j,:]-mean_anomaly,color_list[2],linewidth = 1.5,linestyle = '--')
            else:
                if j == 12:
                    confi = confidence_interval_2d(data[i,j:j+5,:]-mean_anomaly,color_list[3],0)
                    plt.fill_between(X,confi[0,:],confi[1,:],color=color_list[3],alpha=.5)
                    #t3, = plt.plot(x,np.mean(data[i,j:j+5,:],axis = 0)-mean_anomaly,color_list[3],linewidth = 3.5,linestyle = '--')                                         
                    if i == 4:
                        p3, = plt.plot(x,np.mean(data[i,j:j+5,:],axis = 0)-mean_anomaly,color_list[3],linewidth = 3.5,linestyle = '--')
                        plt.plot(x,np.mean(data[i,j:j+5,:],axis = 0)-mean_anomaly,color_list[3],label= label_list[3] + '   ' + str(sim_trend_group[i,2]),linewidth = 3.5,linestyle = '--')                         
                    else:
                        plt.plot(x,np.mean(data[i,j:j+5,:],axis = 0)-mean_anomaly,color_list[3],label= label_list[3] + '   ' + str(sim_trend_group[i,2]),linewidth = 3.5,linestyle = '--')                         
                #plt.plot(x,data[i,j,:]-mean_anomaly,color_list[3],linewidth = 1.5,linestyle = '--')                

            #if j == 12:
                #l2 = plt.legend([t1, t2, t3], [str(sim_trend_group[i,0]), str(sim_trend_group[i,1]),str(sim_trend_group[i,2])],bbox_to_anchor=(1,-0.05),prop={'size':19},frameon = True,ncol=3)
                #plt.gca().add_artist(l2) 
            
        if i == 8:
            l1 = plt.legend([p1, p2, p3], [label_list[1], label_list[2],label_list[3]],bbox_to_anchor=(1.2,-0.22),prop={'size':30},frameon = True,ncol=3)

        plt.legend(prop={'size':18},frameon = True,ncol=2 ,bbox_to_anchor=(1.022,1.3), shadow = False,loc = 'upper right')       
        if i == 8:
            plt.gca().add_artist(l1) 
        plt.title(title_list[i],fontproperties='Time New Roman',fontsize = 30,loc = 'left')
        #plt.ylim(40,80)#坐标系范围
        plt.xlim(1,33)#坐标系范围
        if i == 4 or i == 5 or i == 8:
            plt.xticks(np.array([9,19,29]),(['1990','2000','2010']),fontsize = 24)
        else:
            plt.xticks(np.array([9,19,29]),(['','','']),fontsize = 24)
        #plt.xticks(fontsize = 24)
        plt.yticks(fontsize = 24)
        #if (int(map_list[i])-1)%3 != 0:
        #    plt.yticks([])  #去掉纵坐标值,去除部分放在最后面
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)        
    if savepictures:
        os.chdir('/Volumes/NO NAME 1/output/Figures')
        plt.savefig('confi_variety_multiple')
    plt.show()


def scale_variety_multiple(data,grid_scale,color_list,label_list,title_list,map_list,savepictures): #图画规格，图画数量
    
    plt.figure(figsize=(20,14))
    x = np.linspace(1,data.shape[2],data.shape[2])
    #plt.figure(grid_scale)
    plt.style.use('seaborn-whitegrid')
    
    """
    添加趋势变化值
    """
    trend_group = np.zeros(shape = data.shape[:-1])
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            trend_group[i,j] = round(trend_1d(data[i,j,:]),3)
    for i in range(data.shape[0]):
        subplot_num = int(map_list[i]) + grid_scale*10
        plt.subplot(subplot_num)
        for j in range(data.shape[1]):
            if j == 0:
                plt.plot(x,data[i,j,:],color_list[j],label= label_list[j] + '   ' + str(trend_group[i,j]),linewidth = 2.5,linestyle = '-')
            else:
                plt.plot(x,data[i,j,:],color_list[j],label= label_list[j] + '   ' + str(trend_group[i,j]),linewidth = 1.5,linestyle = '--')                
        plt.legend(prop={'size':18},frameon = True,ncol=2 , shadow = True,loc = 'upper right')
        plt.title(title_list[i],fontproperties='Time New Roman',fontsize = 30)
        plt.ylim(-0.3,0.5)#坐标系范围
        plt.xticks(fontsize = 24)
        plt.yticks(fontsize = 24)
        if (int(map_list[i])-1)%3 != 0:
            plt.yticks([])  #去掉纵坐标值,去除部分放在最后面
        plt.tight_layout()
        
    if savepictures:
        os.chdir('/Volumes/NO NAME 1/output/Figures')
        plt.savefig('variety_multiple')
    plt.show()

    
def ff_open(address):
    
    ff = os.listdir(address)
    ff.sort()
    ff = remove_abnormal_file(ff)
    
    return ff
    

def trend_1d(m_data):

    x = np.linspace(1,m_data.shape[0],m_data.shape[0])
    
    pfit = np.polyfit(x,m_data,1)
    y_fun = np.poly1d(pfit)
    out_map = (y_fun(x)[-1] - y_fun(x)[0])   
            
    return out_map


def extract_trend_3d(data,SWpoint,NEpoint,mean_num):
    
    lats = NEpoint[0] - SWpoint[0]
    lons = NEpoint[1] - SWpoint[1]
    
    cutdata = np.zeros(shape = (data.shape[0],int((lats/180)*data.shape[1]),int((lons/360)*data.shape[2])))
    
    for i in range(int((lats/180)*data.shape[1])):
        for j in range(int((lons/360)*data.shape[2])):
            cutdata[:,i,j] = data[:,i+int((SWpoint[0]/180)*data.shape[1]),j + int((SWpoint[1]/360)*data.shape[2])]    
    
    data_list = pro_mean_3d(cutdata,SWpoint,NEpoint,mean_num)
    
    return trend_1d(data_list)


def extract_list_3d(data,SWpoint,NEpoint,mean_num):
    
    lats = NEpoint[0] - SWpoint[0]
    lons = NEpoint[1] - SWpoint[1]
    
    cutdata = np.zeros(shape = (data.shape[0],int((lats/180)*data.shape[1]),int((lons/360)*data.shape[2])))
    
    for i in range(int((lats/180)*data.shape[1])):
        for j in range(int((lons/360)*data.shape[2])):
            cutdata[:,i,j] = data[:,i+int((SWpoint[0]/180)*data.shape[1]),j + int((SWpoint[1]/360)*data.shape[2])]    
    
    data_list = pro_mean_3d(cutdata,SWpoint,NEpoint,mean_num)
    
    return data_list

def extract_data_3d(data,SWpoint,NEpoint,mean_num):
    
    lats = NEpoint[0] - SWpoint[0]
    lons = NEpoint[1] - SWpoint[1]
    
    cutdata = np.zeros(shape = (data.shape[0],int((lats/180)*data.shape[1]),int((lons/360)*data.shape[2])))
    
    for i in range(int((lats/180)*data.shape[1])):
        for j in range(int((lons/360)*data.shape[2])):
            cutdata[:,i,j] = data[:,i+int((SWpoint[0]/180)*data.shape[1]),j + int((SWpoint[1]/360)*data.shape[2])]    
    
    return cutdata

def extract_trend_map_3d(data,SWpoint,NEpoint,mean_num):
    
    lats = NEpoint[0] - SWpoint[0]
    lons = NEpoint[1] - SWpoint[1]
    
    cutdata = np.zeros(shape = (data.shape[0],int((lats/180)*data.shape[1]),int((lons/360)*data.shape[2])))
    
    for i in range(int((lats/180)*data.shape[1])):
        for j in range(int((lons/360)*data.shape[2])):
            cutdata[:,i,j] = data[:,i+int((SWpoint[0]/180)*data.shape[1]),j + int((SWpoint[1]/360)*data.shape[2])]    
    
    trend_map = np.zeros(shape = cutdata.shape[-2:])
    
    for i in range(cutdata.shape[1]):
        for j in range(cutdata.shape[2]):
            trend_map[i,j] = trend_1d(cutdata[:,i,j])
    
    return trend_map


def map_pearson(mat1,mat2):
	"""Root mean square error between two matrices, ignoring zeroes"""
	assert mat1.shape == mat2.shape #不相等触发异常
	#convert to vectors
	vec1 = mat1.flatten()#将多维数组转化为一维数组
	vec2 = mat2.flatten()

	#remove zeroes
	nonzero = [i for i in range(len(vec1)) if vec1[i] != 0 and vec2[i] != 0]
	vec1 = vec1[nonzero]
	vec2 = vec2[nonzero]

	r, p = stats.pearsonr(vec1, vec2)
	return r,p

def map_speamanr(mat1,mat2):
	"""Root mean square error between two matrices, ignoring zeroes"""
	assert mat1.shape == mat2.shape
	#convert to vectors
	vec1 = mat1.flatten()
	vec2 = mat2.flatten()

	#remove zeroes
	nonzero = [i for i in range(len(vec1)) if vec1[i] != 0 and vec2[i] != 0]
	vec1 = vec1[nonzero]
	vec2 = vec2[nonzero]

	r, p = spearmanr(vec1, vec2)
	return r,p



# Subtract data
    
"""
monthoryear = 'year'
time_start = '198201'
time_end = '201412'
interpolateornot = 0

os.chdir('/Volumes/NO NAME 1/Work/CMIP-AER Work')
address = '/Volumes/NO NAME 1/Work/CMIP-AER Work'
clt_aer_dataset,clt_aer_dataname,clt_aer_modelnum = cmip_read_set(address,monthoryear,time_start,time_end,interpolateornot)

clt_aer_fulldata = np.zeros(shape=clt_aer_dataset.shape)

for i in range(len(clt_aer_dataset)):
    clt_aer_fulldata[i,:,:,:] = paramove_3d_valid_map(clt_aer_dataset[i,:,:,:])


os.chdir('/Volumes/TOSHIBA EXT/Work（2020.1.15—2020.2.08）/CMIP-GHG Work')
address = '/Volumes/TOSHIBA EXT/Work（2020.1.15—2020.2.08）/CMIP-GHG Work'
clt_ghg_dataset,clt_ghg_dataname,clt_ghg_modelnum = cmip_read_set(address,monthoryear,time_start,time_end,interpolateornot)

clt_ghg_fulldata = np.zeros(shape=clt_ghg_dataset.shape)

for i in range(len(clt_ghg_dataset)):
    clt_ghg_fulldata[i,:,:,:] = paramove_3d_valid_map(clt_ghg_dataset[i,:,:,:])

os.chdir('/Volumes/TOSHIBA EXT/Work（2020.1.15—2020.2.08）/CMIP-NAT Work')
address = '/Volumes/TOSHIBA EXT/Work（2020.1.15—2020.2.08）/CMIP-NAT Work'
clt_nat_dataset,clt_nat_dataname,clt_nat_modelnum = cmip_read_set(address,monthoryear,time_start,time_end,interpolateornot)

clt_nat_fulldata = np.zeros(shape=clt_nat_dataset.shape)

for i in range(len(clt_nat_dataset)):
    clt_nat_fulldata[i,:,:,:] = paramove_3d_valid_map(clt_nat_dataset[i,:,:,:])
"""


"""
SA_row_range = [120,240]
SA_col_range = [0,300]    

NA_row_range = [240,360]
NA_col_range = [0,300] 

AF_row_range = [120,240]
AF_col_range = [300,480] 

EU_row_range = [240,360]
EU_col_range = [300,480] 

AO_row_range = [120,240]
AO_col_range = [480,720] 

AS_row_range = [240,360]
AS_col_range = [480,720] 

TR_row_range = [120,240]
TR_col_range = [0,720] 

ML_row_range = [240,360]
ML_col_range = [0,720] 

LH_row_range = [120,360]
LH_col_range = [0,720]

mean_num = 1
valid_delete = 1

extract_obs_clisccp('TR',TR_row_range,TR_col_range)
extract_obs_clisccp('ML',ML_row_range,ML_col_range)
extract_obs_clisccp('LH',LH_row_range,LH_col_range)

clisccp_TR_list = cci_clisccp_mean_3d(clisccp_TR,SWpoint_TR,NEpoint_TR,mean_num,valid_delete)
clisccp_ML_list = cci_clisccp_mean_3d(clisccp_ML,SWpoint_ML,NEpoint_ML,mean_num,valid_delete)
clisccp_LH_list = cci_clisccp_mean_3d(clisccp_LH,SWpoint_LH,NEpoint_LH,mean_num,valid_delete)
clisccp_AS_list = cci_clisccp_mean_3d(clisccp_AS,SWpoint_AS,NEpoint_AS,mean_num,valid_delete)
clisccp_AO_list = cci_clisccp_mean_3d(clisccp_AO,SWpoint_AO,NEpoint_AO,mean_num,valid_delete)
clisccp_AF_list = cci_clisccp_mean_3d(clisccp_AF,SWpoint_AF,NEpoint_AF,mean_num,valid_delete)
clisccp_SA_list = cci_clisccp_mean_3d(clisccp_SA,SWpoint_SA,NEpoint_SA,mean_num,valid_delete)
clisccp_EU_list = cci_clisccp_mean_3d(clisccp_EU,SWpoint_EU,NEpoint_EU,mean_num,valid_delete)
clisccp_NA_list = cci_clisccp_mean_3d(clisccp_NA,SWpoint_NA,NEpoint_NA,mean_num,valid_delete)

os.chdir('/Volumes/NO NAME 1/Work/clisccp_list')
np.save('clisccp_TR_list.npy',clisccp_TR_list)
np.save('clisccp_ML_list.npy',clisccp_ML_list)
np.save('clisccp_LH_list.npy',clisccp_LH_list)
np.save('clisccp_AO_list.npy',clisccp_AO_list)
np.save('clisccp_AS_list.npy',clisccp_AS_list)
np.save('clisccp_AF_list.npy',clisccp_AF_list)
np.save('clisccp_EU_list.npy',clisccp_EU_list)
np.save('clisccp_SA_list.npy',clisccp_SA_list)
np.save('clisccp_NA_list.npy',clisccp_NA_list)

"""
#============================================================================================
#============================================================================================
# Obs cfc

"""

# tiff_data
# cci_annual_origdata_r
# check:

mapping(tiff_data)
mapping(cci_annual_origdata_r[0,:,:])

mean_num = 1

creat_obs(0,cci_annual_origdata_r,SWpoint_SA,NEpoint_SA,mean_num,'SA')
creat_obs(0,cci_annual_origdata_r,SWpoint_NA,NEpoint_NA,mean_num,'NA')
creat_obs(0,cci_annual_origdata_r,SWpoint_AS,NEpoint_AS,mean_num,'AS')
creat_obs(0,cci_annual_origdata_r,SWpoint_AO,NEpoint_AO,mean_num,'AO')
creat_obs(0,cci_annual_origdata_r,SWpoint_AF,NEpoint_AF,mean_num,'AF')
creat_obs(0,cci_annual_origdata_r,SWpoint_EU,NEpoint_EU,mean_num,'EU')
creat_obs(0,cci_annual_origdata_r,SWpoint_ML,NEpoint_ML,mean_num,'ML')
creat_obs(0,cci_annual_origdata_r,SWpoint_TR,NEpoint_TR,mean_num,'TR')
creat_obs(0,cci_annual_origdata_r,SWpoint_LH,NEpoint_LH,mean_num,'LH')

nat_NA_list,ghg_NA_list,aer_NA_list = creat_clt(0,clt_nat_origdata,clt_ghg_origdata,clt_aer_origdata,SWpoint_NA,NEpoint_NA,mean_num,'NA')
nat_EU_list,ghg_EU_list,aer_EU_list = creat_clt(0,clt_nat_origdata,clt_ghg_origdata,clt_aer_origdata,SWpoint_EU,NEpoint_EU,mean_num,'EU')
nat_AF_list,ghg_AF_list,aer_AF_list = creat_clt(0,clt_nat_origdata,clt_ghg_origdata,clt_aer_origdata,SWpoint_AF,NEpoint_AF,mean_num,'AF')
nat_AO_list,ghg_AO_list,aer_AO_list = creat_clt(0,clt_nat_origdata,clt_ghg_origdata,clt_aer_origdata,SWpoint_AO,NEpoint_AO,mean_num,'AO')
nat_AS_list,ghg_AS_list,aer_AS_list = creat_clt(0,clt_nat_origdata,clt_ghg_origdata,clt_aer_origdata,SWpoint_AS,NEpoint_AS,mean_num,'AS')
nat_SA_list,ghg_SA_list,aer_SA_list = creat_clt(0,clt_nat_origdata,clt_ghg_origdata,clt_aer_origdata,SWpoint_SA,NEpoint_SA,mean_num,'SA')
nat_ML_list,ghg_ML_list,aer_ML_list = creat_clt(0,clt_nat_origdata,clt_ghg_origdata,clt_aer_origdata,SWpoint_ML,NEpoint_ML,mean_num,'ML')
nat_TR_list,ghg_TR_list,aer_TR_list = creat_clt(0,clt_nat_origdata,clt_ghg_origdata,clt_aer_origdata,SWpoint_TR,NEpoint_TR,mean_num,'TR')
nat_LH_list,ghg_LH_list,aer_LH_list = creat_clt(0,clt_nat_origdata,clt_ghg_origdata,clt_aer_origdata,SWpoint_LH,NEpoint_LH,mean_num,'LH')


"""
#============================================================================================
#============================================================================================
# Cmip clt
"""
mean_num = 1

clt_nat_fulldata = clt_nat_origdata
clt_aer_fulldata = clt_aer_origdata
clt_ghg_fulldata = clt_ghg_origdata

nat_NA_list,ghg_NA_list,aer_NA_list = creat_clt(0,clt_nat_origdata,clt_ghg_origdata,clt_aer_origdata,SWpoint_NA,NEpoint_NA,mean_num,'NA')
nat_EU_list,ghg_EU_list,aer_EU_list = creat_clt(0,clt_nat_origdata,clt_ghg_origdata,clt_aer_origdata,SWpoint_EU,NEpoint_EU,mean_num,'EU')
nat_AF_list,ghg_AF_list,aer_AF_list = creat_clt(0,clt_nat_origdata,clt_ghg_origdata,clt_aer_origdata,SWpoint_AF,NEpoint_AF,mean_num,'AF')
nat_AO_list,ghg_AO_list,aer_AO_list = creat_clt(0,clt_nat_origdata,clt_ghg_origdata,clt_aer_origdata,SWpoint_AO,NEpoint_AO,mean_num,'AO')
nat_AS_list,ghg_AS_list,aer_AS_list = creat_clt(0,clt_nat_origdata,clt_ghg_origdata,clt_aer_origdata,SWpoint_AS,NEpoint_AS,mean_num,'AS')
nat_SA_list,ghg_SA_list,aer_SA_list = creat_clt(0,clt_nat_origdata,clt_ghg_origdata,clt_aer_origdata,SWpoint_SA,NEpoint_SA,mean_num,'SA')
nat_ML_list,ghg_ML_list,aer_ML_list = creat_clt(0,clt_nat_origdata,clt_ghg_origdata,clt_aer_origdata,SWpoint_ML,NEpoint_ML,mean_num,'ML')
nat_TR_list,ghg_TR_list,aer_TR_list = creat_clt(0,clt_nat_origdata,clt_ghg_origdata,clt_aer_origdata,SWpoint_TR,NEpoint_TR,mean_num,'TR')
nat_LH_list,ghg_LH_list,aer_LH_list = creat_clt(0,clt_nat_origdata,clt_ghg_origdata,clt_aer_origdata,SWpoint_LH,NEpoint_LH,mean_num,'LH')

"""

#============================================================================================
#pi Control
#提取数据

"""
mean_num = 1

pi_no = ['1','2','3','4','5']
monthoryear = 'year'
time_start = '198201'
time_end = '201412'
full_period = 33
interpolateornot = 0

pi_ML_list_1_5 = creat_pi(0,SWpoint_ML,NEpoint_ML,mean_num,'ML',pi_no,monthoryear,full_period,time_start,time_end,0)
pi_TR_list_1_5 = creat_pi(0,SWpoint_TR,NEpoint_TR,mean_num,'TR',pi_no,monthoryear,full_period,time_start,time_end,0)
pi_LH_list_1_5 = creat_pi(0,SWpoint_LH,NEpoint_LH,mean_num,'LH',pi_no,monthoryear,full_period,time_start,time_end,0)
pi_AS_list_1_5 = creat_pi(0,SWpoint_AS,NEpoint_AS,mean_num,'AS',pi_no,monthoryear,full_period,time_start,time_end,0)
pi_SA_list_1_5 = creat_pi(0,SWpoint_SA,NEpoint_SA,mean_num,'SA',pi_no,monthoryear,full_period,time_start,time_end,0)
pi_AO_list_1_5 = creat_pi(0,SWpoint_AO,NEpoint_AO,mean_num,'AO',pi_no,monthoryear,full_period,time_start,time_end,0)
pi_NA_list_1_5 = creat_pi(0,SWpoint_NA,NEpoint_NA,mean_num,'NA',pi_no,monthoryear,full_period,time_start,time_end,0)
pi_AF_list_1_5 = creat_pi(0,SWpoint_AF,NEpoint_AF,mean_num,'AF',pi_no,monthoryear,full_period,time_start,time_end,0)
pi_EU_list_1_5 = creat_pi(0,SWpoint_EU,NEpoint_EU,mean_num,'EU',pi_no,monthoryear,full_period,time_start,time_end,0)


save_address = '/Volumes/NO NAME 1/output/xlsx_files/pi_ML_list_1_5.xlsx'
creat_xlsx_2d(pi_ML_list_1_5,save_address)
save_address = '/Volumes/NO NAME 1/output/xlsx_files/pi_TR_list_1_5.xlsx'
creat_xlsx_2d(pi_TR_list_1_5,save_address)
save_address = '/Volumes/NO NAME 1/output/xlsx_files/pi_LH_list_1_5.xlsx'
creat_xlsx_2d(pi_LH_list_1_5,save_address)
save_address = '/Volumes/NO NAME 1/output/xlsx_files/pi_AO_list_1_5.xlsx'
creat_xlsx_2d(pi_AO_list_1_5,save_address)
save_address = '/Volumes/NO NAME 1/output/xlsx_files/pi_AF_list_1_5.xlsx'
creat_xlsx_2d(pi_AF_list_1_5,save_address)
save_address = '/Volumes/NO NAME 1/output/xlsx_files/pi_AS_list_1_5.xlsx'
creat_xlsx_2d(pi_AS_list_1_5,save_address)
save_address = '/Volumes/NO NAME 1/output/xlsx_files/pi_NA_list_1_5.xlsx'
creat_xlsx_2d(pi_NA_list_1_5,save_address)
save_address = '/Volumes/NO NAME 1/output/xlsx_files/pi_EU_list_1_5.xlsx'
creat_xlsx_2d(pi_EU_list_1_5,save_address)
save_address = '/Volumes/NO NAME 1/output/xlsx_files/pi_SA_list_1_5.xlsx'
creat_xlsx_2d(pi_SA_list_1_5,save_address)


address_1 = '/Volumes/NO NAME 1/Work/piControl_1'
monthoryear = 'year'
time_start = '198201'
time_end = '201412'
full_period = 33
interpolateornot = 0

pi_group_1,pi_fullname_set_1,pi_class_list_1 = piControl_read_set(address_1,monthoryear,full_period,time_start,time_end,interpolateornot) 

del(pi_group_1)

address_1 = '/Volumes/NO NAME 1/Work/piControl_2'

pi_group_2,pi_fullname_set_2,pi_class_list_2 = piControl_read_set(address_1,monthoryear,full_period,time_start,time_end,interpolateornot) 

del(pi_group_2)

address_1 = '/Volumes/NO NAME 1/Work/piControl_3'

pi_group_3,pi_fullname_set_3,pi_class_list_3 = piControl_read_set(address_1,monthoryear,full_period,time_start,time_end,interpolateornot) 

del(pi_group_3)

address_1 = '/Volumes/NO NAME 1/Work/piControl_4'

pi_group_4,pi_fullname_set_4,pi_class_list_4 = piControl_read_set(address_1,monthoryear,full_period,time_start,time_end,interpolateornot) 

del(pi_group_4)

address_1 = '/Volumes/NO NAME 1/Work/piControl_5'

pi_group_5,pi_fullname_set_5,pi_class_list_5 = piControl_read_set(address_1,monthoryear,full_period,time_start,time_end,interpolateornot) 

del(pi_group_5)
"""

#============================================================================================
# Obs joint histogram
    
"""

#SA: 
SWpoint_SA = [60,0]
NEpoint_SA = [120,150]

#NA:
SWpoint_NA = [120,0]
NEpoint_NA = [180,150]

#AF:
SWpoint_AF = [60,150]
NEpoint_AF = [120,240]    

#EU:
SWpoint_EU = [120,150]
NEpoint_EU = [180,240]  

#AO:
SWpoint_AO = [60,240]
NEpoint_AO = [120,360]  

#AS:
SWpoint_AS = [120,240]
NEpoint_AS = [180,360] 

#ML:
SWpoint_ML = [120,0]
NEpoint_ML = [180,360] 

#TR
SWpoint_TR = [60,0]
NEpoint_TR = [120,360] 

#LH
SWpoint_LH = [60,0]
NEpoint_LH = [180,360] 


#范围
SWpoint = [0,0]
NEpoint = [180,360]
#时间
#月数据:396m 
#年数据：33y
period = '33y'
#资料地址
address = '/Volumes/TOSHIBA EXT/Cloud_cci'
#附加数据
plus_var = 'cfc'
var = 'hist2d_cot_ctp'

ff = os.listdir(address)
ff.sort()


ff1 = ff[:96]
ff2 = ff[96:153]
ff3 = ff[153:260]
ff4 = ff[260:368]
ff5 = ff[368:392]
period1 = '8y'
period2 = '5y'
period3 = '9y'
period4 = '9y'
period5 = '2y'


os.chdir(address)

cci_clisccp_annual1 = cci_clisccp_data(ff1,address,var,plus_var,SWpoint,NEpoint,period1)

os.chdir('/Volumes/NO NAME 1/output/Base_data/cmip_clisccp')
np.save('cci_clisccp_annual1.npy',cci_clisccp_annual1)

del(cci_clisccp_annual1)

os.chdir(address)
cci_clisccp_annual2 = cci_clisccp_data(ff2,address,var,plus_var,SWpoint,NEpoint,period2)

os.chdir('/Volumes/NO NAME 1/output/Base_data/cmip_clisccp')
np.save('cci_clisccp_annual2.npy',cci_clisccp_annual2)
del(cci_clisccp_annual2)

os.chdir(address)
cci_clisccp_annual3 = cci_clisccp_data(ff3,address,var,plus_var,SWpoint,NEpoint,period3)

os.chdir('/Volumes/NO NAME 1/output/Base_data/cmip_clisccp')
np.save('cci_clisccp_annual3.npy',cci_clisccp_annual3)
del(cci_clisccp_annual3)

os.chdir(address)
cci_clisccp_annual4 = cci_clisccp_data(ff4,address,var,plus_var,SWpoint,NEpoint,period4)

os.chdir('/Volumes/NO NAME 1/output/Base_data/cmip_clisccp')
np.save('cci_clisccp_annual4.npy',cci_clisccp_annual4)
del(cci_clisccp_annual4)

os.chdir(address)
cci_clisccp_annual5 = cci_clisccp_data(ff5,address,var,plus_var,SWpoint,NEpoint,period5)

os.chdir('/Volumes/NO NAME 1/output/Base_data/cmip_clisccp')
np.save('cci_clisccp_annual5.npy',cci_clisccp_annual5)
del(cci_clisccp_annual5)

"""

"""
#++++++++++++#++++++++++++#++++++++++++#++++++++++++#++++++++++++


# pi control period number for each simulation
#1
array([ 5.,  5.,  5., 25., 10.,  2.,  2., 23., 36.,  7., 23.])
#小计：143
array([12., 30., 14.])
array([ 9., 15., 15.,  5., 16., 35.])
array([15., 15., 15., 14., 22.])
array([14., 27., 15., 15., 15.])


model_num = [5,49, 23, 43, 23, 12, 30 ,14, 9, 15, 15, 5, 16, 35, 45, 14, 22, 14, 27, 15, 15, 15]

#++++++++++++#++++++++++++#++++++++++++#++++++++++++#++++++++++++
"""

