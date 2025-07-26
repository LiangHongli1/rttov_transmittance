# -*- coding: utf-8 -*-
"""
@ Time: 2019/9/25
@ author: LiangHongli
@ Mail: Helen_Liang1@outlook.com
"""

import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np
import matplotlib as mpl
from sklearn.linear_model import LinearRegression

def plot_profile_var(var,pressure,var_name):
    '''
    plotting T,q,et al. variables of the profile,including the mean, maximum and minimum, 95 percentiles.
    :param var: tensor, shape=(nprofile, nlevel)
    :param pressure: vector, atmospheric pressure, y axis
    :param var_name: string
    :return: figure
    '''
    var_mean = np.mean(var,axis=0)
    var_percentile = np.percentile(var,[10,90,25,75,0.3,99.7],axis=0)
    var_max = np.max(var,axis=0)
    var_min = np.min(var,axis=0)
    # print(pressure.dtype,var_max.dtype,var_min.dtype)

    plt.figure(figsize=(8,6),dpi=300)
    plt.fill_betweenx(pressure, var_min, var_max,color='gray')
    plt.fill_betweenx(pressure, var_percentile[4,:], var_percentile[5,:],color='blue')
    plt.fill_betweenx(pressure, var_percentile[0,:], var_percentile[1,:],color='orange')
    plt.fill_betweenx(pressure, var_percentile[2,:], var_percentile[3,:],color='r')
    plt.plot(var_mean,pressure,'k-',linewidth=1)

    plt.xlabel(var_name)
    plt.ylabel('Pressure(hPa)')
    plt.ylim(0.05,10**3)
    ax = plt.gca()
    ax.invert_yaxis()
    plt.yscale('log',basey=10)
    if np.mean(var_max)<0.1:
        plt.xscale('log',basex=10)
    plt.tick_params(labelsize=11)

    return

def bar_profile(y):
    '''
    Bin the number of profiles monthly
    :param y: sequence, numbers of profile for each month
    :return: figure object
    '''
    x = np.arange(1,13)
    label_list = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    plt.bar(x,y,width=0.6,alpha=0.8,color='orange')
    plt.ylabel('Number of profile')
    plt.xticks(x,label_list)
    return

def bar_train_test_pro(ytrain,ytest):
    '''
    Bar the number of profiles monthly according to training and testing dataset
    :param ytrain: sequence, numbers of profile for each month in training set
    :param ytest: sequence, numbers of profile for each month in testing set
    :return: figure
    '''
    x = np.arange(1,13)
    label_list = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    plt.bar(x, ytrain, width=0.6, label='Training set', fc='blue')
    plt.bar(x,ytest,bottom=ytrain,width=0.6, label='Testing set', fc='red')
    plt.ylabel('Number of profile')
    plt.xticks(x, label_list)
    plt.legend()
    return

def plot_srf(x,y,title):
    plt.plot(x,y,'b-',linewidth=1.2)
    plt.xlabel('Wave Number($cm^{-1}$)')
    plt.ylabel('Spectral Response Function')
    plt.title(title)
    return

def plot_bt(x,y1,y2,y3,title):
    '''
    画亮温
    :param x: index of profile
    :param y1: BT true
    :param y2: BT GBT_preds
    :param y3: BT RTTOV
    :return: figure
    '''
    plt.plot(x,y1,'k-',x,y2,'b--',x,y3,'r--',linewidth=1.2)
    plt.xlabel('profile',fontsize=12)
    plt.ylabel('Brightness Temperature(K)',fontsize=12)
    plt.legend(['True','GBT','RTTOV'])
    plt.xlim([0,x[-1]])
    plt.title(title)
    plt.grid(True)
    plt.grid(color='k', linestyle='--', linewidth=0.5)
    return

def plot_bias_bt(x,y1,y2,title,method):
    '''
    Plot bias: bt_GBT-bt_true, bt_RTTOV-bt_true
    :param x: the index of profile number, array
    :param y1: bt_GBT-bt_true, array
    :param y2: bt_RTTOV-bt_true, array
    :return: figure object
    '''
    plt.plot(x,y1,'o',x,y2,'*',linewidth=0.8)
    plt.xlabel('Profile',fontsize=12)
    plt.ylabel('Deviation of BT(K)',fontsize=12)
    plt.legend([method,'RTTOV'])
    plt.xlim([0, x[-1]])
    plt.title(title)
    plt.grid(True)
    plt.grid(color='k',linestyle='--',linewidth=0.5)
    return

def dot_error_bt(y_gbt,y_rttov,title,ylabel):
    x = np.array([1,2,3])
    xticks = ['7.43μm','7.33μm','6.52μm']
    plt.plot(x,y_gbt,'o--',label='GBT')
    plt.plot(x,y_rttov,'*--',label='RTTOV')
    plt.legend()
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(x,xticks)
    plt.grid(True)
    plt.grid(color='k',linestyle='--',linewidth=0.5)
    return

def dot_error_bt_method(y_gbt,y_xgboost,y_rf,title,ylabel):
    x = np.array([1,2,3])
    xticks = ['7.43μm','7.33μm','6.52μm']
    plt.plot(x,y_gbt,'o--',label='GBT')
    plt.plot(x,y_xgboost,'o--',label='XGBoost')
    plt.plot(x,y_rf,'o--',label='RF')
    # plt.plot(x,y_rttov,'*--',label='RTTOV')
    plt.legend()
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(x,xticks)
    plt.grid(True)
    plt.grid(color='k',linestyle='--',linewidth=0.5)
    return

##########################################################################
def plot_error_level(x1,x2,x3,x4,p,title,method):
    '''
    Plot mean bias and RMSE of every level
    :param x1: bias of GBT, array
    :param x2: rmse of GBT, array
    :param x3: bias of RTTOV, array
    :param x4: rmse of RTTOV, array
    :param p: array, pressure of every level
    :param title: string. title of the figure
    :return:
    '''
    plt.plot(x1,p,'-',linewidth=1.2,label='bias_'+method)
    plt.plot(x2,p,'--',linewidth=1.2,label='RMSE_'+method)
    plt.plot(x3,p,'-',linewidth=1.2,label='bias_RTTOV')
    plt.plot(x4,p,'--',linewidth=1.2,label='RMSE_RTTOV')
    plt.title(title)
    plt.xlabel('Bias and RMSE of transmission')
    plt.ylabel('Pressure(hPa)')
    plt.ylim([1,p[-1]])
    plt.legend()
    # plt.yscale('log',basey=10)
    ax = plt.gca()
    ax.invert_yaxis()
    ax.xaxis.get_major_formatter().set_powerlimits((0,3))
    plt.grid(True)
    plt.grid(color='k', linestyle='--', linewidth=0.5)
    return

def plot_bias_ratio_lev(ratio_gbt,ratio_rttov,p,title,method):
    plt.plot(ratio_gbt,p,'-',linewidth=1.2,label=method)
    plt.plot(ratio_rttov,p,'-',linewidth=1.2,label='RTTOV')
    plt.title(title)
    plt.xlabel('bias/transmittance')
    plt.ylabel('Pressure(hPa)')
    plt.ylim([1,p[-1]])
    plt.legend()
    # plt.yscale('log',basey=10)
    ax = plt.gca()
    ax.invert_yaxis()
    ax.xaxis.get_major_formatter().set_powerlimits((0,3))
    plt.grid(True)
    plt.grid(color='k', linestyle='--', linewidth=0.5)
    return

def plot_std_lev(std_gbt,std_rttov,p,title,method):
    plt.plot(std_gbt,p,'-',linewidth=1.2,label=method)
    plt.plot(std_rttov,p,'-',linewidth=1.2,label='RTTOV')
    plt.title(title)
    plt.xlabel('std')
    plt.ylabel('Pressure(hPa)')
    plt.ylim([1,p[-1]])
    plt.legend()
    # plt.yscale('log',basey=10)
    ax = plt.gca()
    ax.invert_yaxis()
    ax.xaxis.get_major_formatter().set_powerlimits((0,3))
    plt.grid(True)
    plt.grid(color='k', linestyle='--', linewidth=0.5)
    return

def plot_trans_level(x1,x2,x3,p,xlabel,title,method):
    '''
    Plot mean bias and RMSE of every level
    :param x1: transmittance ground-truth, array
    :param x2: transmittance computed by GBT, array
    :param x3: transmittance computed by RTTOV, array
    :param p: array, pressure of every level
    :param title: string. title of the figure
    :return:
    '''
    plt.plot(x1, p, 'k-', linewidth=1.2, label='True')
    plt.plot(x2, p, '--', linewidth=1.2, label=method)
    plt.plot(x3, p, '--', linewidth=1.2, label='RTTOV')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('Pressure(hPa)')
    plt.ylim([50,1100])
    # plt.legend()
    # plt.yscale('log', basey=10)
    ax = plt.gca()
    ax.invert_yaxis()
    ax.xaxis.get_major_formatter().set_powerlimits((0, 3))
    plt.grid(True)
    plt.grid(color='k', linestyle='--', linewidth=0.5)
    return

def plot_mre(x1,x2,title,method):
    bins = np.linspace(0,1,10)
    numlist1 = []
    numlist2 = []
    ticklabels = ['{:.2f}'.format(bins[0])]
    for k in range(1,len(bins)):
        mask1 = np.logical_and(x1>=bins[k-1],x1<bins[k])
        mask2 = np.logical_and(x2>=bins[k-1],x2<bins[k])
        numlist1.append(len(x1[mask1]))
        numlist2.append(len(x2[mask2]))
        ticklabels.append('{:.2f}'.format(bins[k]))
    plt.bar(bins[1:],height=numlist1,width=0.03,alpha=0.8,color='blue',label='MRE_'+method)
    plt.bar(bins[1:]+0.03, height=numlist2, width=0.03, alpha=0.8, color='red', label='MRE_RTTOV')
    plt.xticks(bins+0.015,ticklabels)
    plt.xlabel('MRE of each profile')
    plt.ylabel('Profile numbers')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.grid(color='k', linestyle='--', linewidth=0.5)
    return

# 
def scatter_line_trans(x1,x2,x3,title,MAE_GBT,RMSE_GBT,MSLE_GBT,MAE_RTTOV,RMSE_RTTOV,MSLE_RTTOV,method):
    x1 = x1.T.reshape(-1)
    x2 = x2.T.reshape(-1)
    x3 = x3.T.reshape(-1)
    nx = np.arange(len(x1))
    # p1 = np.polyfit(x1,x2,1)
    # p2 = np.polyfit(x1,x3,1)
    plt.scatter(x1,x2,s=1,label=method)
    plt.scatter(x1,x3,s=1,label='RTTOV')
    plt.plot(x1,x1,'k-',linewidth=1.2)
    plt.plot(nx,x1,'k-',nx,x2,'r--',linewidth=1.2)
    plt.title(title)
    # txt1 = 'y_'+method+'='+'{:.5f}'.format(p1[0])+'y_true'+'{:.5f}'.format(p1[1])
    # plt.text(0.02,0.94,txt1)
    # txt2 = 'y_rttov='+'{:.5f}'.format(p2[0])+'y_true'+'{:.5f}'.format(p2[1])
    # plt.text(0.02,0.88,txt1)
    plt.text(0.02,0.94,'MAE_ML={:.4f} MAE_RTTOV={:.4f}'.format(MAE_GBT,MAE_RTTOV))
    plt.text(0.02, 0.88, 'RMSE_ML={:.4f} RMSE_RTTOV={:.4f}'.format(RMSE_GBT, RMSE_RTTOV))
    plt.text(0.02, 0.82, 'MSLE_ML={:.4f} MSLE_RTTOV={:.4f}'.format(MSLE_GBT, MSLE_RTTOV))
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.xlabel('Convolved IASI transmittance(true)')
    plt.ylabel('Transmittance predicted by GBT and RTTOV')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.grid(color='k', linestyle='--', linewidth=0.5)
    return


def plot_single_var(x1,p,xlabel,title):
    plt.plot(x1,p,linewidth=1.2)
    plt.yscale('log',basey=10)
    ax = plt.gca()
    ax.invert_yaxis()
    # ax.xaxis.get_major_formatter().set_powerlimits((0,3))
    plt.xlabel(xlabel)
    plt.ylabel('Pressure(hPa)')
    plt.title(title)
    if np.mean(np.max(x1))<0.1:
        plt.xscale('log',basex=10)
    plt.tick_params(labelsize=11)
    return

def bar_feature_rank(weight,name,title):
    plt.barh(range(len(weight)),width=weight,height=0.2,tick_label=name)
    plt.xlabel('Weight of the feature')
    plt.title(title)
    return
#########################################################################################

def map_scatter(lons,lats,data,boundary,title,vmin,vmax,cb_label=None,yticks=True):
    '''
    mapping the information of data through projection
    :param lats: tensor,vector or matrix, latitude
    :param lons: tensor,vector or matrix, longitude
    :param data: tensor, or empty, data to be mapped
    :param boundary: vector,shape=(4,), boundary=[llcrnrlat,urcrnrlat,llcrnrlon,urcrnrlon]
    :return: figure
    '''
    m = Basemap(llcrnrlat=boundary[0],urcrnrlat=boundary[1],llcrnrlon=boundary[-2],urcrnrlon=boundary[-1],lat_0=0.)
    # meridians = np.array([int(np.min(lons)+4),int(np.max(lons)-3)])
    meridians = np.arange(-180,180,60)
    # parallels = np.arange(-30,31,10)
    parallels = np.arange(-90,91,30)
    m.drawmeridians(meridians,labels=[0,0,0,1],color='k',linewidth=0.8)
    m.drawparallels(parallels,labels=[1,0,0,0],color='k',linewidth=0.8)
    m.drawcoastlines(linewidth=0.7,color='k')

    cmap = mpl.cm.get_cmap('jet')
    cmap.set_bad(color='k', alpha=1)
    # if len(data.shape)<2:
    #     m.scatter(lons,lats,latlon=True,s=3,c=data,marker=',')
    # elif len(data.shape)>1:
        # m.pcolormesh(lons,lats,data,cmap=cmap,vmin=210,vmax=290,latlon=True,shading='gouraud')
    m.scatter(lons,lats,latlon=True,s=1,c=data,cmap=cmap,marker=',',vmin=vmin,vmax=vmax)
    # if cb_label!=None:
    # cb = m.colorbar(location='bottom',pad=0.1,fraction=0.8,aspect=9,spacing='uniform')
        # cb.set_label(cb_label,fontsize=10)
    # else:
    #     pass
    plt.title(title,fontsize=10)
    if yticks==False:
        plt.yticks([])
    plt.tick_params(labelsize=8)
    return
# vmin=210,vmax=290,
def map_dot(lons,lats,boundary,title,color,marker):
    '''
    Mapping the distribution of the profiles
    :param lons: array
    :param lats: aray
    :param boundary: list or array,shape=(4,), boundary=[llcrnrlat,urcrnrlat,llcrnrlon,urcrnrlon]
    :param title: string
    :return: figure
    '''
    m = Basemap(llcrnrlat=boundary[0], urcrnrlat=boundary[1], llcrnrlon=boundary[-2], urcrnrlon=boundary[-1], lat_0=0.)
    meridians = np.arange(-180, 180, 60)
    parallels = np.arange(-90, 91, 30)
    m.drawmeridians(meridians, labels=[0, 0, 0, 1], color='k', linewidth=0.8)
    m.drawparallels(parallels, labels=[1, 0, 0, 0], color='k', linewidth=0.8)
    m.drawcoastlines(linewidth=0.7, color='k')
    m.scatter(lons, lats, latlon=True, s=2, c=color, marker=marker)
    plt.title(title)
    plt.tick_params(labelsize=11)
    return
