# -*- coding: utf-8 -*-
"""
@ Time: 2020-1-3
@ author: LiangHongli
@ Mail: Helen_Liang1@outlook.com
Analyze each layer's variable, check the distribution for a better way to preprocess
2020-1-6   temporary code, dealing with some temporary issues
"""
import numpy as np
from sklearn.preprocessing import PowerTransformer,MinMaxScaler
import matplotlib.pyplot as plt
import h5py
import pandas as pd

from calc_new_predictor import var_star,layer_profile,predictors
from LUT import feature_names101
from pathlib import Path

def bar_var(var,title,xlabel):
    bins = 20
    plt.hist(var,bins)
    plt.xlabel(xlabel)
    plt.title(title)
    return

def analysis():
    fpath = Path(r'G:/DL_transmitance/revised datasets/dataset_101L_final.HDF')
    figpath = Path(r'G:\DL_transmitance\figures\profile_analysis\layer_variables_1500')
    nlv = 101
    with h5py.File(fpath,'r') as f:
        profiles = f['dependent/X'][:,:3*nlv]
        new_preds = f['dependent/predictor_RTTOV'][:]

    # X = np.concatenate((profiles,new_preds),axis=1)
    # x = profiles
    bc = MinMaxScaler()
    # bc = PowerTransformer(method='box-cox',standardize=True)
    yj = PowerTransformer(method='yeo-johnson',standardize=True)

    for k in range(profiles.shape[1]):
        x = profiles[:,k].reshape(-1,1)
        X_trans_bc = bc.fit_transform(x)
        X_trans_yj = yj.fit_transform(x)
        # print(X_trans_bc,X_trans_yj)
        titles = ['original','box-cox','yeo-johnson']
        xlabel = feature_names101[k]

        fig = plt.figure(dpi=300)
        plt.subplot(131)
        bar_var(profiles[:,k],titles[0],xlabel)
        plt.subplot(132)
        bar_var(X_trans_bc,titles[1],xlabel)
        plt.yticks([])
        plt.subplot(133)
        bar_var(X_trans_yj,titles[2],xlabel)
        plt.yticks([])
        figname = xlabel+'.png'
        plt.savefig(figpath/figname)
        plt.close(fig)

    m,n = new_preds.shape[1],new_preds.shape[2]
    for k in range(m):
        for j in range(n):
            x = new_preds[:,k,j].reshape(-1,1)
            X_trans_bc = bc.fit_transform(x)
            X_trans_yj = yj.fit_transform(x)
            # print(X_trans_bc,X_trans_yj)
            titles = ['original','box-cox','yeo-johnson']
            xlabel = 'feature'+str(k)+'_layer'+str(j)

            fig = plt.figure(dpi=300)
            plt.subplot(131)
            bar_var(new_preds[:,k,j],titles[0],xlabel)
            plt.subplot(132)
            bar_var(X_trans_bc,titles[1],xlabel)
            plt.yticks([])
            plt.subplot(133)
            bar_var(X_trans_yj,titles[2],xlabel)
            plt.yticks([])
            figname = xlabel+'.png'
            plt.savefig(figpath/figname)
            plt.close(fig)

analysis()



