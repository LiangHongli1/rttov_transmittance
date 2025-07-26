'''
for drawing the BT validation results
'''

import numpy as np
import h5py
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from LUT import ch_info
from plot_map_funcs import map_scatter

def main():
    fx = Path(r'G:\DL_transmitance\O-B_validation\X4O-B-20080714.HDF')
    fbt_rttov = Path(r'G:\DL_transmitance\O-B_validation\bt_rttov_200807140006.xlsx')
    fbt_ml = Path(r'G:\DL_transmitance\O-B_validation\bt_predicted_by_ml.HDF')
    figpath = Path(r'G:\DL_transmitance\figures\101L\o-b')
    
    ########################### read data ########################
    with h5py.File(fx,'r') as f:
        obs_bt = f['BT_obs'][:]  # shape = (sample,channel)
        
    with h5py.File(fbt_ml,'r') as f:
        bt_ml = f['bt'][:]  # shape = (sample,channel,method)

    lon_lat = obs_bt[:,-2:]
    df_bt = pd.read_excel(fbt_rttov,sheet_name='bt')
    bt_rttov = df_bt.loc[:,['ch11','ch12','ch13']].values # shape = (sample,channel)
    ################### end ###############################################
    for k in range(3):
        channel = 'ch'+str(k+11)
        index = np.argsort(obs_bt[:,k]) # ascending
        biask_rttov = bt_rttov[index,k]-obs_bt[index,k]
        biask_gbt = bt_ml[index,k,0]-obs_bt[index,k]
        biask_xgb = bt_ml[index,k,1]-obs_bt[index,k]
        biask_rf = bt_ml[index,k,2]-obs_bt[index,k]
        
        floor = np.floor(np.min(obs_bt[:,k]))
        ceil = np.ceil(np.max(obs_bt[:,k]))
        steps = np.arange(floor,ceil+1,0.5)
        xk = steps+0.25
        xk = xk[:-1]
        n = len(steps)
        # print(xk,n)
        mean_bias_rttov = np.zeros(n-1)
        mean_bias_gbt = np.zeros(n-1)
        mean_bias_xgb = np.zeros(n-1)
        mean_bias_rf = np.zeros(n-1)
        for j in range(1,n):
            mask = np.logical_and(obs_bt[index,k]>=steps[j-1],obs_bt[index,k]<steps[j])
            index2 = np.where(mask)[0]
            mean_bias_rttov[j-1] = np.mean(biask_rttov[index2])
            mean_bias_gbt[j-1] = np.mean(biask_gbt[index2])
            mean_bias_xgb[j-1] = np.mean(biask_xgb[mask])
            mean_bias_rf[j-1] = np.mean(biask_rf[mask])
            
        # print(mean_bias_rttov)
        plt.figure(dpi=300)
        if k<2:
            xlim = [255,280]
            ylim = [-4,4]
        else:
            xlim = [235,260]
            ylim = [-12,2]
        
        plt.plot(xk[1:],mean_bias_rttov[1:],'ko-',label='RTTOV',markersize=3.5)
        plt.plot(xk[1:],mean_bias_gbt[1:],'--',label='GBT')
        plt.plot(xk[1:],mean_bias_xgb[1:],'--',label='XGBoost')
        plt.plot(xk[1:],mean_bias_rf[1:],'--',label='RF')
        plt.plot(xk[1:],np.zeros(len(xk[1:])),'-m')
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.legend()
        plt.xlabel('Scene BT(K)')
        plt.ylabel('Mean bias of BT(K)')
        plt.grid(True)
        plt.title(ch_info[channel]['cwl'])
        figname = channel+'bias_of_BT0.5k.png'
        plt.savefig(figpath/figname)
        plt.close()
        ####################################################
        # xlim = [np.min(biask_rttov),np.max(biask_rttov)]
        ylim = [0,0.55]
        
        if k==0:
            x = 0.7
            y = 0.37
            xlim = [-4,4]
        elif k==1:
            x = 0.7
            y = 0.37
            xlim = [-4,4]
        else:
            x = -9
            y = 0.48
            xlim = [-10,2]
            
        plt.figure(dpi=300)
        plt.subplot(2,2,1)
        title = 'RTTOV - obs'
        plt.hist(biask_rttov,bins=n-1,histtype='stepfilled',density=True)
        sns.kdeplot(biask_rttov)
        median = np.median(biask_rttov)
        plt.plot(median,0.01,'ko')
        plt.plot([0,0],[0,0.55],'m--')
        plt.ylabel('Probability')
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.text(x,0.42,title,fontsize=9)
        plt.text(x-0.1,y,'median: {:.2f}'.format(median),fontsize=9,color='r')
        
        plt.subplot(2,2,2)
        title = 'GBT - obs' #,'Bias of Brightness Temperature(K)'
        plt.hist(biask_gbt,bins=n-1,histtype='stepfilled',density=True)
        sns.kdeplot(biask_gbt)
        median = np.median(biask_gbt)
        plt.plot(median,0.01,'ko')
        plt.plot([0,0],[0,0.55],'m--')
        plt.yticks([])
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.text(x,0.42,title,fontsize=9)
        plt.text(x-0.1,y,'median: {:.2f}'.format(median),fontsize=9,color='r')
        
        plt.subplot(2,2,3)
        title = 'XGBoost - obs'
        plt.hist(biask_xgb,bins=n-1,histtype='stepfilled',density=True)
        sns.kdeplot(biask_xgb)
        median = np.median(biask_xgb)
        plt.plot([0,0],[0,0.55],'m--')
        plt.plot(median,0.01,'ko')
        plt.xlabel('Bias of BT(K)')
        plt.ylabel('Probability')
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.text(x,0.42,title,fontsize=9)
        plt.text(x-0.1,y,'median: {:.2f}'.format(median),fontsize=9,color='r')
        
        plt.subplot(2,2,4)
        title = 'RF - obs'
        plt.hist(biask_rf,bins=n-1,histtype='stepfilled',density=True)
        sns.kdeplot(biask_rf)
        median = np.median(biask_rf)
        plt.plot(median,0.01,'ko')
        plt.plot([0,0],[0,0.55],'m--')
        plt.xlabel('Bias of BT(K)')
        plt.yticks([])
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.text(x,0.42,title,fontsize=9)
        plt.text(x-0.1,y,'median: {:.2f}'.format(median),fontsize=9,color='r')
        
        plt.suptitle(ch_info[channel]['cwl'])
        
        figname = 'GBT_20080714_bias_PDF'+channel+'.png'
        plt.savefig(figpath/figname)
        plt.close()
    
    return

if __name__=='__main__':
    main()