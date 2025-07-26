# -*- coding: utf-8 -*-
"""
比较IASI卷积的透过率代入辐射传输方程计算的亮温和IASI卷积辐射率计算的亮温；
比较RTTOV直接计算的亮温和IASI卷积辐射率计算的亮温；验证卷积的透过率是否更接近真值
@author: LiangHongli
"""
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
import h5py
from calc_BT import bt_GBT
from LUT import ch_info,feature_names101,trans_limit

def main():
    fx = Path(r'G:\DL_transmitance\revised datasets\final\dataset_ifs101L_v1.HDF')
    figpath = Path(r'G:\DL_transmitance\figures\101L\final')
    nlv = 101
    with h5py.File(fx,'r') as f:
        bt_true = f['BT_true'][:]
        bt_rttov = f['BT_RTTOV'][:]
        Y = f['Y'][:]
        X = f['X'][:]
        trans_rttov = f['transmission_RTTOV'][:]
        emiss = f['emissivity'][:]

    print('ifs-101: land %d, ocean %d' % (sum(X[:,308]==0),sum(X[:,308]==1)))
    n = X.shape[0]
    mask = X[:,303]>=600 #2m pressure
    r = np.where(mask)[0]
    bt_true = bt_true[r,:]
    bt_rttov = bt_rttov[r,:]
    Y = Y[:,r,:]
    X = X[r,:]
    trans_rttov = trans_rttov[r,:,:]
    emiss = emiss[r,:]
    tlevel = X[:,:nlv]
    tsrf = X[:,304]
    tskin = X[:,307]
    # fsave = Path(r'G:\DL_transmitance\revised datasets\final\dataset_ifs101L_v2.HDF')
    # with h5py.File(fsave,'w') as f:
    #     f.create_dataset('X',data=X,compression='gzip')
    #     f.create_dataset('Y',data=Y,compression='gzip')
    #     f.create_dataset('BT_true',data=bt_true,compression='gzip')
    #     f.create_dataset('BT_RTTOV',data=bt_rttov,compression='gzip')
    #     f.create_dataset('transmission_RTTOV',data=trans_rttov,compression='gzip')
    #     f.create_dataset('emissivity',data=emiss,compression='gzip')

    print('ifs-101: land %d, ocean %d' % (sum(X[:,308]==0),sum(X[:,308]==1)))
    for k in range(3):
        channel = 'ch'+str(k+11)
        transsrf_true = Y[k,:,-1]
        transsrf_rttov = trans_rttov[:,k,-1]
        emissk = emiss[:,k]

        bt_conv,btair_conv = bt_GBT(Y[k,:,:-1],tlevel,emissk,tsrf,tskin,transsrf_true,
                                ch_info[channel]['scale'],ch_info[channel]['offset'],ch_info[channel]['cv'])
        bt_rttov2,btair_rttov = bt_GBT(trans_rttov[:,k,:-1],tlevel,emissk,tsrf,tskin,transsrf_rttov,
                                    ch_info[channel]['scale'],ch_info[channel]['offset'],ch_info[channel]['cv'])

        bias_conv = bt_conv-bt_true[:,k]
        bias_rttov = bt_rttov[:,k]-bt_true[:,k]
        bias = bt_rttov2-bt_rttov[:,k]
        xland = np.arange(sum(X[:,308]==0))
        xocean = np.arange(sum(X[:,308]==1))
        lineland = np.zeros(xland.shape[0])
        lineocean = np.zeros(xocean.shape[0])
        ymax = np.max(bias_rttov)
        ymin = np.min(bias_rttov)
        xmax_land = xland.shape[0]
        xmax_ocean = xocean.shape[0]
        xmin = 0
        plt.figure(dpi=300)
        plt.subplot(121)
        plt.plot(xland,bias_conv[X[:,308]==0],'b-',linewidth=1,label='BT_conv-BT_true')
        plt.plot(xland,bias_rttov[X[:,308]==0],'r-',linewidth=1,label='BT_rttov-BT_true')
        plt.plot(xland,lineland,'k-')
        plt.title(ch_info[channel]['cwl']+' land')
        plt.ylabel('bias of BT(K)')
        plt.ylim([ymin,ymax])
        plt.xlim([xmin,xmax_land])
        plt.subplot(122)
        plt.plot(xocean,bias_conv[X[:,308]==1],'b-',linewidth=1,label='BT_conv-BT_true')
        plt.plot(xocean,bias_rttov[X[:,308]==1],'r-',linewidth=1,label='BT_rttov-BT_true')
        plt.plot(xocean,lineocean,'k-')
        plt.ylim([ymin,ymax])
        plt.xlim([xmin,xmax_ocean])
        plt.yticks([])
        plt.legend(loc='lower center')
        plt.title(ch_info[channel]['cwl']+' ocean')
        figname = channel+'_validate_iasi_conv_trans.png'
        plt.savefig(figpath/figname)
        plt.close()

        ymax = np.max(bias)
        ymin = np.min(bias)
        print([ymin,ymax])
        plt.figure(dpi=300)
        plt.subplot(121)
        plt.plot(xland,bias[X[:,308]==0],'g-',linewidth=1)
        plt.plot(xland,lineland,'k-')
        plt.title(ch_info[channel]['cwl']+' land')
        plt.ylabel('bias of BT(K)')
        plt.ylim([ymin,ymax])
        plt.xlim([xmin,xmax_land])
        plt.subplot(122)
        plt.plot(xocean,bias[X[:,308]==1],'g-',linewidth=1)
        plt.plot(xocean,lineocean,'k-')
        plt.ylim([ymin,ymax])
        plt.xlim([xmin,xmax_ocean])
        plt.yticks([])
        plt.title(ch_info[channel]['cwl']+' ocean')
        figname = channel+'_validate_code.png'
        plt.savefig(figpath/figname)
        plt.close()

    # index = [679,684,692,1666]
    # plt.plot(trans_rttov[])

if __name__ == '__main__':
    main()
