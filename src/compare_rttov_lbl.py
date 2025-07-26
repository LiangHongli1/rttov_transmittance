'''
Concatenate the data, compare the LBLRTM, IASI convolved and RTTOV transmittance
2019-12-23
LiangHongli
'''
import h5py
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_log_error,mean_squared_error
from pathlib import Path
from plot_map_funcs import plot_trans_level
import pandas as pd

ifs137_ch110 = r'G:\DL_transmitance\revised datasets\dataset_IFS137_ch1_ch10.HDF'
ifs137_ch1215 = r'G:\DL_transmitance\revised datasets\dataset_IFS137_ch12_ch15.HDF'
ifs137_ch110_v7 = r'G:\DL_transmitance\revised datasets\dataset_IFS137_v7_ch1_ch10.HDF'
ifs137_ch1215_v7 = r'G:\DL_transmitance\revised datasets\dataset_IFS137_v7_ch12_ch15.HDF'
ec_ch110_v7 = r'G:\DL_transmitance\revised datasets\EC83_v7_ch1_ch10.HDF'
ec_ch1215_v7 = r'G:\DL_transmitance\revised datasets\EC83_v7_ch12_ch15.HDF'
ec_ch110_v9 = r'G:\DL_transmitance\revised datasets\EC83_v9_ch1_ch10.HDF'
ec_ch1215_v9 = r'G:\DL_transmitance\revised datasets\EC83_v9_ch12_ch15.HDF'

ec_lbl = r'G:\DL_transmitance\revised datasets\trans_EC83_LBL.HDF'
ec_lbl2 = r'G:\DL_transmitance\revised datasets\ec83_LBL.HDF'
ec_lbl3 = r'G:\DL_transmitance\revised datasets\ec_IRAS_trans_ch1-10.xlsx'
########## Concatenate the same version data of different channel to one single file ##########
npro = 83

def plot_error_level(x1,x2,x3,x4,p,title):
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
    plt.plot(x1,p,'b-',linewidth=1.2,label='bias_conv')
    plt.plot(x2,p,'b--',linewidth=1.2,label='RMSE_conv')
    plt.plot(x3,p,'r-',linewidth=1.2,label='bias_RTTOV')
    plt.plot(x4,p,'r--',linewidth=1.2,label='RMSE_RTTOV')
    plt.title(title)
    plt.xlabel('Bias and RMSE of transmittance')
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


def scatter_line_trans(x1,x2,x3,title,MAE_GBT,RMSE_GBT,MSLE_GBT,MAE_RTTOV,RMSE_RTTOV,MSLE_RTTOV):
    x1 = x1.T.reshape(-1)
    x2 = x2.T.reshape(-1)
    x3 = x3.T.reshape(-1)
    n = np.arange(len(x1))
    plt.scatter(x1,x2,s=1,c='b',label='concolved from IASI')
    plt.scatter(x1,x3,s=1,c='r',label='RTTOV')
    plt.plot(x1,x1,'k-',linewidth=1.2)
    plt.plot(n,x1,'k-',n,x2,'r--',linewidth=1.2)
    plt.title(title)
    plt.text(0.02,0.94,'MAE_conv={:.4f} MAE_RTTOV={:.4f}'.format(MAE_GBT,MAE_RTTOV))
    plt.text(0.02, 0.88, 'RMSE_conv={:.4f} RMSE_RTTOV={:.4f}'.format(RMSE_GBT, RMSE_RTTOV))
    plt.text(0.02, 0.82, 'MSLE_conv={:.4f} MSLE_RTTOV={:.4f}'.format(MSLE_GBT, MSLE_RTTOV))
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.xlabel('Convolved transmittance from LBLRTM')
    plt.ylabel('Convolved transmittance from IASI(blue), RTTOV(red)')
    plt.legend(loc='lower right')
    return

########## Compare the first 10 channels between RTTOV and LBLRTM ################
# with h5py.File(ec_lbl,'r') as f:
#     trans_lbl = f['transmittance'][:] #shape=[83,101,10]

with h5py.File(ec_ch110_v7,'r') as f:
    trans_convolve_v7 = f['Y'][:9,:,:-1] #shape=[9,83,101], 9 channels(no 2nd channel)
    trans_rttov_v7 = f['transmission_RTTOV'][:,:10,:-1] # shape=[83,10,101]
    p = f['X'][0,:101]

with h5py.File(ec_ch110_v9,'r') as f:
    trans_convolve_v9 = f['Y'][:9,:,:-1]
    trans_rttov_v9 = f['transmission_RTTOV'][:,:10,:-1]

# with h5py.File(ec_lbl2,'r') as f:
#     od_layer = f['od_single_layer'][:]
#     od_space = f['od_layer2space'][:]

trans_lbl_origin = pd.read_excel(ec_lbl3,sheet_name='ch1-10')
o3_trans = trans_lbl_origin.loc[:,['o31','o33','o34','o35','o36','o37','o38','o39','o310']]
o3_trans = np.array(o3_trans.values,dtype=float)
trans_lbl = np.zeros((83,100,9))
for k in range(9):
    trans_lbl[:,:,k] = o3_trans[:,k].reshape(-1,100)

from LUT import ch_info

trans_convolve_v7 = np.transpose(trans_convolve_v7,[1,2,0])
trans_rttov_v7 = np.transpose(trans_rttov_v7,[0,2,1])
trans_convolve_v9 = np.transpose(trans_convolve_v9,[1,2,0])
trans_rttov_v9 = np.transpose(trans_rttov_v9,[0,2,1])
# od_layer = np.transpose(od_layer,[2,1,0])
# od_space = np.transpose(od_space,[2,1,0])

# trans_lbl = np.zeros((83,101,10),dtype=np.float)
# trans_lbl[:,0,:] = 1
# trans_lbl[:,1:,:] = np.exp(-od_space)
# trans_lbl = np.exp(-od_layer)
trans_rttov_v7 = np.concatenate((trans_rttov_v7[:,:,0].reshape(83,101,1),trans_rttov_v7[:,:,2:]),axis=2)
trans_rttov_v9 = np.concatenate((trans_rttov_v9[:,:,0].reshape(83,101,1),trans_rttov_v9[:,:,2:]),axis=2)
# trans_lbl = np.concatenate((trans_lbl[:,:,0].reshape(83,100,1),trans_lbl[:,:,2:]),axis=2)
# od_layer = np.concatenate((od_layer[:,:,0].reshape(83,100,1),od_layer[:,:,2:]),axis=2)

######## calculate single layer's optical depth ##########
# conv_od_layer = np.zeros((83,100,9),dtype=np.float)
# rttov_od_layer = np.zeros((83,100,9),dtype=np.float)
# for j in range(1,101):
#     conv_od_layer[:,j-1,:] = np.log(trans_convolve_v9[:,j-1,:])-np.log(trans_convolve_v9[:,j,:])
#     rttov_od_layer[:,j-1,:] = np.log(trans_rttov_v9[:,j-1,:])-np.log(trans_rttov_v9[:,j,:])

# trans_convolve_v9 = np.exp(-conv_od_layer)
# trans_rttov_v9 = np.exp(-rttov_od_layer)
trans_convolve_v9 = trans_convolve_v9[:,1:,:]
trans_rttov_v9 = trans_rttov_v9[:,1:,:]
# print(conv_od_layer,rttov_od_layer)
#######################################################################################
# figpath = Path(r'G:\DL_transmitance\figures\RTTOV vs LBLRTM\conv_od2trans')
figpath = Path(r'G:\DL_transmitance\figures\RTTOV vs LBLRTM\assumption1')
for k in range(9):
    # mae_convol7 = mean_absolute_error(trans_lbl[:,:,k],trans_convolve_v7[:,:,k])
    # rmse_convol7 = mean_squared_error(trans_lbl[:,:,k],trans_convolve_v7[:,:,k])**0.5
    # msle_convol7 = mean_squared_log_error(trans_lbl[:,:,k],1+trans_convolve_v7[:,:,k])
    mae_convol9 = mean_absolute_error(trans_lbl[:,:,k],trans_convolve_v9[:,:,k])
    rmse_convol9 = mean_squared_error(trans_lbl[:,:,k],trans_convolve_v9[:,:,k])**0.5
    msle_convol9 = mean_squared_log_error(1+trans_lbl[:,:,k],1+trans_convolve_v9[:,:,k])
    #
    # mae_rttov7 = mean_absolute_error(trans_lbl[:,:,k],trans_rttov_v7[:,:,k])
    # rmse_rttov7 = mean_squared_error(trans_lbl[:,:,k],trans_rttov_v7[:,:,k])**0.5
    # msle_rttov7 = mean_squared_log_error(trans_lbl[:,:,k],1+trans_rttov_v7[:,:,k])
    mae_rttov9 = mean_absolute_error(trans_lbl[:,:,k],trans_rttov_v9[:,:,k])
    rmse_rttov9 = mean_squared_error(trans_lbl[:,:,k],trans_rttov_v9[:,:,k])**0.5
    msle_rttov9 = mean_squared_log_error(1+trans_lbl[:,:,k],1+trans_rttov_v9[:,:,k])
    #
    # conv7_bias_levs = np.mean(trans_convolve_v7[:,:,k]-trans_lbl[:,:,k],axis=0)
    # conv7_rmse_levs = (np.sum((trans_convolve_v7[:,:,k]-trans_lbl[:,:,k])**2,axis=0)/npro)**0.5
    # conv9_bias_levs = np.mean(trans_convolve_v9[:,:,k]-trans_lbl[:,:,k],axis=0)
    # conv9_rmse_levs = (np.sum((trans_convolve_v9[:,:,k]-trans_lbl[:,:,k])**2,axis=0)/npro)**0.5
    # rttov7_bias_levs = np.mean(trans_rttov_v7[:,:,k]-trans_lbl[:,:,k],axis=0)
    # rttov7_rmse_levs = (np.sum((trans_rttov_v7[:,:,k]-trans_lbl[:,:,k])**2,axis=0)/npro)**0.5
    # rttov9_bias_levs = np.mean(trans_rttov_v9[:,:,k]-trans_lbl[:,:,k],axis=0)
    # rttov9_rmse_levs = (np.sum((trans_rttov_v9[:,:,k]-trans_lbl[:,:,k])**2,axis=0)/npro)**0.5
    if k==0:
        ch = 'ch1'
    else:
        ch = 'ch'+str(k+2)

    # title = ch_info[ch]['cwl'] + ' v7 redictors'
    # plt.figure(dpi=300)
    # scatter_line_trans(trans_lbl[:,:,k],trans_convolve_v7[:,:,k],trans_rttov_v7[:,:,k],title,\
    #                    mae_convol7,rmse_convol7,msle_convol7,\
    #                    mae_rttov7,rmse_rttov7,msle_rttov7)
    # figname = title+'.png'
    # plt.savefig(figpath/figname)
    #
    # plt.figure(dpi=300)
    # plot_error_level(conv7_bias_levs,conv7_rmse_levs,rttov7_bias_levs,rttov7_rmse_levs,p,title)
    # figname = title+'_error_level'+'.png'
    # plt.savefig(figpath/figname)

    title = ch_info[ch]['cwl'] + ' v9 redictors'
    plt.figure(dpi=300)
    scatter_line_trans(trans_lbl[:,:,k],trans_convolve_v9[:,:,k],trans_rttov_v9[:,:,k],title,\
                       mae_convol9,rmse_convol9,msle_convol9,\
                       mae_rttov9,rmse_rttov9,msle_rttov9)
    figname = title+'.png'
    plt.savefig(figpath/figname)

    plt.figure(dpi=300)
    xlabel = 'transmittance'
    # plot_error_level(conv9_bias_levs,conv9_rmse_levs,rttov9_bias_levs,rttov9_rmse_levs,p,title)
    plot_trans_level(trans_lbl[0,:,k],trans_convolve_v9[0,:,k],trans_rttov_v9[0,:,k],p[1:],xlabel,title)
    figname = 'transmittance'+ch+'.png'
    plt.savefig(figpath/figname)


    ########### Plot the optical depth of single layer ############################################################3
    # print(od_layer.shape,conv_od_layer.shape)
    # mae_convol9 = mean_absolute_error(od_layer[:,:,k],conv_od_layer[:,:,k])
    # rmse_convol9 = mean_squared_error(od_layer[:,:,k],conv_od_layer[:,:,k])**0.5
    # msle_convol9 = mean_squared_log_error(od_layer[:,:,k],1+conv_od_layer[:,:,k])
    # mae_rttov9 = mean_absolute_error(od_layer[:,:,k],rttov_od_layer[:,:,k])
    # rmse_rttov9 = mean_squared_error(od_layer[:,:,k],rttov_od_layer[:,:,k])**0.5
    # msle_rttov9 = mean_squared_log_error(od_layer[:,:,k],1+rttov_od_layer[:,:,k])
    #
    # conv9_bias_levs = np.mean(conv_od_layer[:,:,k]-od_layer[:,:,k],axis=0)
    # conv9_rmse_levs = (np.sum((conv_od_layer[:,:,k]-od_layer[:,:,k])**2,axis=0)/npro)**0.5
    # rttov9_bias_levs = np.mean(rttov_od_layer[:,:,k]-od_layer[:,:,k],axis=0)
    # rttov9_rmse_levs = (np.sum((rttov_od_layer[:,:,k]-od_layer[:,:,k])**2,axis=0)/npro)**0.5
    # title = ch_info[ch]['cwl'] + ' v9 redictors'
    # plt.figure(dpi=300)
    # scatter_line_trans(od_layer[:,:,k],conv_od_layer[:,:,k],rttov_od_layer[:,:,k],title,\
    #                    mae_convol9,rmse_convol9,msle_convol9,\
    #                    mae_rttov9,rmse_rttov9,msle_rttov9)
    # figname = title+'.png'
    # plt.savefig(figpath/figname)
    #
    # plt.figure(dpi=300)
    # plot_error_level(conv9_bias_levs,conv9_rmse_levs,rttov9_bias_levs,rttov9_rmse_levs,p[1:],title)
    # figname = title+'_error_level'+'.png'
    # plt.savefig(figpath/figname)
    #####################################################################################################3
    # figpath = Path(r'G:\DL_transmitance\figures\RTTOV vs LBLRTM\some profiles od')
    # for j in range(83):
    #     title = ch_info[ch]['cwl'] + ' v9 redictors'
    #     xlabel = 'OD of single layer'
    #     plt.figure(dpi=300)
    #     plot_trans_level(od_layer[j,:,k],conv_od_layer[j,:,k],rttov_od_layer[j,:,k],p[1:],xlabel,title)
    #     figname = ch_info[ch]['cwl']+'_profile'+str(j)+'.png'
    #     plt.savefig(figpath/figname)


