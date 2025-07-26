#----------- interpolate top atmospheric profile of era4 to era5 --------------
import h5py
from netCDF4 import Dataset
import pandas as pd
from numpy import ma
import numpy as np
from pathlib import Path
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

import intepolate_functions as intf
from main_drawing import plot_train_test_all
from plot_map_funcs import plot_profile_var
from my_funcs import rh2q,interp_profile
import read_files as rf

def main():
    # fx = Path(r'G:\DL_transmitance\revised datasets\dataset_IFS137_ch1_ch10.HDF')
    fera4 = Path(r'G:\DL_transmitance\O-B_validation\ERA5\ERA4_model_level_200807_001218.nc')
    fera5 = Path(r'G:\DL_transmitance\O-B_validation\ERA5\ERA5_ensemble_mean_pressure_level_20080714.nc')
    fp = Path(r'G:\DL_transmitance\profiles\model_level_definition.xlsx')
    figpath = Path(r'G:\DL_transmitance\figures\profile_analysis\gfs_t639')
    savepath = Path(r'G:\DL_transmitance\O-B_validation\ERA5')

    df = pd.read_excel(fp,sheet_name='101L')
    p101 = np.array(df.iloc[:,0].values,dtype=float)
    df = pd.read_excel(fp,sheet_name='60L')
    p60 = np.array(df.loc[:,'ph [hPa]'].values,dtype=float)
    level4 = p60[1:]
    
    p2int = level4[:6]
    pint = np.array([0.1361,0.1861,0.2499,0.3299,0.4288,0.5496,0.6952,0.8690,1])
    nint = len(pint)
    
    index4 = 41
    index5 = 7
    tmp_l1005_,q_l1005_,o3_l1005_,cc_l1005,level5,_,_ = rf.read_era5_pressure_level(fera5,index5)
    tmp_l1004,q_l1004,o3_l1004,_,_,_ = rf.read_era4_model_level(fera4,index4)
    
    ###-------------------
    tmp_l1005_ = tmp_l1005_[:,120:241,:]
    q_l1005_ = q_l1005_[:,120:241,:]
    o3_l1005_ = o3_l1005_[:,120:241,:]
    # print(q_l1005_.shape)
    ###-------------------
    
    # cc_l1005 = cc_l1005_[index5]
    
    # tmp_l1004 = tmp_l1004[index4] # currently, we only use data of one moment: 06:00
    # print(tmp_l100[:,0,5])
    # q_l1004 = q_l1004[index4]
    # o3_l1004 = o3_l1004[index4]
    
    _,s1,s2 = tmp_l1004.shape    
    n = s1*s2
    qint = np.zeros((nint,s1,s2))
    tint = np.zeros((nint,s1,s2))
    o3int = np.zeros((nint,s1,s2))
    for k in range(s1):
        for j in range(s2):
            qint[:,k,j] = interp_profile(p2int,q_l1004[:6,k,j],pint)
            tint[:,k,j] = interp_profile(p2int,tmp_l1004[:6,k,j],pint)
            o3int[:,k,j] = interp_profile(p2int,o3_l1004[:6,k,j],pint)
            qratio = q_l1005_[0,k,j]/qint[-1,k,j]
            tratio = tmp_l1005_[0,k,j]/tint[-1,k,j]
            o3ratio = o3_l1005_[0,k,j]/o3int[-1,k,j]
            qint[:,k,j] = qratio*qint[:,k,j]
            tint[:,k,j] = tratio*tint[:,k,j]
            o3int[:,k,j] = o3ratio*o3int[:,k,j]
        
    
    print(tint.shape,tmp_l1005_[:,:,:].shape)
    tmp_l1005 = np.concatenate((tint[:-1],tmp_l1005_[:,:,:]),axis=0)
    q_l1005 = np.concatenate((qint[:-1],q_l1005_[:,:,:]),axis=0)
    o3_l1005 = np.concatenate((o3int[:-1],o3_l1005_[:,:,:]),axis=0)
    level = np.concatenate((pint[:-1],level5))
    
    with h5py.File(savepath/'make_ERA5_pressure_level_2008071418.HDF','w') as f:
        f['t'] = tmp_l1005
        f['q'] = q_l1005
        f['o3'] = o3_l1005
        f['level'] = level
    rs = []
    cs = []
    for kk in range(s1):
        for jj in range(s2):
            # print(cc_l100[:,k,j])
            if (cc_l1005[:,kk,jj]<0.3).all():
                rs.append(kk)
                cs.append(jj)
                
    qvalid = (q_l1005[:,rs,cs]).T
    tvalid = np.transpose(tmp_l1005[:,rs,cs],[1,0])
    o3valid = (o3_l1005[:,rs,cs]).T
    
    
    plot_profile_var(qvalid,level,'Specific humidity(kg/kg)')
    figname = 'combine_ERA5_water_200807140018_specific.png'
    plt.savefig(figpath/figname)
    plot_profile_var(tvalid,level,'Temperature(K)')
    figname = 'combine_ERA5_tmp_200807140018.png'
    plt.savefig(figpath/figname)
    plot_profile_var(o3valid,level,'Ozone(kg/kg)')
    figname = 'combine_ERA5_o3_200807140018.png'
    plt.savefig(figpath/figname)
    
    
    # plot_profile_var(qint,p101,'Specific humidity(kg/kg)')
    # figname = 'ERA4_water_200807090006_interp.png'
    # plt.savefig(figpath/figname)
    # plot_profile_var(tint,p101,'Temperature(K)')
    # figname = 'ERA4_tmp_200807090006_interp.png'
    # plt.savefig(figpath/figname)
    # plot_profile_var(o3int,p101,'Ozone(kg/kg)')
    # figname = 'ERA4_o3_200807090006_interp.png'
    # plt.savefig(figpath/figname)
    
if __name__ == '__main__':
    main()