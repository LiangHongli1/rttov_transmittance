# -*- coding: utf-8 -*-
"""
@ Time: 2020-4-6
@ author: LiangHongli
@ Mail: Helen_Liang1@outlook.com
to test the profile interpolation code
"""
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
    fx = Path(r'G:\DL_transmitance\revised datasets\dataset_IFS137_ch1_ch10.HDF')
    fgfs = Path(r'G:\DL_transmitance\O-B_validation\200807_grb2_to_nc\gfsanl_4_20080709_0000_003.nc')
    ft639 = Path(r'G:\DL_transmitance\O-B_validation\201905_grb2_to_nc\20190517\gmf.gra.2019051700003.nc')
    fera = Path(r'G:\DL_transmitance\O-B_validation\ERA5\ERA4_model_level_200807.nc')
    fp = Path(r'G:\DL_transmitance\profiles\model_level_definition.xlsx')
    figpath = Path(r'G:\DL_transmitance\figures\profile_analysis\gfs_t639')

    df = pd.read_excel(fp,sheet_name='101L')
    p101 = np.array(df.iloc[:,0].values,dtype=float)
    df = pd.read_excel(fp,sheet_name='60L')
    p60 = np.array(df.loc[:,'ph [hPa]'].values,dtype=float)
    level = p60[1:]
    
    # with h5py.File(fx,'r') as f:
    #     X = f['X'][:]

    # with Dataset(ft639,'r') as f:
    #     lv_gfs_tmp = f['lv_ISBL0'][:]*0.01
    #     lv_gfs_w = f['lv_ISBL2'][:]*0.01
    #     tmp = f['TMP_P0_L100_GLL0'][:,241:480,:]
    #     tmp_2m = f['TMP_P0_L103_GLL0'][241:480,:]
    #     rh = f['SPFH_P0_L100_GLL0'][:,241:480,:]

    # with Dataset(fgfs,'r') as g:
    #     lv_gfs_tmp = g.variables['lv_ISBL0'][:]*0.01
    #     lv_gfs_w = g.variables['lv_ISBL4'][:]*0.01
    #     tmp = g.variables['TMP_P0_L100_GLL0'][:,120:241,:]
    #     rh = g.variables['RH_P0_L100_GLL0'][:,120:241,:]*0.01
    #     tmp_2m = g.variables['TMP_P0_L103_GLL0'][120:241,:]
    # tmp_l100,q_l100,o3_l100,cc_l100,level,lons_fore,lats_fore = rf.read_era5_pressure_level(fera)
    tmp_l100,q_l100,o3_l100,cc_l100,lons_fore,lats_fore = rf.read_era4_model_level(fera)
    index = 8
    tmp_l100 = tmp_l100[index] # currently, we only use data of one moment: 06:00
    print(tmp_l100[:,0,5])
    q_l100 = q_l100[index]
    o3_l100 = o3_l100[index]
    cc_l100 = cc_l100[index]
    _,s1,s2 = tmp_l100.shape
    rs = []
    cs = []
    
    for k in range(s1):
        for j in range(s2):
            # print(cc_l100[:,k,j])
            if (cc_l100[:,k,j]<0.3).all():
                rs.append(k)
                cs.append(j)
                
    qvalid = (q_l100[:,rs,cs]).T
    tvalid = np.transpose(tmp_l100[:,rs,cs],[1,0])
    o3valid = (o3_l100[:,rs,cs]).T
    n = qvalid.shape[0]
    qint = np.zeros((n,101))
    tint = np.zeros((n,101))
    o3int = np.zeros((n,101))
    for k in range(n):
        qint[k,:] = interp_profile(level,qvalid[k,:],p101)
        tint[k:] = interp_profile(level,tvalid[k,:],p101)
        o3int[k,:] = interp_profile(level,o3valid[k,:],p101)
        
    plot_profile_var(qvalid,level,'Specific humidity(kg/kg)')
    figname = 'ERA4_water_200807090006_specific.png'
    plt.savefig(figpath/figname)
    plot_profile_var(tvalid,level,'Temperature(K)')
    figname = 'ERA4_tmp_200807090006.png'
    plt.savefig(figpath/figname)
    plot_profile_var(o3valid,level,'Ozone(kg/kg)')
    figname = 'ERA4_o3_200807090006.png'
    plt.savefig(figpath/figname)
    
    plot_profile_var(qint,p101,'Specific humidity(kg/kg)')
    figname = 'ERA4_water_200807090006_interp.png'
    plt.savefig(figpath/figname)
    plot_profile_var(tint,p101,'Temperature(K)')
    figname = 'ERA4_tmp_200807090006_interp.png'
    plt.savefig(figpath/figname)
    plot_profile_var(o3int,p101,'Ozone(kg/kg)')
    figname = 'ERA4_o3_200807090006_interp.png'
    plt.savefig(figpath/figname)


    # pw = np.array(lv_gfs_w)
    # s1,s2,s3 = tmp.shape
    # tmp_reshape = tmp.reshape(s1,-1)
    # nprof = tmp_reshape.shape[1]
    # plot_profile_var(tmp,lv_gfs_tmp,'Temperature(K)')
    # # # figname = 'T639_tmp_201905170003.png'
    # figname = 'GFS_tmp_200807090003.png'
    # plt.savefig(figpath/figname)
    # rh = np.transpose(rh.reshape(rh.shape[0],-1),[1,0])
    # plot_profile_var(rh,lv_gfs_w,'Specific humidity(kg/kg)')
    # figname = 'GFS_water_200807090003.png'
    # plt.savefig(figpath/figname)

    # n = rh.shape[0]
    # tmp_lvw = np.zeros((n,nprof),dtype=float)
    # # print(ma.)
    # for j in range(nprof):
    #     tmpj = tmp_reshape[:,j]
    #     tmp_interp = interp1d(np.log(lv_gfs_tmp),tmpj,kind='linear',fill_value='extrapolate')
    #     tmp_lvw[:,j] = tmp_interp(np.log(pw))

    # wv = rh2q(rh.reshape(n,-1),tmp_lvw,np.tile(lv_gfs_w.reshape(-1,1),(1,87120)))
    # prof_interp = np.zeros((nprof,101),dtype=float)
    # lvk = lv_gfs_tmp
    # for k in range(nprof): # 100 profiles
    #     # lvk = tmp[k,:]
    #     # tmpj = tmp[k,:]
    #     rhj = wv[:,k]
    #     # o3j = X[k,137*3:137*4]
    #     # if lvk[-1]<p101[-1]:
    #     #     jj = np.where(p101>lvk[-1])[0][0]
    #     # else:
    #     #     jj = -1
    #     # tmp_interp = interp1d(np.log(lvk),tmpj,kind='linear',fill_value='extrapolate')
    #     rh_interp = interp1d(np.log(lv_gfs_w),rhj,kind='linear',fill_value='extrapolate')
    #     # o3_interp = interp1d(np.log(lvk),o3j,kind='linear',fill_value='extrapolate')
    #     # prof_interp[k] = tmp_interp(np.log(p101))
    #     prof_interp[k] = rh_interp(np.log(p101))
        # prof_interp[k,202:303] = o3_interp(np.log(p101))

        # prof_interp[k,jj:101] = tmpj[jj]
        # prof_interp[k,101+jj:202] = rhj[jj]
        # prof_interp[k,202+jj:303] = o3j[jj]

    # plot_profile_var(prof_interp,p101,'Temperature(K)')
    # figname = 'GFS_tmp_200807090003_interped.png'
    # plot_profile_var(wv.T,lv_gfs_w,'Specific humidity(kg/kg)')
    # figname = 'GFS_water_200807090003_specific.png'
    # plt.savefig(figpath/figname)

    # r,c = tmp_2m.shape
    # prof_interp = np.zeros((r*c,3*101),dtype=float)
    # m = 0
    # for ii in range(r):
    #     for jj in range(c):
    #         tmpij = tmp[:,ii,jj]
    #         rhij = rh[:,ii,jj]
    #         # o3j = rh[:,ii,jj]
    #
    #         if lv_gfs_tmp[-1]<p101[-1]:
    #             k = np.where(p101>lv_gfs_tmp[-1])[0][0]
    #         else:
    #             k = -1
    #         tmp_interp = interp1d(lv_gfs_tmp,tmpij,kind='linear',fill_value='extrapolate')
    #         prof_interp[m,:101] = tmp_interp(p101)
    #         # prof_interp[m,:101] = intf.InterpProfile(lv_gfs_tmp,p101,tmpij)
    #         # prof_interp[m,k:101] = tmpij[-1]
    #
    #         if lv_gfs_w[-1]<p101[-1]:
    #             k = np.where(p101>lv_gfs_w[-1])[0][0]
    #         else:
    #             k = -1
    #         rh_interp = interp1d(lv_gfs_w,rhij,kind='linear',fill_value='extrapolate')
    #         prof_interp[m,101:202] = rh_interp(p101)
    #         # prof_interp[m,101:202] = intf.InterpProfile(lv_gfs_w,p101,rhij)
    #         # prof_interp[m,101+k:202] = rhij[-1]
    #
    #         prof_interp[m,202:303] = prof_interp[m,101:202]



    # plot_train_test_all(prof_interp,p101,figpath,title)



if __name__ == '__main__':
    main()
