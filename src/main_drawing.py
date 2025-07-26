# -*- coding: utf-8 -*-
"""
@ Time: 2019/9/25
@ author: LiangHongli
@ Mail: Helen_Liang1@outlook.com
"""
from pathlib import Path
import h5py
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import pandas as pd
import random
from sklearn.model_selection import train_test_split

from plot_map_funcs import plot_profile_var,map_dot,map_scatter,bar_profile,plot_srf,bar_train_test_pro,plot_single_var
from read_files import read_IRAS_3a
import my_funcs as mf
from LUT import ch_info
from calc_new_predictor import var_star,layer_profile,predictors
############################# plot profile variables #################################
def plot_train_test_all(profiles,p,figpath,title):
    # figpath = Path(r'G:\DL_transmitance\figures\101L\profiles')
    nlv = 101

    npro = profiles.shape[0]
    # bad_index = []
    # for k in range(npro):
    #     if (profiles[k,nlv*2:nlv*3]<10**-6).any():
    #         bad_index.append(k)
    #     else:
    #         continue
    #
    # profiles = np.delete(profiles,bad_index,axis=0)
    # ###########################################################
    # N = 1200
    # n = 1000
    # index,dmin = mf.profile_selection(profiles[:,nlv:nlv*4],N,n)
    # profs = profiles[index,:]
    # print('The minimun distance is %.4f' % dmin)
    # month_col = -1
    # lon_col = 410
    # lat_col = 411
    # npro = profs.shape[0]
    # ntrain = int(npro * 0.8)
    # random.seed(2019)
    # train_index = random.sample(range(npro), ntrain)
    # test_index = [x for x in range(npro) if x not in train_index]
    #
    # trainx = profs[train_index,:]
    # testx = profs[test_index,:]
    # vars = ['Pressure(hPa)','Temperature(K)','Specific humidity(kg/kg)','Ozone mixing ratio(kg/kg)',\
    #         'CO2(kg/kg)','CO(kg/kg)','CH4(kg/kg)','N2O(kg/kg)']
    vars = ['Temperature(K)','Specific humidity(kg/kg)','Ozone mixing ratio(kg/kg)']
    for k in range(3):
        var = profiles[:,k*nlv:k*nlv+nlv]
        # print(var.shape,p.shape)
        plt.figure(dpi=300)
        plot_profile_var(var,p,vars[k])
        plt.title(title)
        figname = title+vars[k].split('(')[0]+'_all.png'
        plt.savefig(figpath/figname)

    return

def plot_train_test_split(profs,p,figpath):
    # profile_path = Path(r'G:/DL_transmitance/revised datasets/dataset_101L_final.HDF')
    # fp = Path(r'G:/DL_transmitance/profiles/model_level_definition.xlsx')

    nlv = 101
    # df = pd.read_excel(fp,sheet_name='101L')
    # p = np.array(df['ph[hPa]'].values,dtype=np.float)
    # with h5py.File(fx,'r') as f:
    #     profs = f['X'][:]
    #     new_preds = f['dependent/predictor_RTTOV'][:]

    npro = profs.shape[0]
    land_mask = profs[:,308]==0
    sea_mask = profs[:,308]==1
    X_train_land,X_test_land,Y_train_land,Y_test_land = train_test_split(profs[land_mask,:-1],profs[land_mask,-1],test_size=0.2,random_state=1269)
    X_train_sea,X_test_sea,Y_train_sea,Y_test_sea = train_test_split(profs[sea_mask,:-1],profs[sea_mask,-1],test_size=0.2,random_state=1269)
    X_train = np.concatenate((X_train_land,X_train_sea),axis=0)
    X_test = np.concatenate((X_test_land,X_test_sea),axis=0)
    Y_train = np.concatenate((Y_train_land,Y_train_sea),axis=0)
    Y_test = np.concatenate((Y_test_land,Y_test_sea),axis=0)
    vars = ['Temperature(K)','Specific humidity(kg/kg)','Ozone mixing ratio(kg/kg)']
    # for k in range(3):
    #     var = profs[:,k*nlv:k*nlv+nlv]
    #     # print(var.shape,p.shape)
    #     plt.figure(dpi=300)
    #
    #     plot_profile_var(var,p,vars[k])
    #
    #     figname = vars[k].split('(')[0]+'_split.png'
    #     plt.savefig(figpath/figname)

    # months = mf.stat_month(profile[:,-1])
    months_train = mf.stat_month(Y_train)
    months_test = mf.stat_month(Y_test)
    plt.figure(dpi=300)
    # bar_profile(months)
    bar_train_test_pro(months_train,months_test)
    figname = figpath/'bar_monthly_split.png'
    plt.savefig(figname)

    lon_col,lat_col = 309,310
    plt.figure(dpi=300)
    boundary = [-31,31,-180,180]
    title = 'Distribution of the profiles'
    map_dot(X_train[:,lon_col],X_train[:,lat_col],boundary,title,'r','o')
    map_dot(X_test[:,lon_col],X_test[:,lat_col],boundary,title,'b','^')
    figname = figpath/'profile_distribution_v3.png'
    plt.savefig(figname)
    return

def plot_ec83():
    fprofile = Path(r'G:\DL_transmitance\training_set\EC83pro_trainx_IRAS.xlsx')
    figpath = Path(r'G:\DL_transmitance\figures\profile_analysis\EC83')
    data = pd.read_excel(fprofile,sheet_name='pro_meta')
    vars = ['Pressure(hPa)','Temperature(K)','Specific humidity(kg/kg)','CO2(kg/kg)',
            'Ozone mixing ratio(kg/kg)','N2O(kg/kg)','CO(kg/kg)','CH4(kg/kg)']

    p = np.array(data.iloc[:101,1].values,dtype=np.float)
    index = [0,5,10,1,20,25,30,35,40,45,50,55,60,65,70]
    for k in range(1,5):
        var = data.iloc[:,k+1].values
        var = var.reshape(-1,101)
        if k==2:
            var = var*1.0E-6*18.01528/28.9647
        elif k==3:
            var = var*1.0E-6*44.0095/28.9647
        elif k==4:
            var = var*1.0E-6*47.9982/28.9647
        elif k==5:
            var = var*1.0E-6*44.0128/28.9647

        for j in np.arange(0,80,5):
            x = var[j,:]
            title = 'profile '+str(j+1)+'-'+str(j+5)
            plt.figure(dpi=300)
            plot_single_var(var[j,:],var[j+1,:],var[j+2,:],var[j+3,:],var[j+4,:],p,vars[k],title)
            figname = vars[k][:4]+'profile'+str(j)+'.png'
            plt.savefig(figpath/figname)
    return


def srf():
    figpath = Path('G:/DL_transmitance/figures/FY3_IRAS_srf/3C')
    fsrf = 'G:/DL_transmitance/fy3_iras_znsys_srfch1-26/fy3c_IRAS_srf.xlsx'
    center = ['14.95μm','14.71μm','14.49μm','14.22μm','13.97μm','13.64μm','13.35μm','12.47μm','11.11μm',\
              '9.71μm','7.43μm','7.33μm','6.52μm','4.57μm','4.52μm','4.47μm','4.45μm','4.19μm','3.98μm','3.76μm']
    for k in range(1,21):
        if k<10:
            name = '0'+str(k)
        else:
            name = str(k)
        data = pd.read_excel(fsrf,sheet_name=name)
        x = data.iloc[:,0]
        y = data.iloc[:,1]
        figname = name+'.png'
        title = name+' '+center[k-1]
        plt.figure(k)
        plot_srf(x,y,title)
        plt.savefig(figpath/figname,dpi=300)

    return

def weight_func():
    fdata1 = Path(r'G:\DL_transmitance\revised datasets\dataset_101L_rttov7_iasi7.HDF')
    fp = Path(r'G:\DL_transmitance\profiles\model_level_definition.xlsx')
    fsave = Path(r'G:\DL_transmitance\profiles\normed_weight_fy3c.xlsx')
    with h5py.File(fdata1,'r') as f:
        # p = f['X'][0,:101]
        # trans = f['Y'][:,:,:]
        trans = f['dependent/Y'][:,:,:]
    df = pd.read_excel(fp,sheet_name='101L')
    p = np.array(df['ph[hPa]'].values,dtype=np.float)
    # with h5py.File(fdata2,'r') as f:
    #     trans2 = f['Y'][:,:,:]

    # trans = np.concatenate((trans1[:9,:,:],trans2[10:,:,:]),axis=0)
    writer = pd.ExcelWriter(fsave)
    weight = np.zeros((3,1500,100),dtype=float)
    plt.figure(dpi=300)
    for k in range(3):
        # if k==0:
        #     ch = 'ch'+str(k+1)
        # elif k>=1 and k<=8:
        #     ch = 'ch'+str(k+2)
        # else:
        #     ch = 'ch'+str(k+3)
        ch = 'ch'+str(11+k)
        print(ch)
        # transk = np.mean(trans[k,:,:-1],axis=0)
        transk = trans[k,:,:-1]
        for j in range(1,101):
            weight[k,:,j-1] = (transk[:,j]-transk[:,j-1])/(np.log(p[j-1])-np.log(p[j]))

        x = np.mean(weight[k,:,:],axis=0)
        minos = np.max(x)-np.min(x)
        plt.plot(x,p[1:],label=ch_info[ch]['cwl'],linewidth=1)
        df = pd.DataFrame((x/minos).reshape(-1,1))
        df.to_excel(writer,sheet_name=str(ch))
        writer.save()
    writer.close()
    plt.legend()
    plt.xlabel('IRAS Weight Function')
    plt.ylabel('Pressure(hPa)')
    plt.yscale('log', basey=10)
    ax = plt.gca()
    ax.invert_yaxis()
    ax.xaxis.get_major_formatter().set_powerlimits((0, 3))
    plt.grid(True)
    plt.grid(color='k', linestyle='--', linewidth=0.5)
    figname = Path(r'G:\DL_transmitance\figures\FY3_IRAS_srf\weight_func11-13_fy3c_iasi7.png')
    plt.savefig(figname)
    return
############################### map IRAS observation ##############################
def map_bt():
    iras_obs_path = Path(r'G:\DL_transmitance\O-B_validation\FY3A_OBS')
    figpath = Path(r'G:\DL_transmitance\figures\profile_analysis\fy3a_OBS\ch13')
    tit = 'FY/3A/IRAS (6.52μm) Brightness Temperature(K) '
    boundary = [-90,90,-180,180]
    n = 1
    for xdir in iras_obs_path.iterdir():
        if n!=14:
            n+= 1
            continue
        # drawing the everyday's ascending part, one figure per day #
        plt.figure(num=n,figsize=(8,6),dpi=300)
        title = tit + xdir.name.split('/')[-1]
        for f in xdir.glob('*.HDF'):
            # if 'jpg' in f.name:
            #     continue
            # else:
            try:
                bt, lons, lats,_,_,_,_,_,_ = read_IRAS_3a(f)
            except:
                continue
            # according to the ascending order, split the ascending part
            if lats.shape[0]<300:
                continue
            for ii in range(270,290):
                if lats[ii,0] < lats[ii+1,0] and lats[ii+2,0] < lats[ii+3,0]:
                    r1 = ii # each file contains both ascending and descending part, label the first row of ascending part
                    break
                else:
                    continue

            for ii in range(r1,lats.shape[0]-3):
                if lats[ii,0] < lats[ii+1,0] and lats[ii+2,0] < lats[ii+3,0]:
                    continue
                else:
                    r2 = ii # each file contains both ascending and descending part, label the second row of ascending part
                    break

            bt12 = bt[12,r1:r2,:]
            lons_ascend = lons[r1:r2,:]
            lats_ascend = lats[r1:r2,:]

            # mask = bt12 == np.nan
            mbt12 = ma.masked_invalid(bt12)
            vmin = 210
            vmax = 290
            map_scatter(lons_ascend,lats_ascend,mbt12,boundary,title,vmin,vmax)
                # plt.show()
        plt.colorbar(pad=0.07,fraction=0.1,orientation='horizontal',aspect=35)
        plt.savefig(figpath/xdir.name.split('/')[-1])
        plt.close()

        print('The {:d}th figure is done.'.format(n))
        n += 1

    return

def map_ifs():
    fdata = '/mnt/hgfs/DL_transmitance/revised datasets/dataset_IFS91.HDF'
    with h5py.File(fdata,'r') as f:
        lons = f['X'][:,-4]
        lats = f['X'][:,-3]

    ntrain = int(893 * 0.9)
    random.seed(2019)
    train_index = random.sample(range(893), ntrain)
    test_index = [x for x in range(893) if x not in train_index]
    lons_train = lons[train_index]
    lats_train = lats[train_index]
    print(lons[:10])
    lons_test =lons[test_index]
    lats_test = lats[test_index]
    boundary = [-31,31,-180,180]
    title = 'Spatial distribution of the 893 profiles'
    plt.figure(dpi=300)
    map_dot(lons_train,lats_train,boundary,title,'r','o')
    map_dot(lons_test,lats_test,boundary,title,'b','^')
    plt.savefig(Path('/mnt/hgfs/DL_transmitance/figures/profile_analysis/clear-sky_ifs91_origin')/'profile_distribution')
    return

def plot_grapes_profile():
    profile_path = Path(r'G:/DL_transmitance/revised datasets/grapes_20080709.HDF')
    fp = Path(r'G:/DL_transmitance/profiles/model_level_definition.xlsx')
    figpath = Path(r'G:\DL_transmitance\figures\profile_analysis')
    nlv = 101

    with h5py.File(profile_path,'r') as f:
        profs = f['X'][:]

    lon_col = 309
    lat_col = 310
    vars = ['Temperature(K)','Specific humidity(kg/kg)']
    pdata = pd.read_excel(fp, sheet_name='101L')
    p = np.array(pdata.loc[:,'ph[hPa]'].values,dtype=np.float)
    for k in range(2):
        var = profs[:,k*nlv:k*nlv+nlv]
        # print(var.shape,p.shape)
        plt.figure(dpi=300)

        plot_profile_var(var,p,vars[k])

        figname = vars[k].split('(')[0]+'_grapes.png'
        plt.savefig(figpath/figname)


    plt.figure(dpi=300)
    boundary = [-31,31,-180,180]
    title = 'Distribution of the GFS profiles'
    map_dot(profs[:,lon_col],profs[:,lat_col],boundary,title,'r','o')
    figname = figpath/'profile_distribution_grapes.png'
    plt.savefig(figname)
    return


############################## Call the drawing functions ############################################
map_bt()
# plot_IFS()
# bar_dataset()
# srf()
# map_ifs()
# plot_ec83()
# weight_func()
# plot_grapes_profile()
def plot_new_preds(profiles,p,figpath,title):
    vars = ['qr2','qw','qw2','qr_tdif','qr21','qr41','qr','qr3',
      'qr4','qr_tdif2','qr21_tdif','qr2_div_qtw','qr21qr_div_qtw','tr','tr2','tfw','tfu','o3r',
      'o3r21','o3r_tdif','o3r21_tdif','o3r2_ow','o3r23_div_ow','o3r_ow','o3r_ow21','ow','ow2']
    n = len(vars)
    nlv = 100
    for k in range(n):
        var = profiles[:,k,:]
        # print(var.shape,p.shape)
        plt.figure(dpi=300)
        plot_profile_var(var,p,vars[k])
        plt.title(title)
        figname = title+vars[k]+'_all.png'
        plt.savefig(figpath/figname)

    return

######### [qw,qr_tdif,qr,qr21_tdif,qr2_div_qtw,qr21qr_div_qtw,tr,
def dataset():
    '''
    according to the distance, select profiles
    :return: profiles
    '''
    fx = Path(r'G:\DL_transmitance\revised datasets\final\dataset_ifs101L_v3.HDF')
    fp = Path(r'G:/DL_transmitance/profiles/model_level_definition.xlsx')
    savepath = Path(r'G:\DL_transmitance\revised datasets\final')
    figpath = Path(r'G:\DL_transmitance\figures\101L\final\profiles\v3')
    nlv = 101
    fenge = 30 #hPa
    df = pd.read_excel(fp,sheet_name='101L')
    p = np.array(df['ph[hPa]'].values,dtype=np.float)
    with h5py.File(fx,'r') as f:
        X = f['X'][:]
        # bt_rttov = f['BT_RTTOV'][:]
        # bt_true = f['BT_true'][:]
        # Y = f['Y'][:]
        # emissivity = f['emissivity'][:]
        # trans_rttov = f['transmission_RTTOV'][:]
        preds = f['predictor_RTTOV'][:]
    ###################plot and map ##################
    # plot_train_test_split(X[X[:,308]==0,:],p,figpath)
    plot_train_test_split(X[:],p,figpath)


    ##########################################################################
    # npro = X.shape[0]
    # bad_index = []
    # for k in range(npro):
    #     if X[k,304]<280 or X[k,201]<0.001:
    #         bad_index.append(k)
    #     else:
    #         continue
# #(X[k,nlv:nlv*2]<10**-6).any() or
#     X = np.delete(X,bad_index,axis=0)
#     bt_rttov = np.delete(bt_rttov,bad_index,axis=0)
#     bt_true = np.delete(bt_true,bad_index,axis=0)
#     Y = np.delete(Y,bad_index,axis=1)
#     trans_rttov = np.delete(trans_rttov,bad_index,axis=0)
#     emiss = np.delete(emissivity,bad_index,axis=0)
#     # print(X.shape[0])
#     var_percentile = np.percentile(X[:,:303],[0.5,99.5],axis=0)
#     outlier = [-1]
#     fanwei = np.concatenate((np.arange(30),np.arange(121,202),np.arange(247,303)))
#     for k in fanwei:#range(303):
#         mask = np.logical_or(X[:,k]<var_percentile[0,k],X[:,k]>var_percentile[1,k])
#         r = np.where(mask)[0]
#         for i in r:
#             if i in outlier:
#                 continue
#             else:
#                 outlier.append(i)
#
#     print(len(outlier))
#     X = np.delete(X,outlier[1:],axis=0)
#     print('number of profiles=%d' % X.shape[0])
#     print('ifs-137: land %d, ocean %d' % (sum(X[:,308]==0),sum(X[:,308]==1)))
#     bt_rttov = np.delete(bt_rttov,outlier[1:],axis=0)
#     bt_true = np.delete(bt_true,outlier[1:],axis=0)
#     Y = np.delete(Y,outlier[1:],axis=1)
#     trans_rttov = np.delete(trans_rttov,outlier[1:],axis=0)
#     emiss = np.delete(emiss,outlier[1:],axis=0)


#####################################################################
    # title = 'Profiles from IFS-137'
    # plot_train_test_all(X,p,figpath,title)
    # vstar = var_star(X[:,:nlv*3],nlv)
    # layerp = layer_profile(X[:,:nlv*3],nlv)
    # preds = predictors(vstar,layerp,p,nlv-1)

    # boundary = [-31,31,-180,180]
    # title = 'Spacial distribution of the profiles'
    # lon_col = 309
    # lat_col = 310
    # map_dot(X[:,lon_col],X[:,lat_col],boundary,title,'r','o')
    # figname = 'profile_distribution.png'
    # plt.savefig(figpath/figname)
    # title = 'Predictors from IFS-137'
    # plot_new_preds(preds[X[:,308]==0,:,:],p[1:],figpath,title)

    # fsave = savepath/'dataset_ifs101L_v4.HDF'
    # with h5py.File(fsave,'w') as f:
    #     f.create_dataset('BT_RTTOV',data=bt_rttov,chunks=True,compression='gzip')
    #     f.create_dataset('BT_true',data=bt_true,compression='gzip')
    #     f.create_dataset('Y',data=Y,compression='gzip')
    #     f.create_dataset('emissivity',data=emiss,compression='gzip')
    #     f.create_dataset('transmission_RTTOV',data=trans_rttov,compression='gzip')
    #     x = f.create_dataset('X',data=X,compression='gzip')
    #     x.attrs['name'] = 'profiles, the variables are: temperature(101),water vapor(101),ozone(101),2m variables, surface variable, geo, month'
    #     predictor = f.create_dataset('predictor_RTTOV',data=preds,compression='gzip')
    #     predictor.attrs['name'] = 'predictors same as RTTOV'
    #     predictor.attrs['shape'] = 'n_sample,n_feature(6),n_layer'

# dataset()
