# -*- coding: utf-8 -*-
"""
@ Time: 2019/9/25
@ author: LiangHongli
@ Mail: Helen_Liang1@outlook.com
Select and analyse profiles from an original set. The selected profiles must fall in 12 months and the globe evenly.
"""

import numpy as np
from pathlib import Path
import random
import matplotlib.pyplot as plt
import shutil
import subprocess
from scipy.interpolate import interp1d
import pandas as pd

import my_funcs as mf
from calc_new_predictor import var_star,layer_profile,predictors

#################### Plot and map the original set to check the distribution ##########################

def move_file():
    srcpath = Path(r'/home/sky/profiles/IFS_137/00-30')
    tarpath = Path(r'/home/sky/rttov121/rttov_test/profile-datasets/new_set')
    src_dirs = sorted([x for x in srcpath.glob('*')])
    for k in range(1,len(src_dirs)+1):
        tardir = str(k // 100 + 1) + 'redai'
        child_dir = tarpath/tardir
        p = src_dirs[k - 1].name

        shutil.copytree(src_dirs[k-1],dst=child_dir/p)
    return
# move_file()

def interp_profile():
    prodir = Path(r'/home/sky/rttov121/rttov_test/profile-datasets/new_set')
    # prodir = Path(r'/home/sky/078')
    pfile = Path(r'/mnt/hgfs/DL_transmitance/profiles/model_level_definition.xlsx')
    pdata = pd.read_excel(pfile,sheet_name='101L')
    pp = np.array(pdata['ph[hPa]'].values,dtype=np.float)
    for dd in sorted(prodir.glob('*')):
        for child in sorted(dd.glob('*')):
            fvar2m = child/'ground'/'s2m.txt'
            with open(fvar2m,'r') as f:
                lines = f.readlines()
                for l in lines:
                    if 's0%t' in l:
                        tsrf = float(l.split('=')[-1])
                    elif 's0%p' in l:
                        psrf = float(l.split('=')[-1])

            atmdir = child/'atm'
            # (atmdir/'cc.txt').unlink()
            p = np.loadtxt(atmdir/'p.txt')
            t = np.loadtxt(atmdir/'t.txt')
            q = np.loadtxt(atmdir/'q.txt')
            o3 = np.loadtxt(atmdir/'o3.txt')
            qinter = interp1d(p, q, kind='linear', fill_value='extrapolate')
            qq = qinter(pp)
            if psrf>=1100:
                tinter = interp1d(p,t,kind='linear',fill_value='extrapolate')
                tt = tinter(pp)
                o3inter = interp1d(p,o3,kind='linear',fill_value='extrapolate')
                o33 = o3inter(pp)
            else:
                larger = np.where(pp>=psrf)[0]
                smaller = np.where(pp<psrf)[0]
                tinter = interp1d(p, t, kind='linear', fill_value='extrapolate')
                tt = np.concatenate((tinter(pp[smaller]),np.array([tsrf]*len(larger))))
                o3inter = interp1d(p, o3, kind='linear', fill_value='extrapolate')
                o33 = np.concatenate((o3inter(pp[smaller]),np.array([o3[-1]]*len(larger))))
            (atmdir/'p.txt').unlink()
            (atmdir/'t.txt').unlink()
            (atmdir/'q.txt').unlink()
            (atmdir/'o3.txt').unlink()
            np.savetxt(atmdir/'p.txt',pp.reshape(-1,1),fmt='%.6E')
            np.savetxt(atmdir/'t.txt',tt.reshape(-1,1),fmt='%.6E')
            np.savetxt(atmdir / 'q.txt', qq.reshape(-1, 1), fmt='%.6E')
            np.savetxt(atmdir / 'o3.txt', o33.reshape(-1, 1), fmt='%.6E')
            if (qq<0.1E-10).any():
                print(child)
                subprocess.call(['rm', '-rf', str(child)])
            else:
                continue

    return
# interp_profile()

def rename_profile_dir():
    srcpath = Path(r'/home/sky/rttov121/rttov_test/profile-datasets/GRAPES/20080714/0616')
    iasipath = Path(r'/home/sky/rttov121/rttov_test/tests.0/iasi')
    iraspath = Path(r'/home/sky/rttov121/rttov_test/tests.0/iras')

    # iasi_ch = np.arange(38,1853)
    # iasi_ch = np.arange(2052,2452)
    # iasi_ch = np.arange(2052,4219)
    # iasi_ch = np.arange(5839,6583)
    # iasi_ch = np.arange(2431,3913)
    iras_ch = np.arange(11,14)
    # mf.generate_chpr_txt(iasipath / 'div83'/'in', iasi_ch, 83)
    # mf.generate_chpr_txt(iraspath / '200807140006'/'in', iras_ch, 469)
    k = 1
    for cdir in sorted(srcpath.glob('*')):
    #     k = 1
        ccdirs = sorted([x for x in cdir.glob('*')])
        npro = len(ccdirs)
        if npro==0:
            continue
    #     for ccdir in ccdirs:
    #         try:
    #             name = '0'*(3-len(str(k)))+str(k)
    #             ccdir.rename(cdir/name)
    #         except:
    #             pass
    #         k += 1
    #     #     # name = str(k)+'grapes0709'
    #     #     # ccdir.rename(cdir/name)
    #     #     npro = len([x for x in ccdir.glob('*')])
    #     #     if not (iraspath / ccdir.name / 'in').exists():
    #     #         (iraspath / ccdir.name / 'in').mkdir(parents=True)
    #     #     else:
    #     #         pass
    #     #     mf.generate_chpr_txt(iraspath / ccdir.name / 'in', iras_ch, npro)
    #     #     piras = iraspath / ccdir.name / 'in' / 'profiles'
    #     #     # if piras.exists():
    #     #     #     subprocess.call(['rm', '-rf', str(piras)])
    #     #
    #     #     status = subprocess.call(['ln', '-s', str(ccdir), str(piras)])
    #     #     k += 1
    #
    #     if not (iasipath/cdir.name/'in').exists():
    #         (iasipath/cdir.name/'in').mkdir(parents=True)
    #     else:
    #         pass
        if not (iraspath/cdir.name/'in').exists():
            (iraspath/cdir.name/'in').mkdir(parents=True)
        else:
            pass
    #     mf.generate_chpr_txt(iasipath/cdir.name/'in',iasi_ch,npro)
        mf.generate_chpr_txt(iraspath / cdir.name / 'in', iras_ch, npro)
        piras = iraspath/cdir.name/'in'/'profiles'
    #     piasi = iasipath/cdir.name/'in'/'profiles'
    #     if piras.exists():
    #         subprocess.call(['rm','-rf',str(piras)])
    #     if piasi.exists():
    #         subprocess.call(['rm','-rf',str(piasi)])
    #
        status = subprocess.call(['ln', '-s', str(cdir),str(piras)])
    #     status = subprocess.call(['ln', '-s', str(cdir),str(piasi)])
    return
rename_profile_dir()

def move_iasi_out():
    srcpath = Path('/home/sky/rttov121/rttov_test/my_test_v9.1.gfortan/iasi')
    tarpath = Path('/home/sky/rttov121/rttov_test/my_test_v9.1.gfortan/iasi/RTTOV_results')
    s = '_2452-4219'
    for sdir in sorted(srcpath.glob('*1-*')):
        tardir = tarpath/sdir.name
        if tardir.exists():
            pass
        else:
            tardir.mkdir()
        outdir = sdir/'out'/'direct'
        for f in outdir.glob('*.txt'):
            fmovename = f.name.split('.')[0]+s+'.txt'
            shutil.copyfile(f,tardir/fmovename)

    return
# move_iasi_out()
####################### execute the function ####################################

'''
Notice: Error could accur even deleting the profiles beyond the hard limit when executing RTTOV. 
New profiles should be selected to replace the bad profile.
'''
import h5py
def dataset():
    '''
    according to the distance, select profiles
    :return: profiles
    '''
    # x1 = Path(r'G:/DL_transmitance/revised datasets/dataset_101L_iasirttov7_former.HDF')
    # x2 = Path(r'G:/DL_transmitance/revised datasets/dataset_101L_iasirttov7_latter.HDF')
    fx = Path(r'G:\DL_transmitance\revised datasets\dataset_101L_XX.HDF')
    fy1 = Path(r'G:\DL_transmitance\revised datasets\dataset_101L_Y_0-13.HDF')
    fy2 = Path(r'G:\DL_transmitance\revised datasets\dataset_101L_Y_after13.HDF')
    fp = Path(r'G:/DL_transmitance/profiles/model_level_definition.xlsx')
    savepath = Path(r'G:\DL_transmitance\revised datasets')
    nlv = 101

    df = pd.read_excel(fp,sheet_name='101L')
    p = np.array(df['ph[hPa]'].values,dtype=np.float)
    # with h5py.File(x1,'r') as f:
    #     profiles = f['X'][:]
    #     bt_rttov1 = f['BT_RTTOV'][:]
    #     bt_true1 = f['BT_true'][:]
    #     Y1 = f['Y'][:]
    #     emissivity = f['emissivity'][:]
    #     transmission1 = f['transmission_RTTOV'][:]
    #
    # with h5py.File(x2,'r') as f:
    #     # cld = f['X'][:,413:550]
    #     bt_rttov2 = f['BT_RTTOV'][:]
    #     bt_true2 = f['BT_true'][:]
    #     Y2 = f['Y'][:]
    #     transmission2 = f['transmission_RTTOV'][:]
    with h5py.File(fx,'r') as f:
        profiles = f['X'][:,nlv:]

    with h5py.File(fy1,'r') as f:
        bt_rttov1 = f['BT_RTTOV'][:]
        bt_true1 = f['BT_true'][:]
        Y1 = f['Y'][:]
        emissivity = f['emissivity'][:]
        transmission1 = f['transmission_RTTOV'][:]

    with h5py.File(fy2,'r') as f:
        bt_rttov2 = f['BT_RTTOV'][:]
        bt_true2 = f['BT_true'][:]
        Y2 = f['Y'][:]
        transmission2 = f['transmission_RTTOV'][:]

    bt_rttov = np.concatenate((bt_rttov1,bt_rttov2),axis=0)
    bt_true = np.concatenate((bt_true1,bt_true2),axis=0)
    Y = np.concatenate((Y1,Y2),axis=1)
    trans_rttov = np.concatenate((transmission1,transmission2),axis=0)

    npro = profiles.shape[0]
    bad_index = []
    for k in range(npro):
        if (profiles[k,nlv:nlv*2]<10**-6).any() or profiles[k,304]<280:
            bad_index.append(k)
        else:
            continue

    profiles = np.delete(profiles,bad_index,axis=0)
    bt_rttov = np.delete(bt_rttov,bad_index,axis=0)
    bt_true = np.delete(bt_true,bad_index,axis=0)
    Y = np.delete(Y,bad_index,axis=1)
    trans_rttov = np.delete(trans_rttov,bad_index,axis=0)
    emissivity = np.delete(emissivity,bad_index,axis=0)
    npro = profiles.shape[0]
    ###########################################################
    N = 1500
    n = 1000
    index,dmin = mf.profile_selection(profiles[:,:nlv*3],N,n)
    index2 = [x for x in range(npro) if x not in index]
    X1 = profiles[index,:]
    bt_rttov1 = bt_rttov[index,:]
    bt_true1 = bt_true[index,:]
    Y1 = Y[:,index,:]
    trans_rttov1 = trans_rttov[index,:,:]
    emissivity1 = emissivity[index,:]

    X2 = profiles[index2,:]
    bt_rttov2 = bt_rttov[index2,:]
    bt_true2 = bt_true[index2,:]
    Y2 = Y[:,index2,:]
    trans_rttov2 = trans_rttov[index2,:,:]
    emissivity2 = emissivity[index2,:]
    print('The minimun distance is %.4f' % dmin)

    vstar = var_star(X1[:,:nlv*3],nlv)
    layerp = layer_profile(X1[:,:nlv*3],nlv)
    new_preds1 = predictors(vstar,layerp,p,nlv-1)

    vstar = var_star(X2[:,:nlv*3],nlv)
    layerp = layer_profile(X2[:,:nlv*3],nlv)
    new_preds2 = predictors(vstar,layerp,p,nlv-1)

    fsave = savepath/'dataset_101L_final.HDF'
    with h5py.File(fsave,'w') as f:
        dependent = f.create_group('dependent')
        dependent.create_dataset('BT_RTTOV',data=bt_rttov1,chunks=True,compression='gzip')
        dependent.create_dataset('BT_true',data=bt_true1,compression='gzip')
        dependent.create_dataset('Y',data=Y1,compression='gzip')
        dependent.create_dataset('emissivity',data=emissivity1,compression='gzip')
        dependent.create_dataset('transmission_RTTOV',data=trans_rttov1,compression='gzip')
        x = dependent.create_dataset('X',data=X1,compression='gzip')
        x.attrs['name'] = 'profiles, the variables are: temperature(101),water vapor(101),ozone(101),2m variables, surface variable, geo, month'
        predictor = dependent.create_dataset('predictor_RTTOV',data=new_preds1,compression='gzip')
        predictor.attrs['name'] = 'predictors same as RTTOV'
        predictor.attrs['shape'] = 'n_sample,n_feature(6),n_layer'

        independent = f.create_group('independent')
        independent.create_dataset('BT_RTTOV',data=bt_rttov2,chunks=True,compression='gzip')
        independent.create_dataset('BT_true',data=bt_true2,compression='gzip')
        independent.create_dataset('Y',data=Y2,compression='gzip')
        independent.create_dataset('emissivity',data=emissivity2,compression='gzip')
        independent.create_dataset('transmission_RTTOV',data=trans_rttov2,compression='gzip')
        x = independent.create_dataset('X',data=X2,compression='gzip')
        x.attrs['name'] = 'profiles, the variables are: temperature(101),water vapor(101),ozone(101),2m variables, surface variable, geo, month'
        predictor = independent.create_dataset('predictor_RTTOV',data=new_preds2,compression='gzip')
        predictor.attrs['name'] = 'predictors same as RTTOV'
        predictor.attrs['shape'] = 'n_sample,n_feature(6),n_layer'

# dataset()

def dataset_grapes():
    x1 = Path(r'G:\DL_transmitance\O-B_validation\xbt_rttov4obs.HDF')
    fp = Path(r'G:/DL_transmitance/profiles/model_level_definition.xlsx')
    savepath = Path(r'G:\DL_transmitance\revised datasets')
    fobs_bt = Path(r'G:\DL_transmitance\O-B_validation\BT_obs.xlsx')
    nlv = 101

    df = pd.read_excel(fp,sheet_name='101L')
    p = np.array(df['ph[hPa]'].values,dtype=np.float)

    df = pd.read_excel(fobs_bt,sheet_name=0,usecols=[1,2,3,4,5])
    obs_bt = df.values  # observational BT of IRAS channel 11 12 13

    with h5py.File(x1,'r') as f:
        X = f['X'][:]
        bt_rttov = f['BT_rttov'][:]
        emissivity = f['emissivity'][:]

    npro = X.shape[0]
    bad_index = []
    for k in range(npro):
        if X[k,nlv-1]<280 or X[k,312]>30:
            bad_index.append(k)
        else:
            continue

    X = np.delete(X,bad_index,axis=0)
    bt_rttov = np.delete(bt_rttov,bad_index,axis=0)
    emissivity = np.delete(emissivity,bad_index,axis=0)
    obs_bt = np.delete(obs_bt,bad_index,axis=0)

    print('There are %d valid profiles' % X.shape[0])
    vstar = var_star(X[:,:nlv*3],nlv)
    layerp = layer_profile(X[:,:nlv*3],nlv)
    new_preds = predictors(vstar,layerp,p,nlv-1)

    fsave = savepath/'grapes_dataset_101L_valid.HDF'
    with h5py.File(fsave,'w') as f:
        f.create_dataset('BT_RTTOV',data=bt_rttov,chunks=True,compression='gzip')
        x = f.create_dataset('X',data=X[:,:-2],compression='gzip')
        x.attrs['name'] = 'profiles, the variables are: temperature(101),water vapor(101),ozone(101),2m variables, surface variable, geo, month'
        f.create_dataset('emissivity',data=emissivity,compression='gzip')
        predictor = f.create_dataset('predictor_RTTOV',data=new_preds,compression='gzip')
        predictor.attrs['name'] = 'predictors same as RTTOV'
        predictor.attrs['shape'] = 'n_sample,n_feature(6),n_layer'
        f.create_dataset('BT_obs',data=obs_bt,compression='gzip')

# dataset_grapes()
