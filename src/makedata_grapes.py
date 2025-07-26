# -*- coding: utf-8 -*-
"""
@ Time: {DATE},{TIME}
@ author: LiangHongli
@ Mail: Helen_Liang1@outlook.com
"""
import numpy as np
import pandas as pd
from pathlib import Path
import h5py
import my_funcs as mf
import read_files as rf
from LUT import ch_info
from calc_BT import emi

iras_dir = Path('/home/sky/rttov121/rttov_test/my_test.1.gfortran/iras')
srf_dir = '/mnt/hgfs/DL_transmitance/fy3_iras_znsys_srfch1-26/IRAS_srf.xlsx'
pro_dir = Path('/home/sky/rttov121/rttov_test/profile-datasets/GRAPES/20080709')
dir_flag = '*grapes*'
prodirs = [x for x in sorted(pro_dir.glob('**/*grapes*'))]
print(prodirs)
# ################# Reconstruct transmission and radiance #############
npros = []
for idir in prodirs:
    n = len([x for x in (idir).glob('*')])
    npros.append(n)

def construct_iras():
    nch = 3
    nlv = 101
    for k,kdir in enumerate(sorted(iras_dir.glob(dir_flag))):

        npro = npros[k]
        out_dir = kdir/'out' / 'direct'
        if k == 0:
            srf_iras = rf.trans_srf(out_dir / 'transmission.txt', nch)[:,:,np.newaxis]
            trans_iras = rf.trans_lv2space(out_dir / 'transmission.txt', npro, nch, nlv)
            trans_iras = np.concatenate((trans_iras,srf_iras),axis=2)
            bt_iras = rf.bt_clear(out_dir / 'radiance.txt', nch)
        else:
            srf_iras = rf.trans_srf(out_dir / 'transmission.txt', nch)[:,:,np.newaxis]
            temp = np.concatenate((rf.trans_lv2space(out_dir / 'transmission.txt', npro, nch, nlv),srf_iras),axis=2)
            trans_iras = np.append(trans_iras, temp, axis=0)
            bt_iras = np.append(bt_iras, rf.bt_clear(out_dir / 'radiance.txt', nch), axis=0)
    return trans_iras,bt_iras
# print('IRAS reconstruction done.')

############# Reconstruction done ######################################
chs = ['ch11','ch12','ch13']
trans_iras,bt_iras = construct_iras()
nprof = np.sum(np.array(npros))
# ############# Save the ground-truth and RTTOV's transmission and radiance to new file ##################
fsavename = 'grapes_20080709.HDF'
fsave = Path('/mnt/hgfs/DL_transmitance/revised datasets')/fsavename
k = 0
for parent in prodirs:
    childs = sorted([x for x in (parent).glob('*') if x.is_dir()])
    for child in childs:
        if k==0:
            profiles = rf.read_profile(child)
        else:
            profiles = np.vstack((profiles,rf.read_profile(child)))
        k += 1

from scipy.interpolate import interp1d
fp = Path(r'/mnt/hgfs/DL_transmitance/profiles/model_level_definition.xlsx')
df = pd.read_excel(fp,sheet_name='101L')
p = np.array(df['ph[hPa]'].values,dtype=np.float)
fo3 = '/home/sky/rttov121/rtcoef_rttov12/rttov9pred54L/rtcoef_fy3_1_iras.dat'
with open(fo3,'r') as f:
    lines = f.readlines()[432:486]

    for k,l in enumerate(lines):
        if k==0:
            o3 = np.array([float(x) for x in l.split()])
        else:
            o3 = np.vstack((o3,np.array([float(x) for x in l.split()])))

    interper = interp1d(o3[:,0],o3[:,2],kind='linear', fill_value='extrapolate')
    o3_interp = interper(p)

profiles = np.concatenate((profiles[:,:202],np.tile(o3_interp,(nprof,1)),profiles[:,202:]),axis=1)
############ surface emissivity ###################
nch = 3
emis = emi(iras_dir,dir_flag,nch)
with h5py.File(fsave,'w') as g:
    g.attrs['name'] = 'This is a independent dataset for testing GBT, 200 profiles in 0-30, 200 in 30-60, 200 in 60-90'
    x = g.create_dataset('X',data=profiles)
    x.attrs['name'] = 'Input features'
    x.attrs['variables'] = 'p,t,q,o3,p2m,t2m,u2m,v2m,srft,srftype,longitude,latitude,elevation,month'
    iras_bt = g.create_dataset('BT_RTTOV',data=bt_iras)
    iras_bt.attrs['name'] = 'TOA brightness temperature directly computed with RTTOV.'
    iras_emissivity = g.create_dataset('emissivity',data=emis)
    iras_emissivity.attrs['name'] = 'Surface emissivity(IRAS) computed with RTTOV'


