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
################################################ Make dataset start ###############################################################
iasi_dir = Path('/home/sky/rttov121/rttov_test/my_test.1.gfortran/iasi')
iras_dir = Path('/home/sky/rttov121/rttov_test/my_test.1.gfortran/iras')
srf_dir = '/mnt/hgfs/DL_transmitance/fy3_iras_znsys_srfch1-26/IRAS_srf.xlsx'
pro_dir = Path('/home/sky/rttov121/rttov_test/profile-datasets/self_div83')
# ################# Reconstruct transmission and radiance #############
npro = 83

def construct_iasi():
    nch = 2167
    nlv = 101
    ftrans = 'transmission.txt'
    frad = 'radiance.txt'
    # print(prodirs)
    fdir = iasi_dir/'self_div83'/'out'/'direct'
    srf_iasi = rf.trans_srf(fdir / ftrans, nch)[:,:,np.newaxis]
    trans_iasi = rf.trans_lv2space(fdir/ftrans, npro, nch, nlv)
    trans_iasi = np.concatenate((trans_iasi,srf_iasi),axis=2)
    rad_iasi = rf.rad_clear(fdir/frad, npro, nch)
    return trans_iasi, rad_iasi

print('IASI reconstruction done.')

def construct_iras():
    nch = 3
    nlv = 101
    out_dir = iras_dir/'self_div83'/'out' / 'direct'
    srf_iras = rf.trans_srf(out_dir / 'transmission.txt', nch)[:,:,np.newaxis]
    trans_iras = rf.trans_lv2space(out_dir / 'transmission.txt', npro, nch, nlv)
    trans_iras = np.concatenate((trans_iras,srf_iras),axis=2)
    bt_iras = rf.bt_clear(out_dir / 'radiance.txt', nch)
    return trans_iras,bt_iras
print('IRAS reconstruction done.')

############# Reconstruction done ######################################
####################### Convolve the transmission and radiance to specific channels #######################

# chs = ['ch1','ch3','ch4','ch5','ch6','ch7','ch8','ch9','ch10','ch11','ch12','ch13','ch14','ch15']
chs = ['ch11','ch12','ch13']
trans_iasi,rad_iasi = construct_iasi()
trans_iras,bt_iras = construct_iras()
nlv = 102 # 101 levels and surface-to-space
trans_true = np.zeros((len(chs),npro,nlv))
bt_true = np.zeros((npro,len(chs)))
for m in  range(3):
    print('The {:d}th channel'.format(m))
    srf = pd.read_excel(srf_dir,sheet_name=chs[m])
    wv = ch_info[chs[m]]['wn']
    index = ch_info[chs[m]]['range']
    scale = ch_info[chs[m]]['scale']
    offset = ch_info[chs[m]]['offset']
    cv = ch_info[chs[m]]['cv']

    for ii in range(npro):
        # print('The {:d}th profile'.format(ii))
        monorad = rad_iasi[ii,index]
        # print(monorad.shape,wv.shape)
        rad = mf.convolv(monorad,wv,srf.iloc[:,0].values,srf.iloc[:,1].values)
        bt_true[ii,m] = mf.plank_inv(rad,scale,offset,cv)
        for jj in range(nlv):
            monotrans = trans_iasi[ii,index,jj]
            trans_true[m,ii,jj] = mf.convolv(monotrans,wv,srf.iloc[:,0].values,srf.iloc[:,1].values)
    print('The {:d}th channel done.'.format(m))
# ############# Save the ground-truth and RTTOV's transmission and radiance to new file ##################
fsavename = 'dataset_101L_ec83_3ch_nadir.HDF'
fsave = Path('/mnt/hgfs/DL_transmitance/revised datasets')/fsavename
k = 0
childs = sorted([x for x in pro_dir.glob('*') if x.is_dir()])
# print(parent,childs)
for child in childs:
    if k==0:
        profiles = rf.read_profile(child)
    else:
        profiles = np.vstack((profiles,rf.read_profile(child)))
    k += 1

############ surface emissivity ###################
import read_files as rf
nch = 3
# # dir_flag = '*-*'
emis = emi(iras_dir,'*83',nch)

with h5py.File(fsave,'w') as g:
    x = g.create_dataset('X',data=profiles)
    x.attrs['name'] = 'Input features'
    x.attrs['variables'] = 't,q,o3,p2m,t2m,u2m,v2m,srft,srftype,longitude,latitude,zenith,elevation,month'
    y = g.create_dataset('Y',data=trans_true)
    y.attrs['name'] = 'Level-to-space and surface-to-space transmission ground-truth(convolved from IASI)'
    iasi = g.create_dataset('BT_true',data=bt_true)
    iasi.attrs['name'] = 'Brightness Temperature convolved from IASI.'
    iras_trans = g.create_dataset('transmission_RTTOV',data=trans_iras)
    iras_trans.attrs['name'] = 'Level-to-space and surface-to-space transission directly computed from RTTOV.'
    iras_bt = g.create_dataset('BT_RTTOV',data=bt_iras)
    iras_bt.attrs['name'] = 'TOA brightness temperature directly computed with RTTOV.'
    iras_emissivity = g.create_dataset('emissivity',data=emis)
    iras_emissivity.attrs['name'] = 'Surface emissivity(IRAS) computed with RTTOV'
