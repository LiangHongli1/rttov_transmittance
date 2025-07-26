# -*- coding: utf-8 -*-
"""
@ Time: 2019-12-2
@ author: LiangHongli
@ Mail: Helen_Liang1@outlook.com
With the new clear-sky profile, make trining data for 1-14 channels of IRAS(except for channel 2).
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
pro_dir = Path('/home/sky/rttov121/rttov_test/profile-datasets/new_set')
dir_flag = '*redai'
prodirs = [x for x in sorted(pro_dir.glob(dir_flag))]
# # print(prodirs)
# ################# Reconstruct transmission and radiance #############
npros = []
for idir in prodirs:
    n = len([x for x in idir.glob('*')])
    npros.append(n)

def construct_iasi():
    # nchs = [1815,400,1767,744] # 3 files save nchs results respectively.
    # nchs = [2167]
    nlv = 101
    # wn_range = ['38-1853','2052-2452','2452-4219','5839-6583'] # IASI channel, tag of file
    # wn_range = ['38-1853','2452-4219', '5839-6583']
    # for k in range(1):
    nch = 2167
    # ftrans = 'transmission_' + wn_range[k] + '.txt'
    # frad = 'radiance_' + wn_range[k] + '.txt'
    ftrans = 'transmission.txt'
    frad = 'radiance.txt'
    # print(prodirs)
    for j,jdir in enumerate(sorted(iasi_dir.glob(dir_flag))):
        if j<=13:
            continue
        fdir = jdir/'out'/'direct'
        print(str(fdir))
        npro = npros[j]
        # if k==0:
        if j==14:
            srf_iasi = rf.trans_srf(fdir / ftrans, nch)[:,:,np.newaxis]
            trans_iasi1 = rf.trans_lv2space(fdir/ftrans, npro, nch, nlv)
            trans_iasi1 = np.concatenate((trans_iasi1,srf_iasi),axis=2)
            rad_iasi1 = rf.rad_clear(fdir/frad, npro, nch)
        else:
            srf_iasi = rf.trans_srf(fdir / ftrans, nch)[:,:,np.newaxis]
            temp = np.concatenate((rf.trans_lv2space(fdir/ftrans, npro, nch, nlv),srf_iasi),axis=2)
            print(temp.shape)
            trans_iasi1 = np.append(trans_iasi1,temp,axis=0)
            rad_iasi1 = np.append(rad_iasi1,rf.rad_clear(fdir/frad, npro, nch),axis=0)

        # j += 1
        # elif k==1:
        #     if j==0:
        #         srf_iasi = rf.trans_srf(fdir / ftrans, nch)[:, :, np.newaxis]
        #         trans_iasi2 = rf.trans_lv2space(fdir / ftrans, npro, nch, nlv)
        #         trans_iasi2 = np.concatenate((trans_iasi2,srf_iasi),axis=2)
        #         rad_iasi2 = rf.rad_clear(fdir/frad, npro, nch)
        #     else:
        #         srf_iasi = rf.trans_srf(fdir / ftrans, nch)[:,:,np.newaxis]
        #         temp = np.concatenate((rf.trans_lv2space(fdir/ftrans, npro, nch, nlv),srf_iasi),axis=2)
        #         print(temp.shape)
        #         trans_iasi2 = np.append(trans_iasi2,temp,axis=0)
        #         rad_iasi2 = np.append(rad_iasi2,rf.rad_clear(fdir/frad, npro, nch),axis=0)
        # elif k==2:
        #     if j==0:
        #         srf_iasi = rf.trans_srf(fdir / ftrans, nch)[:, :, np.newaxis]
        #         trans_iasi3 = rf.trans_lv2space(fdir / ftrans, npro, nch, nlv)
        #         trans_iasi3 = np.concatenate((trans_iasi3,srf_iasi),axis=2)
        #         rad_iasi3 = rf.rad_clear(fdir/frad, npro, nch)
        #     else:
        #         srf_iasi = rf.trans_srf(fdir / ftrans, nch)[:,:,np.newaxis]
        #         temp = np.concatenate((rf.trans_lv2space(fdir/ftrans, npro, nch, nlv),srf_iasi),axis=2)
        #         print(temp.shape)
        #         trans_iasi3 = np.append(trans_iasi3,temp,axis=0)
        #         rad_iasi3 = np.append(rad_iasi3,rf.rad_clear(fdir/frad, npro, nch),axis=0)
        # elif k==3:
        #     if j==0:
        #         srf_iasi = rf.trans_srf(fdir / ftrans, nch)[:, :, np.newaxis]
        #         trans_iasi4 = rf.trans_lv2space(fdir / ftrans, npro, nch, nlv)
        #         trans_iasi4 = np.concatenate((trans_iasi4,srf_iasi),axis=2)
        #         rad_iasi4 = rf.rad_clear(fdir/frad, npro, nch)
        #     else:
        #         srf_iasi = rf.trans_srf(fdir / ftrans,nch)[:,:,np.newaxis]
        #         temp = np.concatenate((rf.trans_lv2space(fdir/ftrans, npro, nch, nlv),srf_iasi),axis=2)
        #         print(temp.shape)
        #         trans_iasi4 = np.append(trans_iasi4,temp,axis=0)
        #         rad_iasi4 = np.append(rad_iasi4,rf.rad_clear(fdir/frad, npro, nch),axis=0)

    # if k==0:
    #     return trans_iasi1,rad_iasi1
    # else:
    return trans_iasi1, rad_iasi1#, trans_iasi4, rad_iasi4

print('IASI reconstruction done.')

def construct_iras():
    nch = 3
    nlv = 101
    for k,kdir in enumerate(sorted(iras_dir.glob(dir_flag))):
        if k<=13:
            continue
        npro = npros[k]
        out_dir = kdir/'out' / 'direct'
        if k == 14:
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
####################### Convolve the transmission and radiance to specific channels #######################

# chs = ['ch1','ch3','ch4','ch5','ch6','ch7','ch8','ch9','ch10','ch11','ch12','ch13','ch14','ch15']
chs = ['ch11','ch12','ch13']
trans_iasi1,rad_iasi1 = construct_iasi()
# # trans_iasi2, rad_iasi2, trans_iasi3, rad_iasi3 = construct_iasi()
trans_iras,bt_iras = construct_iras()
# #
nprof = np.sum(np.array(npros)[14:])
# # nprof = 83
nlv = 102 # 137 levels and surface-to-space
trans_true = np.zeros((len(chs),nprof,nlv))
bt_true = np.zeros((nprof,len(chs)))
for m in  range(3):
    print('The {:d}th channel'.format(m))
    srf = pd.read_excel(srf_dir,sheet_name=chs[m])
    wv = ch_info[chs[m]]['wn']
    index = ch_info[chs[m]]['range']
    scale = ch_info[chs[m]]['scale']
    offset = ch_info[chs[m]]['offset']
    cv = ch_info[chs[m]]['cv']
#     if m<=8:
#         # continue
    trans_iasi = trans_iasi1
    rad_iasi = rad_iasi1
#     # elif m>11:
#     #     trans_iasi = trans_iasi3
#     #     rad_iasi = rad_iasi3
#     # else:
#     #     trans_iasi = trans_iasi2
#     #     rad_iasi = rad_iasi2
#
    for ii in range(nprof):
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

# propath = Path('/home/sky/rttov121/rttov_test/profile-datasets/new_set')
# prodirs = [x for x in sorted(propath.glob('*'))]
# k = 0
# # import subprocess
# for parent in prodirs:
#     childs = sorted([x for x in parent.glob('*') if x.is_dir()])
#     # print(parent,childs)
#     for child in childs:
#         # fcc = child/'atm'/'cc.txt'
#         # subprocess.call(['rm', '-rf', str(fcc)])
#         if k==0:
#             profiles = rf.read_profile(child)
#         else:
#             profiles = np.vstack((profiles,rf.read_profile(child)))
#         k += 1
    # if k==0:
    #     profiles = rf.read_profile(parent)
    # else:
    #     profiles = np.vstack((profiles,rf.read_profile(parent)))
    # k += 1

############ surface emissivity ###################
# import read_files as rf
# rttpath = Path(r'/home/sky/rttov121/rttov_test/my_test_'+version+'.1.gfortan/iras')
# # femi = '/home/sky/rttov121/rttov_test/my_test.1.gfortan/iras/div83/out/direct/emissivity_out.txt'
nch = 3

# emis = emi(iras_dir,dir_flag,nch)
# with h5py.File(fsave,'w') as f:
    # x = np.concatenate((profiles[:,:557],profiles[:,694:]),axis=1)
#     cc = profiles[:,557:694]
#     pro = f.create_dataset('X',data=profiles,compression='gzip')
#     pro.attrs['name'] = 'p(101), t(101), q(101), o3(101), p2m, t2m, u2m, v2m, st, srftype(land=0,sea=1,seaice=2), lon, lat, ele(km),month'
#     cloud = f.create_dataset('cloud ratio',data=cc)
#     cloud.attrs['name'] = 'vertically cloud ratio'
#     f['emissivity'] = emis
#     f['transmittance_rttov'] = trans_iras
#     f['transmittance_true'] = trans_true
#     f['BT_rttov'] = bt_iras
#     f['BT_true'] = bt_true
fsavename = 'dataset_ifs101L_latter.HDF'
fsave = Path('/mnt/hgfs/DL_transmitance/revised datasets/final')/fsavename
with h5py.File(fsave,'w') as g:
    # g.attrs['name'] = 'This is a independent dataset for testing GBT, 200 profiles in 0-30, 200 in 30-60, 200 in 60-90'
    # x = g.create_dataset('X',data=profiles)
    # x.attrs['name'] = 'Input features'
    # x.attrs['variables'] = 't,q,o3,p2m,t2m,u2m,v2m,srft,srftype,longitude,latitude,elevation,month'
    y = g.create_dataset('Y',data=trans_true)
    y.attrs['name'] = 'Level-to-space and surface-to-space transmission ground-truth(convolved from IASI)'
    iasi = g.create_dataset('BT_true',data=bt_true)
    iasi.attrs['name'] = 'Brightness Temperature convolved from IASI.'
    iras_trans = g.create_dataset('transmission_RTTOV',data=trans_iras)
    iras_trans.attrs['name'] = 'Level-to-space and surface-to-space transission directly computed from RTTOV.'
    iras_bt = g.create_dataset('BT_RTTOV',data=bt_iras)
    iras_bt.attrs['name'] = 'TOA brightness temperature directly computed with RTTOV.'
    # iras_emissivity = g.create_dataset('emissivity',data=emis)
    # iras_emissivity.attrs['name'] = 'Surface emissivity(IRAS) computed with RTTOV'


