# -*- coding: utf-8 -*-
"""
@ Time: 2019-10-31
@ author: LiangHongli
@ Mail: Helen_Liang1@outlook.com
Make dataset for IRAS channel 11 and 13 to train GBT models.
"""
import numpy as np
import pandas as pd
from pathlib import Path
import h5py
import my_funcs as mf
import read_files as rf
from LUT import ch_info

########### Make channels.txt and lprofile.txt ########################
#ch_iasi = np.arange(2051,4219)
#ch_iras = np.array([11,13])
#npro = 100
#for k in range(2,10):
#    childir = '893_'+str(k)
#    wholedir_iasi = Path('/home/sky/rttov121/rttov_test/tests.0/iasi')/childir/'in'
#    wholedir_iras = Path('/home/sky/rttov121/rttov_test/tests.0/iras')/childir/'in'
#    if k==9:
#        npro = 93
#    mf.generate_chpr_txt(wholedir_iasi,ch_iasi,npro)
#    mf.generate_chpr_txt(wholedir_iras,ch_iras,npro)

################### End! ##################################
################################################ Make dataset start ###############################################################
iasi_dir = '/home/sky/rttov121/rttov_test/my_test.1.gfortran/iasi'
iras_dir = '/home/sky/rttov121/rttov_test/my_test.1.gfortran/iras'
srf_dir = '/mnt/hgfs/DL_transmitance/fy3_iras_znsys_srfch1-26/IRAS_srf.xlsx'

################# Reconstruct transmission and radiance #############
nch = 2169
nlv = 91
for k in range(1,10):
    print('The {:d}th directory'.format(k))
    if k==9:
        npro = 93
    else:
        npro = 100

    inout = '893_'+str(k)
    out_dir = Path(iasi_dir)/inout/'out'/'direct'
    if k==1:
        trans_iasi = rf.trans_lv2space(out_dir/'transmission.txt', npro, nch, nlv)
        rad_iasi = rf.rad_clear(out_dir/'radiance.txt', npro, nch)
    else:
        trans_iasi = np.append(trans_iasi,rf.trans_lv2space(out_dir/'transmission.txt', npro, nch, nlv),axis=0)
        rad_iasi = np.append(rad_iasi,rf.rad_clear(out_dir/'radiance.txt', npro, nch),axis=0)
print('IASI reconstruction done.')

nch = 2
for k in range(1, 10):
    if k == 9:
        npro = 93
    else:
        npro = 100

    inout = '893_' + str(k)
    out_dir = Path(iras_dir) / inout / 'out' / 'direct'
    if k == 1:
        trans_iras = rf.trans_lv2space(out_dir / 'transmission.txt', npro, nch, nlv)
        bt_iras = rf.bt_clear(out_dir / 'radiance.txt', npro, nch)
    else:
        trans_iras = np.append(trans_iras, rf.trans_lv2space(out_dir / 'transmission.txt', npro, nch, nlv), axis=0)
        bt_iras = np.append(bt_iras, rf.bt_clear(out_dir / 'radiance.txt', npro, nch), axis=0)
print('IRAS reconstruction done.')

############# Reconstruction done ######################################
####################### Convolve the transmission and radiance to specific channels #######################

chs = ['ch11','ch13']

nprof = bt_iras.shape[0]
trans_true = np.zeros((len(chs),nprof,nlv))
bt_true = np.zeros((nprof,len(chs)))
for m in range(len(chs)):
    srf = pd.read_excel(srf_dir,sheet_name=chs[m])
    wv = ch_info[chs[m]]['wn']
    index = ch_info[chs[m]]['range']
    scale = ch_info[chs[m]]['scale']
    offset = ch_info[chs[m]]['offset']
    cv = ch_info[chs[m]]['cv']

    for ii in range(nprof):
        monorad = rad_iasi[ii,index]
        rad = mf.convolv(monorad,wv,srf.iloc[:,0].values,srf.iloc[:,1].values)
        bt_true[ii,m] = mf.plank_inv(rad,scale,offset,cv)
        for jj in range(nlv):
            monotrans = trans_iasi[ii,index,jj]
            trans_true[m,ii,jj] = mf.convolv(monotrans,wv,srf.iloc[:,0].values,srf.iloc[:,1].values)
    print('The {:d}th channel done.'.format(m))
############# Save the ground-truth and RTTOV's transmission and radiance to new file ##################
fx = '/mnt/hgfs/DL_transmitance/revised datasets/training_ch12_IFS91l_894.HDF'
fsave = '/mnt/hgfs/DL_transmitance/revised datasets/dataset_IFS91_ch1113.HDF'

with h5py.File(fx,'r') as f:
    X = f['X'][:]

with h5py.File(fsave,'w') as g:
    x = g.create_dataset('X',data=X)
    x.attrs['name'] = 'Input features'
    x.attrs['variables'] = 'p,t,q,o3,co2,co,ch4,n2o, 91 numbers for every variable'
    y = g.create_dataset('Y',data=trans_true)
    y.attrs['name'] = 'Transmission ground-truth(convolved from IASI)'
    iasi = g.create_dataset('BT_true',data=bt_true)
    iasi.attrs['name'] = 'Brightness Temperature convolved from IASI.'
    iras_trans = g.create_dataset('transmission_RTTOV',data=trans_iras)
    iras_trans.attrs['name'] = 'Level-to-space transission directly computed from RTTOV.'
    iras_bt = g.create_dataset('BT_RTTOV',data=bt_iras)
    iras_bt.attrs['name'] = 'TOA brightness temperature directly computed from RTTOV.'


