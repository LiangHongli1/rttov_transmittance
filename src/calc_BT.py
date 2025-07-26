# -*- coding: utf-8 -*-
"""
@2019/6/17
@author: LiangHongli
Computing the TOA brightness temperature with transmittances calculated by GBT.
Convolving the radiances of IASI calculated by RTTOV.
"""
import h5py
from pathlib import Path
import scipy
import numpy as np
import matplotlib.pyplot as plt
import my_funcs
import read_files as rf
from LUT import trans_limit

def emi(emipath,dir_flag,nch):
    '''
    将9个文件中893条廓线的地表发射率重新保存为一个向量
    :param emipath: 文件所在路径
    :return: 893条廓线的地表发射率
    '''
    emifiles = []
    childs = sorted([x for x in emipath.glob(dir_flag)])
    print(childs)
    for j in range(len(childs)):
        child = childs[j]
        f = emipath/child/'out'/'direct'/'emissivity_out.txt'
        emifiles.append(f)
    #emifiles = [x for x in emipath.glob('893_*/out/direct/emissivity_*.txt')]
    #print(emifiles)
    for k,f in enumerate(emifiles):
        if k==0:
            emis = rf.emisi_suf(f,nch)
        else:
            emis = np.append(emis,rf.emisi_suf(f,nch),axis=0)

    return emis

def t_srf(tspath):
    '''
    地表温度重新存为一个向量
    :param tspath: 存储地表温度的文件所在路径
    :return: 地表温度向量
    '''
    tsfiles = [x for x in tspath.glob('**/**/ground/skin.txt')]
    for k,f in enumerate(tsfiles):
        if k==0:
            tsurf = rf.temp_suf(f)
        else:
            tsurf = np.append(tsurf,rf.temp_suf(f))

    return tsurf

def tao_srf(fpath):
    taofiles = []
    for j in range(1,10):
        child = str(j)+'30'
        f = fpath/child/'out'/'direct'/'transmission.txt'
        taofiles.append(f)
    #taofiles = [x for x in fpath.glob('893_*/out/direct/transmission.txt')]
    nch = 15
    for k,f in enumerate(taofiles):
        if k==0:
            taosurf = rf.trans_srf(f,nch)
        else:
            taosurf = np.append(taosurf,rf.trans_srf(f,nch),axis=0)

    return taosurf

def bts_iras(fpath,nch):
    files = []
    for j in range(1,11):
        child = str(j)+'30'
        f = fpath/child/'out'/'direct'/'radiance.txt'
        files.append(f)
    #files = [x for x in fpath.glob('893_*/out/direct/radiance.txt')]
    for k,f in enumerate(files):
        if k==0:
            bts = rf.bt_clear(f,nch)
        else:
            bts = np.append(bts,rf.bt_clear(f,nch),axis=0)

    return bts

def radj(tlevel,taolevel,tsurf,trans_suf,scale,offset,cv):
    '''
    用一条廓线中每个等压面和地表的温度、透过率，计算radiance
    :param tlevel: 等压面的温度，向量
    :param taolevel: 等压面的透过率，向量
    :param tsurf: 地表温度，标量
    :param trans_suf: 地表到space的透过率，向量
    :return:
    '''
    n = tlevel.shape[0]
    tlevels = np.append(tlevel,tsurf)
    taolevels = np.append(taolevel,trans_suf)
    Ljs = []
    Ljs_suf = []
    for k in range(n):
        # if taolevels[k]<=trans_limit:
        #     Ljs.append(0)
        #     Ljs_suf.append(0)
        # else:
        #     if taolevels[k]-taolevels[k+1]<0:
        #         taolevels[k+1] = taolevels[k]-trans_limit

        rad_layer = my_funcs.plank(tlevels[k],scale,offset,cv)+my_funcs.plank(tlevels[k+1],scale,offset,cv)
        Lj = 0.5*(taolevels[k]-taolevels[k+1])*rad_layer
        Ljs.append(Lj)
        ratio = trans_suf * trans_suf / (taolevels[k] * taolevels[k + 1])
        # if ratio>trans_suf*70:
        #     ratio = trans_suf*70
        Ljs_suf.append(Lj*ratio)
    Ljs = np.array(Ljs)
    Ljs_suf = np.array(Ljs_suf)

    return Ljs,Ljs_suf

def bt_GBT(trans_preds,tlevel,emis,tsrf,tskin,trans_srf,scale,offset,cv):
    '''
    用GBT计算的透过率计算大气顶亮温
    '''
    radiances = []
    n = tlevel.shape[0]
    radianceair = []
    radiancesrf = []
    for k in range(n):
        # print(tlevel[k,:],tao_preds[k,:],tsurf[k],taosurf[k])
        Ljs,Ljs_suf = radj(tlevel[k,:],trans_preds[k,:],tsrf[k],trans_srf[k],scale,offset,cv)
        # print(Ljs)
        rad_suf = my_funcs.plank(tskin[k],scale,offset,cv)
        if trans_srf[k]<trans_limit:
            rad_clr = np.sum(Ljs) + (1-emis[k])*np.sum(Ljs_suf)
            rad_air = np.sum(Ljs)
        else:
            rad_clr = trans_srf[k]*emis[k]*rad_suf + np.sum(Ljs) + (1-emis[k])*np.sum(Ljs_suf)
            rad_air = np.sum(Ljs)
        radiances.append(rad_clr)
        radianceair.append(rad_air)
        radiancesrf.append((1-emis[k])*np.sum(Ljs_suf))

    print('radiance',radiances[:5])
    # print('surface radiance',radiancesrf[:10])
    radiances = np.array(radiances)
    radianceair = np.array(radianceair)
    bt_clr = my_funcs.plank_inv(radiances,scale,offset,cv)
    bt_air = my_funcs.plank_inv(radianceair,scale,offset,cv)
    
    return bt_clr,bt_air

