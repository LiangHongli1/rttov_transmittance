# -*- coding: utf-8 -*-
"""
@ Time: 2020/2/23
@ author: LiangHongli
@ Mail: Helen_Liang1@outlook.com
With the cloud detection algorithm to mask FY3A/IRAS.
0 = cloud, 1 = probably cloud, 2 = probably clear, 3 = clear
"""
import numpy as np
from pathlib import Path
import h5py

def onech(bt11,bt139,land_sea_mask,day_night_flag):
    cld_mask11 = np.ones(bt11.shape,dtype=int)*255
    cld_mask139 = np.ones(bt11.shape,dtype=int)*255
    thresh_sea_low = 267.
    thresh_sea_mid = 270.
    thresh_sea_high = 273.
    if day_night_flag=='Day':
        thresh_land_high = 305.
        thresh_land_mid = 300.
    else:
        thresh_land_high = 297.5
        thresh_land_mid = 292.5

    r_sea,c_sea = np.where(land_sea_mask==3) # ocean
    r_land,c_land = np.where(land_sea_mask==1)
    condition = bt11[r_sea,c_sea]>=thresh_sea_high
    cld_mask11[r_sea,c_sea][condition] = 3
    condition = np.logical_and(bt11[r_sea,c_sea]>=thresh_sea_mid,bt11[r_sea,c_sea]<thresh_sea_high)
    cld_mask11[r_sea,c_sea][condition] = 2
    condition = bt11[r_sea,c_sea]<thresh_sea_low
    cld_mask11[r_sea,c_sea][condition] = 0
    condition = np.logical_and(bt11[r_sea,c_sea]>=thresh_sea_low,bt11[r_sea,c_sea]<thresh_sea_mid)
    cld_mask11[r_sea,c_sea][condition] = 1

    condition = bt11[r_land,c_land]>=thresh_land_high
    cld_mask11[r_land,c_land][condition] = 3
    condition = bt11[r_land,c_land]<thresh_land_mid
    cld_mask11[r_land,c_land][condition] = 1
    condition = np.logical_and(bt11[r_land,c_land]>=thresh_land_mid,bt11[r_land,c_land]<thresh_land_high)
    cld_mask11[r_land,c_land][condition] = 2
########################## the second: 13.9μm #########################
    condition = bt139<222.
    cld_mask139[condition] = 0
    condition = np.logical_and(bt139<224.,bt139>=222.)
    cld_mask139[condition] = 1
    condition = bt139>=226.
    cld_mask139[condition] = 3
    condition = np.logical_and(bt139<226.,bt139>=224.)
    cld_mask139[condition] = 2

    cld_mask = np.minimum(cld_mask11,cld_mask139)
    return cld_mask

def BTD(bt11,bt39,land_sea_mask,day_night_flag):
    cld_mask = np.ones(bt11.shape,dtype=int)*255
    thresh_land_high = -11.
    thresh_land_mid = -13.
    thresh_land_low = -15.
    if day_night_flag=='Day':
        thresh_sea_low = -10.
        thresh_sea_mid = -8.
        thresh_sea_high = -6.
    else:
        thresh_sea_low = 1.25
        thresh_sea_mid = 1.
        thresh_sea_high = -1.

    r_sea,c_sea = np.where(land_sea_mask==3) # ocean
    difer_sea = bt11[r_sea,c_sea]-bt39[r_sea,c_sea]
    condition = difer_sea>=thresh_sea_high
    cld_mask[r_sea,c_sea][condition] = 3
    condition = np.logical_and(difer_sea>=thresh_sea_mid,difer_sea<thresh_sea_high)
    cld_mask[r_sea,c_sea][condition] = 2
    condition = difer_sea<thresh_sea_low
    cld_mask[r_sea,c_sea][condition] = 0
    condition = np.logical_and(difer_sea>=thresh_sea_low,difer_sea<thresh_sea_mid)
    cld_mask[r_sea,c_sea][condition] = 1

    if day_night_flag=='Day':
        r_land,c_land = np.where(land_sea_mask==1) # land
        difer_land = bt11[r_land,c_land]-bt39[r_land,c_land]
        condition = difer_land>=thresh_land_high
        cld_mask[r_land,c_land][condition] = 3
        condition = np.logical_and(difer_land>=thresh_land_mid,difer_land<thresh_land_high)
        cld_mask[r_land,c_land][condition] = 2
        condition = difer_land<thresh_land_low
        cld_mask[r_land,c_land][condition] = 0
        condition = np.logical_and(difer_land>=thresh_land_low,difer_land<thresh_land_mid)
        cld_mask[r_land,c_land][condition] = 1

    return cld_mask

def reflectance(r88,r69,land_sea_mask):
    cld_mask1 = np.ones(r69.shape,dtype=int)*255
    cld_mask2 = np.ones(r88.shape,dtype=int)*255
    thresh_sea_low = 0.055
    thresh_sea_mid = 0.04
    thresh_sea_high = 0.03
    thresh_land_low = 0.22
    thresh_land_mid = 0.18
    thresh_land_high = 0.14

    r_sea,c_sea = np.where(land_sea_mask==3) # ocean
    r_land,c_land = np.where(land_sea_mask==1)
    condition = r88[r_sea,c_sea]<=thresh_sea_high
    cld_mask1[r_sea,c_sea][condition] = 3
    condition = np.logical_and(r88[r_sea,c_sea]<=thresh_sea_mid,r88[r_sea,c_sea]>thresh_sea_high)
    cld_mask1[r_sea,c_sea][condition] = 2
    condition = r88[r_sea,c_sea]>thresh_sea_low
    cld_mask1[r_sea,c_sea][condition] = 0
    condition = np.logical_and(r88[r_sea,c_sea]<=thresh_sea_low,r88[r_sea,c_sea]>thresh_sea_mid)
    cld_mask1[r_sea,c_sea][condition] = 1

    condition = r69[r_land,c_land]<=thresh_land_high
    cld_mask1[r_land,c_land][condition] = 3
    condition = np.logical_and(r69[r_land,c_land]<=thresh_land_mid,r69[r_land,c_land]>thresh_land_high)
    cld_mask1[r_land,c_land][condition] = 2
    condition = np.logical_and(r69[r_land,c_land]<=thresh_land_low,r69[r_land,c_land]<thresh_land_mid)
    cld_mask1[r_land,c_land][condition] = 1
    condition = r69[r_land,c_land]>thresh_land_low
    cld_mask1[r_land,c_land][condition] = 0
########################## the second: r0.88μm/r0.69μm #########################
    thresh_sea_low = 0.95
    thresh_sea_mid = 0.9
    thresh_sea_high = 0.85
    thresh_land_low = 1.95
    thresh_land_mid = 1.9
    thresh_land_high = 1.85

    div_sea = r88[r_sea,c_sea]/r69[r_sea,c_sea]
    div_land = r88[r_land,c_land]/r69[r_land,c_land]
    condition = div_sea<=thresh_sea_high
    cld_mask2[r_sea,c_sea][condition] = 3
    condition = np.logical_and(div_sea<=thresh_sea_mid,div_sea>thresh_sea_high)
    cld_mask2[r_sea,c_sea][condition] = 2
    condition = div_sea>thresh_sea_low
    cld_mask2[r_sea,c_sea][condition] = 0
    condition = np.logical_and(div_sea<=thresh_sea_low,div_sea>thresh_sea_mid)
    cld_mask2[r_sea,c_sea][condition] = 1

    condition = div_land<=thresh_land_high
    cld_mask2[r_land,c_land][condition] = 3
    condition = np.logical_and(div_land<=thresh_land_mid,div_land>thresh_land_high)
    cld_mask2[r_land,c_land][condition] = 2
    condition = np.logical_and(div_land<=thresh_land_low,div_land<thresh_land_mid)
    cld_mask2[r_land,c_land][condition] = 1
    condition = div_land>thresh_land_low
    cld_mask2[r_land,c_land][condition] = 0

    cld_mask = np.minimum(cld_mask1,cld_mask2)
    return cld_mask

def main():
    datapath = Path(r'G:\DL_transmitance\O-B_validation\FY3A_OBS')
    savepath = Path(r'G:\DL_transmitance\O-B_validation\FY3A_CLD')

    for parent in datapath.iterdir():
        if (savepath/parent.name).exists():
            pass
        else:
            (savepath/parent.name).mkdir()

        for file in parent.glob('*.HDF'):
            # if 'CL' in file.name:
            #     file.unlink()
            #     continue
            with h5py.File(file,'r') as f:
                try:
                    bt = f['FY3A_IRAS_TB'][:]
                    lats = f['Latitude'][:]
                    lons = f['Longitude'][:]
                    day_flag = f.attrs['Day Or Night Flag'][:]
                    lsmask = f['LandSeaMask'][:] # 1=land, 2=inland water, 3=sea, 5=bord
                except:
                    continue

            mask1 = onech(bt[8,:,:],bt[4,:,:],lsmask,day_flag) # channel 9 and 5
            mask2 = BTD(bt[8,:,:],bt[18,:,:],lsmask,day_flag)
            # if day_flag=='Day':
            #     r88 = bt[21,:,:]/30.8
            #     r69 = bt[20,:,:]/45.2084
            #     mask3 = reflectance(r88,r69,lsmask)
            #     cld_mask = np.min()
            # else:
            cld_mask = np.minimum(mask1,mask2)

            fnew_name = file.name.split('.')[0]+'_CL.HDF'
            r1,r2 = lats.shape[0],bt.shape[1]
            if r1==r2:
                pass
            else:
                lats = lats[:r2,:]
                lons = lons[:r2,:]

            with h5py.File(savepath/parent.name/fnew_name,'w') as f:
                cld = f.create_dataset('CloudMask',data=cld_mask,compression='gzip')
                lats = f.create_dataset('Latitude',data=lats,compression='gzip')
                lons = f.create_dataset('Longitude',data=lons,compression='gzip')

            print('File %s done' % file.name)


if __name__ == '__main__':
    main()
