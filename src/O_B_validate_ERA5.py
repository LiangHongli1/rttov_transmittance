# -*- coding: utf-8 -*-
"""
@ Time: 2019/10/17
@ author: LiangHongli
@ Mail: Helen_Liang1@outlook.com

Simulating TOA brightness temperature through RTTOV with GRAPES forecasting field profiles.
Firstly matching profile with IRAS observation. Only the dots where time_diff<3h are simulated.
And the max cloud ratio of all levels should be less than 0.3.
Then: simulating BT - IRAS observation BT
Notice: Generally, O-B should do spatial interpolation and time interpolation. Currently, we only do
spatial interpolation with bilinear interpolation method, as there is only one trajectory to be simulated.
"""
import numpy.ma as ma
import numpy as np
############# To make PyInstaller work properly, these modules need to be imported for now ###############
import numpy.random.common
import numpy.random.bounded_integers
import numpy.random.entropy
###########################################################################################################
import h5py
from pathlib import Path
import pandas as pd
from scipy.spatial import KDTree
from scipy.interpolate import interp1d
import os
import subprocess
from my_funcs import generate_chpr_txt,rh2q,interp_profile,bilinear_interp,calc_p_2m,kg2ppmv,ppmv2kg
import read_files as rf
from plot_map_funcs import plot_profile_var,plot_single_var,map_scatter
import matplotlib.pyplot as plt

######################## Functions to use when resaving the profiles ####################
def resave_angles(fdir,zenangle,azangle,sunzenangle,sunazangle,lat,lon,elev):
    f = open(fdir,'w')
    f.write('&angles\n')
    zen = '{:>10s}{:5s}{:>2s}{:>8.2f}'.format('zenangle','','=',zenangle) +'\n'
    f.write(zen)
    # az = '  ' + 'azangle' + ' '*5 + '=' +' ' + '{:.1f}'.format(azangle) + '\n'
    # f.write(az)
    # sunzen = '  ' + 'sunzenangle' + ' '*5 + '=' +' ' + '{:.1f}'.format(sunzenangle) + '\n'
    # f.write(sunzen)
    # sunaz = '  ' + 'sunaznangle' + ' '*5 + '=' +' ' + '{:.1f}'.format(sunazangle) + '\n'
    # f.write(sunaz)
    lati = '{:>10s}{:5s}{:>2s}{:>8.2f}'.format('latitude','','=',lat) +'\n'
    f.write(lati)
    if lon<0:
        long = '{:>11s}{:5s}{:>2s}{:>8.2f}'.format('longitude','','=',lon) + '\n'
    else:
        long = '{:>11s}{:5s}{:>2s}{:>8.2f}'.format('longitude','','=',lon) + '\n'
    f.write(long)
    ele = '{:>11s}{:5s}{:>2s}{:>8.2f}'.format('elevation','','=',elev) + '\n'
    f.write(ele)
    f.write('/')
    f.write('\n')
    f.close()
    return

def resave_date(fdir,date):
    f = open(fdir,'w')
    fdate = date[:4] + ' '*3 + date[4:6] + ' '*3 + date[-2:] + ' '*3 + '0' +' '*3 + '0'+' '*3 + '0'
    f.write(fdate)
    f.close()
    return

def resave_gas_unit(fdir,unit):
    f = open(fdir,'w')
    f.write('&units\n')
    f.write('{:>15s}{:2d}'.format('gas_units = ',unit)+'\n')
    f.write('/')
    f.write('\n')
    f.close()
    return

def resave_2m(fdir,t,q,p,u,v):
    f = open(fdir,'w')
    f.write('&s2m\n')
    st = ' '*2+'s0%t'+' '*4+'='+' '+'{:.3f}'.format(t)+'\n'
    f.write(st)
    sq = ' '*2+'s0%q'+' '*4+'='+' '+'{:.3f}'.format(q)+'\n'
    f.write(sq)
    sp = ' '*2+'s0%p'+' '*4+'='+' '+'{:.3f}'.format(p)+'\n'
    f.write(sp)
    su = ' '*2+'s0%u'+' '*4+'='+' '+'{:.3f}'.format(u)+'\n'
    f.write(su)
    sv = ' '*2+'s0%v'+' '*4+'='+' '+'{:.3f}'.format(v)+'\n'
    f.write(sv)
    f.write('/')
    f.write('\n')
    f.close()
    return

def resave_skin(fdir,srftype,t):
    f = open(fdir,'w')
    f.write('&skin\n')
    srft = '{:>13s}{:5s}{:>3s}{:2d}'.format('k0%surftype','','=',srftype)+'\n'
    f.write(srft)
    f.write('{:>6s}{:5s}{:>3s}{:7.2f}'.format('k0%t','','=',t)+'\n')
    f.write('/')
    f.write('\n')
    f.close()
    return

################################### End! ############################################

# prof_path = Path('/mnt/hgfs/DL_transmitance/O-B_validation/ERA5') # input('Please input the directory of profiles.')
# obs_path = Path('/mnt/hgfs/DL_transmitance/O-B_validation/FY3A_OBS') # input('Please input the directory of IRAS observation.')
# cld_path = Path('/mnt/hgfs/DL_transmitance/O-B_validation/FY3A_CLD') # input('Please input the directory of IRAS cloud mask.')
# prof_save_path = Path('/home/sky/rttov121/rttov_test/profile-datasets/GRAPES/20080714') # input('Please input the directory where new forms of profile are to be saved.')
# fp = Path('/mnt/hgfs/DL_transmitance/profiles/model_level_definition.xlsx')
# btsave_path = Path('/mnt/hgfs/DL_transmitance/O-B_validation')
# figpath = Path('/mnt/hgfs/DL_transmitance/figures/profile_analysis/gfs_t639/interp101')

prof_path = Path('G:/DL_transmitance/O-B_validation/ERA5') # input('Please input the directory of profiles.')
obs_path = Path('G:/DL_transmitance/O-B_validation/FY3A_OBS') # input('Please input the directory of IRAS observation.')
cld_path = Path('G:/DL_transmitance/O-B_validation/FY3A_CLD') # input('Please input the directory of IRAS cloud mask.')
fp = Path('G:/DL_transmitance/profiles/model_level_definition.xlsx')
btsave_path = Path('G:/DL_transmitance/O-B_validation')
figpath = Path('G:/DL_transmitance/figures/profile_analysis/gfs_t639/interp101')
ch_index = [10,11,12] # input('Please input the IRAS channel you are about to simulate')
dodate = '20080709'

fprof = prof_path/'ERA5_ensemble_mean_pressure_level_20080709.nc'
fprof2 = prof_path/'make_ERA5_pressure_level_20080709.HDF'
fprof_srf = prof_path/'ERA5_ensemble_mean_single_level_20080709.nc'
fobs = obs_path/dodate/'FY3A_IRASX_GBAL_L1_20080709_0610_017KM_MS.HDF'
fcld = cld_path/dodate/'FY3A_IRASX_GBAL_L1_20080709_0610_017KM_MS_CL.HDF'
df = pd.read_excel(fp,sheet_name='101L')
p101 = np.array(df.iloc[:,0].values,dtype=float)
###################################################################################
ftime = fobs.name.split('_')[5]
fhour = int(ftime[:2])
# parent_dir = prof_save_path/ftime
# if not parent_dir.exists():
#     parent_dir.mkdir(parents=True)
# else:
#     pass

############# read IRAS data and extract the ascending part #############################
bt_obs, lons_obs, lats_obs, senzenith,solzenith,azangle,sunazangle,lsmask,dem = rf.read_IRAS_3a(str(fobs))
cld_mask_obs = rf.read_cld(str(fcld))

for ii in range(270,290):
    if lats_obs[ii,0] < lats_obs[ii+1,0] and lats_obs[ii+2,0] < lats_obs[ii+3,0]:
        r1 = ii # each file contains both ascending and descending part, label the first row of ascending part
        break
    else:
        continue

for ii in range(r1,lats_obs.shape[0]-3):
    if lats_obs[ii,0] < lats_obs[ii+1,0] and lats_obs[ii+2,0] < lats_obs[ii+3,0]:
        continue
    else:
        r2 = ii
        break

# print(r1,r2)
bt = bt_obs[ch_index,r1:r2,:]
lons_obs = lons_obs[r1:r2,:]
lats_obs = lats_obs[r1:r2,:]
senzenith = senzenith[r1:r2,:]
solzenith = solzenith[r1:r2,:]
azangle = azangle[r1:r2,:]
sunazangle = sunazangle[r1:r2,:]
cld_mask_obs = cld_mask_obs[r1:r2,:]
lsmask = lsmask[r1:r2,:]
dem = dem[r1:r2,:]
# lons_obs[lons_obs<0] = lons_obs[lons_obs<0]+360

print('dem_shape:',dem.shape)
plt.figure(dpi=200)
map_scatter(lons_obs.reshape(-1),lats_obs.reshape(-1),bt[0,:,:].reshape(-1),[-30,31,-180,180],title='BT 7.43 200807090610',vmin=np.min(bt[0,:,:]),vmax=np.max(bt[0,:,:]))
figname = 'BT7.43_20080709_0006.png'
plt.savefig(figpath/figname)
#------------------------------ Mark the valid and clear-sky pixels ---------------------------
mask1 = np.logical_and(np.logical_and(cld_mask_obs==3, senzenith<=30),np.abs(lats_obs)<=30)# 0=definitely cloudy, 1=possibly cloudy, 2=possibly clear, 3=definitely clear
mask2 = np.logical_and(np.logical_and(bt[0]>150,bt[1]>150),bt[2]>150)
mask = np.logical_and(mask1, mask2)
rs,cs = np.where(mask)
nvalid = len(rs)
print('the number of valid obs: %d' % nvalid)
################################ end #####################################################

################### read profile data and matching it with IRAS observation ##############
_,_,_,cc_l100,_,lons_fore,lats_fore = rf.read_era5_pressure_level(fprof)
tmp_l100,q_l100,o3_l100,level = rf.read_make_era5(fprof2)
tmp_2m,u_2m,v_2m,tmp_srf,pres_srf,_,_ = rf.read_era5_single_level(fprof_srf)

# lons_fore[lons_fore>180] = lons_fore[lons_fore>180]-360 # converting the 0-360 longitude to -180 - 180
# there are 8 moments of the data: 00:00,03:00,06:00,09:00,12:00,15:00,18:00,21:00
# tmp_l100 = tmp_l100[2] # currently, we only use data of one moment: 06:00
# q_l100 = q_l100[2]
# o3_l100 = o3_l100[2]
# cc_l100 = cc_l100[2,:,120:241,:]
# lats_fore = lats_fore[120:241]
cc_l100 = cc_l100[2]
tmp_2m = tmp_2m[2]
u_2m = u_2m[2]
v_2m = v_2m[2]
tmp_srf = tmp_srf[2]
pres_srf = pres_srf[2]

print('the numer of points that tmp_srf<280: %d' % np.sum(tmp_srf<280))
print('the numer of points that pres_srf<600: %d' % np.sum(pres_srf<600))
#----------------------------------------------------------------------------------------------
##### search the nearest rational pixel(without cloud) to the IRAS observation #####
x,y = np.meshgrid(lons_fore,lats_fore)
xy = np.concatenate((x.ravel().reshape(-1,1),y.ravel().reshape(-1,1)),axis=1)
tree = KDTree(xy)
# print(x.shape)
s1,s2,s3 = tmp_l100.shape  # n_level,n_latitude,n_longitude
bt2save = np.zeros((nvalid,5),dtype=float)
print(s2,s3)
X2save = np.zeros((nvalid,306))
m = 1
for k in range(nvalid):
    r,c = rs[k],cs[k]
    if np.abs(lats_obs[r,c]>30):
        print('The %d th point not in tropical area' % k)
        continue
    else:
        pass
    lon_lat = np.array([lons_obs[r,c],lats_obs[r,c]])
    print('The longitude and latitude of obs:',lon_lat)
    distance,index = tree.query(lon_lat,k=1) # find four nearest neighbors
    # if distance>1:
    #     continue

    # print(index)
    r_gfs,c_gfs = index//s3,index%s3
    # print(r_gfs,c_gfs)
    print('The longitude and latitude of the nearest neighbor:',x[r_gfs,c_gfs],y[r_gfs,c_gfs])
    # print(x.shape,y.shape)
    if y[r_gfs,c_gfs]>lon_lat[1]:
        r_gfs1 = r_gfs-1
        r_gfs2 = r_gfs
    else:
        r_gfs1 = r_gfs
        r_gfs2 = r_gfs+1

    if x[r_gfs,c_gfs]>lon_lat[0]:
        c_gfs1 = c_gfs-1
        c_gfs2 = c_gfs
    else:
        c_gfs1 = c_gfs
        c_gfs2 = c_gfs+1

    if c_gfs1==720 or c_gfs2==720 or r_gfs1==121 or r_gfs2==121:
        continue
    # print(x[r_gfs,c_gfs1],x[r_gfs,c_gfs2])
    #---------------- detect cloud of the four nearest points------------------------
    condition = np.max(cc_l100[:,r_gfs1,c_gfs1])>0.3 or np.max(cc_l100[:,r_gfs1,c_gfs2])>0.3 or\
        np.max(cc_l100[:,r_gfs2,c_gfs1])>0.3 or np.max(cc_l100[:,r_gfs2,c_gfs2])>0.3
    if condition:
        print('There is a neighbor that is cloudy')
        continue
    else:
        pass
    #---------------- Firstly, interpolate the four nearest profiles to 101 levels -------------------
    #### There is interpolation methods in RTTOV. If the data is good, this process is not necessary
    tmp1 = tmp_l100[:,r_gfs1,c_gfs1]
    q1 = q_l100[:,r_gfs1,c_gfs1]
    tmp1_int = interp_profile(level,tmp1,p101)
    q1_int = interp_profile(level,q1,p101)
    o31 = o3_l100[:,r_gfs1,c_gfs1]
    o31_int = interp_profile(level,o31,p101)
    
    tmp2 = tmp_l100[:,r_gfs1,c_gfs2]
    q2 = q_l100[:,r_gfs1,c_gfs2]
    tmp2_int = interp_profile(level,tmp2,p101)
    q2_int = interp_profile(level,q2,p101)
    o32 = o3_l100[:,r_gfs1,c_gfs2]
    o32_int = interp_profile(level,o32,p101)

    tmp3 = tmp_l100[:,r_gfs2,c_gfs1]
    q3 = q_l100[:,r_gfs2,c_gfs1]
    tmp3_int = interp_profile(level,tmp3,p101)
    q3_int = interp_profile(level,q3,p101)
    o33 = o3_l100[:,r_gfs2,c_gfs1]
    o33_int = interp_profile(level,o33,p101)

    tmp4 = tmp_l100[:,r_gfs2,c_gfs2]
    q4 = q_l100[:,r_gfs2,c_gfs2]
    tmp4_int = interp_profile(level,tmp4,p101)
    q4_int = interp_profile(level,q4,p101)
    o34 = o3_l100[:,r_gfs2,c_gfs2]
    o34_int = interp_profile(level,o34,p101)
    #----------------------- Secondly, bilnear interpolation ------------------------
    x1 = lons_fore[c_gfs1]
    x2 = lons_fore[c_gfs2]
    y1 = lats_fore[r_gfs1]
    y2 = lats_fore[r_gfs2]
    #----------- convert the unit of kg/kg to ppmv, then interpolate -----------
    q1_int = kg2ppmv(q1_int,18.0153)
    q2_int = kg2ppmv(q2_int,18.0153)
    q3_int = kg2ppmv(q3_int,18.0153)
    q4_int = kg2ppmv(q4_int,18.0153)
    o31_int = kg2ppmv(o31_int,47.9982)
    o32_int = kg2ppmv(o32_int,47.9982)
    o33_int = kg2ppmv(o33_int,47.9982)
    o34_int = kg2ppmv(o34_int,47.9982)
    #--------------------------------------------------------
    tmp_int = bilinear_interp(tmp1_int,tmp2_int,tmp3_int,tmp4_int,x1,x2,y1,y2,lon_lat[0],lon_lat[1])
    q_int = bilinear_interp(q1_int,q2_int,q3_int,q4_int,x1,x2,y1,y2,lon_lat[0],lon_lat[1])
    o3_int = bilinear_interp(o31_int,o32_int,o33_int,o34_int,x1,x2,y1,y2,lon_lat[0],lon_lat[1])
    pres_srf_int = bilinear_interp(pres_srf[r_gfs1,c_gfs1],pres_srf[r_gfs1,c_gfs2],pres_srf[r_gfs2,c_gfs1],
                                   pres_srf[r_gfs2,c_gfs2],x1,x2,y1,y2,lon_lat[0],lon_lat[1])
    tmp_srf_int = bilinear_interp(tmp_srf[r_gfs1,c_gfs1],tmp_srf[r_gfs1,c_gfs2],tmp_srf[r_gfs2,c_gfs1],
                                  tmp_srf[r_gfs2,c_gfs2],x1,x2,y1,y2,lon_lat[0],lon_lat[1])
    tmp_2m_int = bilinear_interp(tmp_2m[r_gfs1,c_gfs1],tmp_2m[r_gfs1,c_gfs2],tmp_2m[r_gfs2,c_gfs1],
                                 tmp_2m[r_gfs2,c_gfs2],x1,x2,y1,y2,lon_lat[0],lon_lat[1])
    # q_2m_int = bilinear_interp(q_2m[r_gfs1,c_gfs1],q_2m[r_gfs1,c_gfs2],q_2m[r_gfs2,c_gfs1],
    #                        q_2m[r_gfs2,c_gfs2],x1,x2,y1,y2,lon_lat[0],lon_lat[1])
    u_2m_int = bilinear_interp(u_2m[r_gfs1,c_gfs1],u_2m[r_gfs1,c_gfs2],u_2m[r_gfs2,c_gfs1],
                           u_2m[r_gfs2,c_gfs2],x1,x2,y1,y2,lon_lat[0],lon_lat[1])
    v_2m_int = bilinear_interp(v_2m[r_gfs1,c_gfs1],v_2m[r_gfs1,c_gfs2],v_2m[r_gfs2,c_gfs1],
                           v_2m[r_gfs2,c_gfs2],x1,x2,y1,y2,lon_lat[0],lon_lat[1])
    
    smaller = np.where(p101>pres_srf_int)[0]
    if len(smaller)<1:
        pass
    else:
        tmp_int[smaller] = tmp_srf_int
        o3_int[smaller] = o3_int[smaller[0]]
    #---------- convert the unit of ppmv back to kg/kg -----------
    q_int = ppmv2kg(q_int,18.0153)
    o3_int = ppmv2kg(o3_int,47.9982)
    p_2m_int = pres_srf_int
    q_2m_int = q_int[-1]
    #---------------------- Thirdly, save the interpolated profile to folders of RTTOV ----------------------
    if pres_srf_int<600 or tmp_int[-1]<280 or q_int[-1]<10**-6 or (o3_int<0.1E-8).any() or (q_int<0.1E-10).any():
        print('Some variable exceeds the hard limit')
        # print(pres_srf_int,tmp_int[-1]<280,q_int[-1])
        continue
    else:
        pass
    # child_dir_num = str(m//999)
    # s = str(m%999)
    # num = '0'*(3-len(s))+s
    # child_dir = parent_dir/str(child_dir_num)/num
    # child_dir = parent_dir / num
    # if not child_dir.exists():
    #     child_dir.mkdir()
    # atm_dir = child_dir/'atm'
    # if not atm_dir.exists():
    #     atm_dir.mkdir()
    # ground_dir = child_dir/'ground'
    # if not ground_dir.exists():
    #     ground_dir.mkdir()

    # np.savetxt(atm_dir/'p.txt',p101,fmt='%.6e')
    # np.savetxt(atm_dir/'t.txt',tmp_int,fmt='%.6e')
    # np.savetxt(atm_dir/'q.txt',q_int,fmt='%.6e')
    # np.savetxt(atm_dir/'o3.txt',o3_int,fmt='%.6e')
    # resave_2m(ground_dir/'s2m.txt',tmp_2m_int,q_2m_int,p_2m_int,u_2m_int,v_2m_int)

    if lsmask[r,c]==1:
        srftype = 0
    else:
        srftype = 1

    # resave_skin(ground_dir/'skin.txt',srftype,tmp_srf_int)
    # resave_angles(child_dir/'angles.txt',senzenith[r,c],azangle[r,c],solzenith[r,c],
    #               sunazangle[r,c],lon_lat[0],lon_lat[1],dem[r,c])
    # resave_date(child_dir/'datetime.txt',dodate)
    # resave_gas_unit(child_dir/'gas_units.txt',1) # 1=kg/kg, 2=ppmv
    
    # m += 1
    part1 = np.concatenate((tmp_int,q_int,o3_int))
    part2 = np.array([tmp_2m_int,tmp_srf_int,srftype])
    X2save[k,:] = np.append(part1,part2)
    # X2save[k,:] = np.vstack((tmp_int,q_int,o3_int,tmp_2m_int,tmp_srf_int,srftype))
    bt2save[k,:3] = bt[:,r,c]
    bt2save[k,3:] = lon_lat
    #------------------------------------------------------------------------------------------------------------

import calc_new_predictor as cnp

fmeanprof = Path(r'G:\DL_transmitance\O-B_validation\mean_prof.xlsx')
df = pd.read_excel(fmeanprof,sheet_name='mean_prof',header=None)
nlevel = 101
refer_prof = cnp.var_star(df.values,nlevel)  # reference profile calculated from the 1406 profiles
print(refer_prof)
nlayer = nlevel-1
xname = 'X4O-B-20080709.HDF'
rs = np.where(X2save[:,0]>1)[0]
X2save = X2save[rs,:]
bt2save = bt2save[rs,:]
print(bt2save.shape)
layer_prof = cnp.layer_profile(X2save[:,:303],nlevel)
new_preds = cnp.predictors(refer_prof,layer_prof,p101,nlayer)
# emiss_temp = np.full(bt2save.shape,fill_value=0.98,dtype=float)
with h5py.File(btsave_path/xname,'w') as f:
    x = f.create_dataset('X',data=X2save,compression='gzip')
    x.variables = 'temperature,specific humidity, ozone, temp_2m,temp_skin,srf_type'
    predictors = f.create_dataset('predictors',data=new_preds,compression='gzip')
#     emiss = f.create_dataset('emissivity',data=emiss_temp)
    bt = f.create_dataset('BT_obs',data=bt2save,compression='gzip')
    
############################### End! ####################################################################
plot_profile_var(X2save[:,101:202],p101,'Specific humidity(kg/kg)')
figname = 'combine_water_200807090006_interped101.png'
plt.savefig(figpath/figname)
plot_profile_var(X2save[:,:101],p101,'Temperature(K)')
figname = 'combine_tmp_200807090006_interped101.png'
plt.savefig(figpath/figname)
plot_profile_var(X2save[:,202:303],p101,'Ozone(kg/kg)')
figname = 'combine_o3_200807090006_interp101.png'
plt.savefig(figpath/figname)
