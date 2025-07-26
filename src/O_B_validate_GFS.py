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
from my_funcs import generate_chpr_txt,rh2q,interp_profile,bilinear_interp,calc_p_2m
import read_files as rf

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

# os.chdir('/home/sky/rttov121/rttov_test')
prof_path = Path('/mnt/hgfs/DL_transmitance/O-B_validation/200807_grb2_to_nc') # input('Please input the directory of profiles.')
obs_path = Path('/mnt/hgfs/DL_transmitance/O-B_validation/FY3A_OBS') # input('Please input the directory of IRAS observation.')
cld_path = Path('/mnt/hgfs/DL_transmitance/O-B_validation/FY3A_CLD') # input('Please input the directory of IRAS cloud mask.')
prof_save_path = Path('/home/sky/rttov121/rttov_test/profile-datasets/GRAPES/20190518') # input('Please input the directory where new forms of profile are to be saved.')
fp = Path('/mnt/hgfs/DL_transmitance/profiles/model_level_definition.xlsx')
btsave_path = Path('/mnt/hgfs/DL_transmitance/O-B_validation')
ch_index = [10,11,12] # input('Please input the IRAS channel you are about to simulate')
dodate = '20080709'

fprof = prof_path/'gfsanl_4_20080709_0000_006.nc'
fobs = obs_path/'FY3A_IRASX_GBAL_L1_20080709_0610_017KM_MS.HDF'
fcld = cld_path/'FY3A_IRASX_GBAL_L1_20080709_0610_017KM_MS_CL.HDF'
df = pd.read_excel(fp,sheet_name='101L')
p101 = np.array(df.iloc[:,0].values,dtype=float)
###################################################################################
ftime = fobs.name.split('_')[5]
fhour = int(ftime[:2])
parent_dir = prof_save_path/ftime
if not parent_dir.exists():
    parent_dir.mkdir()
else:
    pass

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

bt = bt_obs[ch_index,r1:r2,:].reshape(-1,1)
lons_obs = lons_obs[r1:r2,:].reshape(-1,1)
lats_obs = lats_obs[r1:r2,:].reshape(-1,1)
senzenith = senzenith[r1:r2,:].reshape(-1,1)
solzenith = solzenith[r1:r2,:].reshape(-1,1)
azangle = azangle[r1:r2,:].reshape(-1,1)
sunazangle = sunazangle[r1:r2,:].reshape(-1,1)
cld_mask_obs = cld_mask_obs[r1:r2,:].reshape(-1,1)
lsmask = lsmask[r1:r2,:].reshape(-1,1)
dem = dem[r1:r2,:].reshape(-1,1)

#------------------------------ Mask the valid and clear-sky pixels ---------------------------
mask1 = np.logical_or(cld_mask_obs==3, senzenith<=30)# 0=definitely cloudy, 1=possibly cloudy, 2=possibly clear, 3=definitely clear
mask2 = np.logical_or(np.logical(bt[0]>150,bt[1]>150),bt[2]>150)
mask = np.logical_or(mask1, mask2)
rs,cs = np.where(mask)
nvaliad = len(rs)
bt2save = np.concatenate((bt[:,rs,cs].T,lons_obs[rs,cs].reshape(-1,1),lats_obs[rs,cs].reshape(-1,1)))
fbt_name = 'BT_3a_20080709.xlsx'
df = pd.DataFrame(bt2save,columns=['ch11','ch12','ch13','longitude','latitude'])
df.to_excel(btsave_path/fbt_name)
################################ end #####################################################
################### read profile data and matching it with IRAS observation ##############
lv21,lv26,pres_srf,tmp_srf,tmp_2m,q_2m,u_2m,v_2m,tmp_l100,rh_l100,lats_fore,lons_fore = rf.read_nc_analysis(fprof)
#------------------ Interpolate tmp_l100 to the levels of rh_l100 for converting rh to specific humidity --------------------
s1,s2,s3 = tmp_l100.shape
tmp_reshape = tmp_l100.reshape(s1,-1)
nprof = tmp_reshape.shape[1]
n = len(lv21)
tmp_lvw = np.zeros((n,nprof),dtype=float)

for j in range(nprof):
    tmpj = tmp_reshape[:,j]
    tmp_interp = interp1d(np.log(lv26),tmpj,kind='linear',fill_value='extrapolate')
    tmp_lvw[:,j] = tmp_interp(np.log(lv21))

q_l100 = rh2q(rh_l100.reshape(n,-1),tmp_lvw,np.tile(lv21.reshape(-1,1),(1,s2*s3)))
q_l100 = q_l100.reshape(n,s2,s3)
#----------------------------------------------------------------------------------------------
##### search the nearest rational pixel(without cloud) to the IRAS observation #####
x,y = np.meshgrid(lons_fore,lats_fore) # The cloud ratio of analysis field is 0, so no need to detect cloud
xy = np.concatenate((x.ravel().reshape(-1,1),y.ravel().reshape(-1,1)),axis=1)

# no_cloud_index = np.where(cloud.ravel()<10)
# print(len(no_cloud_index[0]))
# xy = xy[no_cloud_index[0],:]
tree = KDTree(xy)

for k in range(nvaliad):
    r,c = rs[k],cs[k]
    lon_lat = np.array([lons_obs[r,c],lats_obs[r,c]])
    distance,index = tree.query(lon_lat,k=1) # find four nearest neighbors
    # if distance>1:
    #     continue

    r_gfs,c_gfs = index[0]//s3, index[0]%s3
    if lats_fore[r_gfs,c_gfs]>lon_lat[1]:
        r_gfs1 = r_gfs-1
        r_gfs2 = r_gfs
    else:
        r_gfs1 = r_gfs
        r_gfs2 = r_gfs-1

    if lons_fore[r_gfs,c_gfs]>lon_lat[0]:
        c_gfs1 = c_gfs-1
        c_gfs2 = c_gfs
    else:
        c_gfs1 = c_gfs
        c_gfs2 = c_gfs-1

    #---------------- Firstly, interpolate the four nearest profiles to 101 levels -------------------
    tmp1 = tmp_l100[:,r_gfs1,c_gfs1]
    q1 = q_l100[:,r_gfs1,c_gfs1]
    tmp1_int = interp_profile(lv21,tmp1,p101)
    q1_int = interp_profile(lv21,q1,p101)

    tmp2 = tmp_l100[:,r_gfs1,c_gfs2]
    q2 = q_l100[:,r_gfs1,c_gfs2]
    tmp2_int = interp_profile(lv21,tmp2,p101)
    q2_int = interp_profile(lv21,q2,p101)

    tmp3 = tmp_l100[:,r_gfs2,c_gfs1]
    q3 = q_l100[:,r_gfs2,c_gfs1]
    tmp3_int = interp_profile(lv21,tmp3,p101)
    q3_int = interp_profile(lv21,q3,p101)

    tmp4 = tmp_l100[:,r_gfs2,c_gfs2]
    q4 = q_l100[:,r_gfs2,c_gfs2]
    tmp4_int = interp_profile(lv21,tmp4,p101)
    q4_int = interp_profile(lv21,q4,p101)
    #----------------------- Secondly, bilnear interpolation ------------------------
    x1 = lons_fore[c_gfs1]
    x2 = lons_fore[c_gfs2]
    y1 = lats_fore[r_gfs1]
    y2 = lats_fore[r_gfs2]
    tmp_int = bilinear_interp(tmp1,tmp2,tmp3,tmp4,x1,x2,y1,y2,lon_lat[0],lon_lat[1])
    q_int = bilinear_interp(q1,q2,q3,q4,x1,x2,y1,y2,lon_lat[0],lon_lat[1])
    pres_srf_int = bilinear_interp(pres_srf[r_gfs1,c_gfs1],pres_srf[r_gfs1,c_gfs2],pres_srf[r_gfs2,c_gfs1],
                                   pres_srf[r_gfs2,c_gfs2],x1,x2,y1,y2,lon_lat[0],lon_lat[1])
    tmp_srf_int = bilinear_interp(tmp_srf[r_gfs1,c_gfs1],tmp_srf[r_gfs1,c_gfs2],tmp_srf[r_gfs2,c_gfs1],
                                  tmp_srf[r_gfs2,c_gfs2],x1,x2,y1,y2,lon_lat[0],lon_lat[1])
    tmp_2m_int = bilinear_interp(tmp_2m[r_gfs1,c_gfs1],tmp_2m[r_gfs1,c_gfs2],tmp_2m[r_gfs2,c_gfs1],
                                 tmp_2m[r_gfs2,c_gfs2],x1,x2,y1,y2,lon_lat[0],lon_lat[1])
    q_2m_int = bilinear_interp(q_2m[r_gfs1,c_gfs1],q_2m[r_gfs1,c_gfs2],q_2m[r_gfs2,c_gfs1],
                           q_2m[r_gfs2,c_gfs2],x1,x2,y1,y2,lon_lat[0],lon_lat[1])
    u_2m_int = bilinear_interp(u_2m[r_gfs1,c_gfs1],u_2m[r_gfs1,c_gfs2],u_2m[r_gfs2,c_gfs1],
                           u_2m[r_gfs2,c_gfs2],x1,x2,y1,y2,lon_lat[0],lon_lat[1])
    v_2m_int = bilinear_interp(v_2m[r_gfs1,c_gfs1],v_2m[r_gfs1,c_gfs2],v_2m[r_gfs2,c_gfs1],
                           v_2m[r_gfs2,c_gfs2],x1,x2,y1,y2,lon_lat[0],lon_lat[1])
    p_2m_int = calc_p_2m(pres_srf_int,tmp_srf_int,q_2m_int)  # it would be better if there is q_srf
    #---------------------- Thirdly, save the interpolated profile to folders of RTTOV ----------------------
    child_dir_num = str(k//999+1)
    s = str(k%999+1)
    num = '0'*(3-len(s))+s
    child_dir = parent_dir/str(child_dir_num)/num
    if not child_dir.exists():
        child_dir.mkdir()
    atm_dir = child_dir/'atm'
    if not atm_dir.exists():
        atm_dir.mkdir()
    ground_dir = child_dir/'ground'
    if not ground_dir.exists():
        ground_dir.mkdir()

    np.savetxt(atm_dir/'p.txt',p101,fmt='%.6e')
    np.savetxt(atm_dir/'t.txt',tmp_int,fmt='%.6e')
    np.savetxt(atm_dir/'q.txt',q_int,fmt='%.6e')
    resave_2m(ground_dir/'s2m.txt',tmp_2m_int,q_2m_int,p_2m_int,u_2m_int,v_2m_int)

    if lsmask[r,c]==1:
        srftype = 0
    else:
        srftype = 1

    resave_skin(ground_dir/'skin.txt',srftype,tmp_srf_int)
    resave_angles(child_dir/'angles.txt',senzenith[r,c],azangle[r,c],solzenith[r,c],
                  sunazangle[r,c],lon_lat[0],lon_lat[1],dem[r,c])
    resave_date(child_dir/'datetime.txt',dodate)
    resave_gas_unit(child_dir/'gas_units.txt',1) # 1=kg/kg, 2=ppmv
    #-----------------------------------------------------------------------------------------------------------------------------
############################### End! ####################################################################
################# Editting shell file to fit the needs ##############################
# fshell = r'G:\RTTOV121\rttov_test\my_test.sh' #需修改
# f = open(fshell,'a')
# f.write('\n')
#
# status = subprocess.call(rttov_test_shell,shell=True)

