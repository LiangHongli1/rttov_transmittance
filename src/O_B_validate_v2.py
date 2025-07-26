# -*- coding: utf-8 -*-
"""
@ Time: 2019/10/28
@ author: LiangHongli
@ Mail: Helen_Liang1@outlook.com

Simulating TOA brightness temperature through RTTOV with GRAPES forecasting field profiles.
Firstly, interp the GRAPES forecasting field to satellite observation.
Then, matching profile with IRAS observation. Only the dots where time_diff<1.5h are simulated.
And the max cloud should be less than 3%.
At last: simulating BT - IRAS observation BT
"""

import numpy as np
############# To make PyInstaller work properly, these modules need to be imported for now ###############
# import numpy.random.common
# import numpy.random.bounded_integers
# import numpy.random.entropy
###########################################################################################################
import h5py
from pathlib import Path
import time
from scipy.spatial import KDTree
from scipy.interpolate import interp1d,LinearNDInterpolator,RegularGridInterpolator
import pandas as pd

from my_funcs import ppmv2kg,kg2ppmv,generate_chpr_txt,calc_p_2m
import read_files as rf
from LUT import consts

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

################################## interpolation end! #############################################
#################### Select the profiles and save them to corresponding directory ######################

def match_profile():
    prof_path = Path('/mnt/hgfs/DL_transmitance/O-B_validation/200807_grb2_to_nc') # input('Please input the directory of profiles.')
    obs_path = Path('/mnt/hgfs/DL_transmitance/O-B_validation/FY3A_OBS') # input('Please input the directory of IRAS observation.')
    cld_path = Path('/mnt/hgfs/DL_transmitance/O-B_validation/FY3A_CLD') # input('Please input the directory of IRAS cloud mask.')
    prof_save_path = '/home/sky/rttov121/rttov_test/profile-datasets/GRAPES/20080709' # input('Please input the directory where new forms of profile are to be saved.')
    fp = Path(r'/mnt/hgfs/DL_transmitance/profiles/model_level_definition.xlsx')
    writer = pd.ExcelWriter('/mnt/hgfs/DL_transmitance/O-B_validation/BT_obs_fy3a_20080709.xlsx')

    dodate = '20080709' # input('Please input the date you want to simulate.')
    channel = 11 # input('Please input the IRAS channel you are about to simulate')
    nlv = 101
    # prof_path = prof_path/dodate
    obs_path = obs_path/dodate
    cld_path = cld_path/dodate
    prof_files = [x for x in sorted(prof_path.glob('*.nc'))]
    obs_files = [x for x in sorted(obs_path.glob('*.HDF'))]
    cld_files = [x for x in sorted(cld_path.glob('*.HDF'))]

    df = pd.read_excel(fp,sheet_name='101L')
    p101 = np.array(df['ph[hPa]'].values,dtype=np.float) # standard 91-level profile
    # fractor = np.pi/180
    obsbt = np.zeros((1, 5), dtype=np.float)
    for f_num,fobs in enumerate(obs_files):
        profile_dir = fobs.name.split('_')[5]
        parent_dir = Path(prof_save_path)/profile_dir # parent directory to save the profiles
        if not parent_dir.exists():
            parent_dir.mkdir()

        ftime = time.strptime(fobs.name.split('_')[4]+':'+fobs.name.split('_')[5],'%Y%m%d:%H%M')
        ftimestamp = time.mktime(ftime)

    ############# read IRAS data and extract the ascending part #############################
        try:
            bt_obs, lons_obs, lats_obs, senzenith,solzenith,azangle,sunazangle,lsmask,dem = rf.read_IRAS_3a(str(fobs))
            cld_mask = rf.read_cld(str(cld_files[f_num]))
            # print(cld_mask)
        except:
            continue

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

        bt = bt_obs[channel-1:channel+2,r1:r2,:]
        lons_obs = lons_obs[r1:r2,:]
        lats_obs = lats_obs[r1:r2,:]
        senzenith = senzenith[r1:r2,:]
        solzenith = solzenith[r1:r2,:]
        azangle = azangle[r1:r2,:]
        sunazangle = sunazangle[r1:r2,:]
        cld_mask = cld_mask[r1:r2,:]
        lsmask = lsmask[r1:r2,:]
        dem = dem[r1:r2,:]

        rindex = list(set(np.where(bt[0,:,:]>100)[0]))
        bt = bt[:,rindex,:]
        lons_obs = lons_obs[rindex, :]
        lats_obs = lats_obs[rindex, :]
        senzenith = senzenith[rindex, :]
        solzenith = solzenith[rindex, :]
        azangle = azangle[rindex, :]
        sunazangle = sunazangle[rindex, :]
        cld_mask = cld_mask[rindex, :]
        lsmask = lsmask[rindex, :]
        dem = dem[rindex, :]
    ################################ end #####################################################
    ################### read profile data and matching it with IRAS observation ##############
        # print(prof_files)
        for fprof in prof_files:
            ptime = fprof.name.split('_')[2] #[:8]+':'
            tmp_var = int(fprof.name.split('_')[3][:2])+int(fprof.name.split('_')[4][:3])#[-2:]
            # if int(fprof.name.split('_')[2][8:10])==0:
            #     phour = tmp_var+'00'
            # else:
            #     phour = str(int(tmp_var)+12)+'00'
            # print(fobs.name, fprof.name)
            if len(str(tmp_var))<2:
                phour = '0'+str(tmp_var)+'00'
            else:
                phour = str(tmp_var)+'00'
            # print(ptime+phour)
            ptimestamp = time.mktime(time.strptime(ptime+phour,'%Y%m%d%H%M'))

            if abs(ftimestamp-ptimestamp)>10800.0:
                print('The difference between IRAS observation time and ECMWF forecasting time is larger than 3 hours. Jump.')
                continue
            else:
                print('fobs file: %s, fprofile file: %s' % (fobs.name,fprof.name))
                fgrapes = fprof
                break

        # lats_fore,lons_fore,cloud = rf.read_nc_llc(str(fgrapes))
        lv21,lv26,pres_surf,tmp_surf,tmp_2m,q_2m,u_2m,v_2m,tmp_l100,q_l100,lats_fore,lons_fore = rf.read_nc_analysis(str(fgrapes))
        # print(q_l100[:,0,0],q_l100[:,10,10])
        ### Converting the q's unit from kg/kg to ppmv for the accuracy of interpolating ######
        q_2m = kg2ppmv(q_2m,18.0153)
        q_l100 = kg2ppmv(q_l100,18.0153)
        # print(cloud)
        ############ Make interpolators and do interp ##############################################
        # lats = lats_fore*fractor
        lats = lats_fore[::-1]
        # lons = lons_fore*fractor
        # print(lats)
        # interpolator_cld = RectSphereBivariateSpline(lats,lons,np.flipud(cloud))
        # interpolator_p_srf = RectSphereBivariateSpline(lats, lons, np.flipud(pres_surf)) # surface pressure interpolator
        # interpolator_tmp_srf = RectSphereBivariateSpline(lats, lons, np.flipud(tmp_surf))
        # interpolator_tmp_2m = RectSphereBivariateSpline(lats, lons, np.flipud(tmp_2m))
        # interpolator_q_2m = RectSphereBivariateSpline(lats, lons, np.flipud(q_2m))
        # interpolator_u_2m = RectSphereBivariateSpline(lats, lons, np.flipud(u_2m))
        # interpolator_v_2m = RectSphereBivariateSpline(lats, lons, np.flipud(v_2m))
        # interpolator_cld = RegularGridInterpolator((lats, lons_fore), np.flipud(cloud),bounds_error=False,fill_value=None)
        # interpolator_p_srf = RegularGridInterpolator((lats, lons_fore), np.flipud(pres_surf),bounds_error=False,fill_value=None)  # surface pressure interpolator
        # interpolator_tmp_srf = RegularGridInterpolator((lats, lons_fore), np.flipud(tmp_surf),bounds_error=False,fill_value=None)
        # interpolator_tmp_2m = RegularGridInterpolator((lats, lons_fore), np.flipud(tmp_2m),bounds_error=False,fill_value=None)
        # interpolator_q_2m = RegularGridInterpolator((lats, lons_fore), np.flipud(q_2m),bounds_error=False,fill_value=None)
        # interpolator_u_2m = RegularGridInterpolator((lats, lons_fore), np.flipud(u_2m),bounds_error=False,fill_value=None)
        # interpolator_v_2m = RegularGridInterpolator((lats, lons_fore), np.flipud(v_2m),bounds_error=False,fill_value=None)
        # interpolator_tmp_l100 = RegularGridInterpolator((lv26,lats, lons_fore),np.flip(tmp_l100,axis=1),method='linear',bounds_error=False,fill_value=None)
        # interpolator_q_l100 = RegularGridInterpolator((lv21,lats, lons_fore),np.flip(q_l100,axis=1),method='linear',bounds_error=False,fill_value=None)
        interpolator_p_srf = LinearNDInterpolator((lats, lons_fore), np.flipud(pres_surf),bounds_error=False,fill_value=None)  # surface pressure interpolator
        interpolator_tmp_srf = LinearNDInterpolator((lats, lons_fore), np.flipud(tmp_surf),bounds_error=False,fill_value=None)
        interpolator_tmp_2m = LinearNDInterpolator((lats, lons_fore), np.flipud(tmp_2m),bounds_error=False,fill_value=None)
        interpolator_q_2m = LinearNDInterpolator((lats, lons_fore), np.flipud(q_2m),bounds_error=False,fill_value=None)
        interpolator_u_2m = LinearNDInterpolator((lats, lons_fore), np.flipud(u_2m),bounds_error=False,fill_value=None)
        interpolator_v_2m = LinearNDInterpolator((lats, lons_fore), np.flipud(v_2m),bounds_error=False,fill_value=None)
        interpolator_tmp_l100 = LinearNDInterpolator((lv26,lats, lons_fore),np.flip(tmp_l100,axis=1),method='linear',bounds_error=False,fill_value=None)
        interpolator_q_l100 = LinearNDInterpolator((lv21,lats, lons_fore),np.flip(q_l100,axis=1),method='linear',bounds_error=False,fill_value=None)
        # lats_new = (lats_obs*fractor).ravel()
        # lons_new = (lons_obs*fractor).ravel()
        xy = np.concatenate((lats_obs.reshape(-1,1),lons_obs.reshape(-1,1)),axis=1)
        # cld_interp = interpolator_cld(xy).reshape(lats_obs.shape)
        p_srf_interp = interpolator_p_srf(xy).reshape(lats_obs.shape)
        tmp_srf_interp = interpolator_tmp_srf(xy).reshape(lats_obs.shape)
        tmp_2m_interp = interpolator_tmp_2m(xy).reshape(lats_obs.shape)
        q_2m_interp = interpolator_q_2m(xy).reshape(lats_obs.shape)
        u_2m_interp = interpolator_u_2m(xy).reshape(lats_obs.shape)
        v_2m_interp = interpolator_v_2m(xy).reshape(lats_obs.shape)
        # print(cld_interp.shape)

        x3 = np.tile(p101,(lats_obs.shape[0],lats_obs.shape[1],1))
        y3 = np.tile(lats_obs,(1,nlv))
        z3 = np.tile(lons_obs,(1,nlv))
        xyz = np.concatenate((x3.reshape(-1,1), y3.reshape(-1,1), z3.reshape(-1,1)),axis=1)
        tmp_l100_interp = interpolator_tmp_l100(xyz).reshape((len(p101),lats_obs.shape[0],lats_obs.shape[1]))
        q_l100_interp = interpolator_q_l100(xyz).reshape((len(p101),lats_obs.shape[0],lats_obs.shape[1]))

        print('The %d th interpolating done!' % (f_num+1))
        ###################### End #####################################
        ################ Kicking off the bad data and leave the good ones. ########################
        ls_mask = np.logical_or(lsmask == 1, lsmask == 3)
        cloud_mask = cld_mask == 3  # clear-sky-only
        # bt_rational_mask = bt[0,:,:] != np.nan
        ss_zenith_mask = senzenith < 30
        # cld_interp_mask = cld_interp<30
        condition = np.logical_and(np.logical_and(ls_mask, cloud_mask),ss_zenith_mask)
        # condition = np.logical_and(condition,cld_interp_mask)

        lats_obs = lats_obs[condition]
        lons_obs = lons_obs[condition]
        bt_rational = bt[:,condition]
        senzenith = senzenith[condition]
        solzenith = solzenith[condition]
        azangle = azangle[condition]
        sunazangle = sunazangle[condition]
        dem = dem[condition]
        lsmask = lsmask[condition]
        print(lsmask.shape)

        p_srf_interp = p_srf_interp[condition]
        tmp_srf_interp = tmp_srf_interp[condition]
        tmp_2m_interp = tmp_2m_interp[condition]
        q_2m_interp = q_2m_interp[condition]
        u_2m_interp = u_2m_interp[condition]
        v_2m_interp = v_2m_interp[condition]
        p_2m_interp = calc_p_2m(p_srf_interp,tmp_srf_interp,q_2m_interp)
        tmp_l100_interp = tmp_l100_interp[:,condition].reshape(len(p101),-1)
        q_l100_interp = q_l100_interp[:,condition].reshape(len(p101),-1)
        ###### Converting back the q's unit ######################
        q_2m_interp = ppmv2kg(q_2m_interp,18.0153)
        q_l100_interp = ppmv2kg(q_l100_interp,18.0153)
        ############### End! #############################################
        ############## Save the interped profiles to RTTOV's directory ##################
        jj = 0

        for k in range(len(lats_obs)):
            ########### tropical areas and the profiles whose teperature at pressure 101-level is greater than 280K ############
            if p_2m_interp[k] < 1100:
                larger = np.where(p101 >= p_2m_interp[k])[0]
                tmp_l100_interp[:, k][larger] = tmp_2m_interp[k]
            if abs(lats_obs[k])>30 or tmp_l100_interp[:,k][-1]<280:
                continue

            condition1 = np.max(tmp_l100_interp[:,k])>consts['tmax'] or np.max(q_l100_interp[:,k])>consts['qmax_kg'] or \
                np.min(tmp_l100_interp[:,k])<consts['tmin'] or np.min(q_l100_interp[:,k])<consts['qmin'] or p_2m_interp[k]>consts['pmax']# if any variable is out of the box, kick it out
            condition2 = tmp_srf_interp[k]>consts['tmax'] or tmp_2m_interp[k]>consts['tmax'] or tmp_srf_interp[k]<consts['tmin'] or \
                tmp_2m_interp[k]<consts['tmin'] or q_2m_interp[k]>consts['qmax_kg'] or q_2m_interp[k]<consts['qmin'] or dem[k]>consts['elemax']
            condition = np.logical_or(condition1,condition2)
            if condition:
                continue
            else:
                parent1_dir = parent_dir/(str(jj//999+1)+'grape')
                if not parent1_dir.exists():
                    parent1_dir.mkdir()
                if len(str(jj%999+1))==1:
                    dir_num = '0'*2+str(jj%999+1)
                elif len(str(jj%999+1))==2:
                    dir_num = '0'+str(jj%999+1)
                else:
                    dir_num = str(jj%999+1)

                child_dir = parent1_dir/dir_num
                if not child_dir.exists():
                    child_dir.mkdir()

                atm_dir = child_dir/'atm'
                if not atm_dir.exists():
                    atm_dir.mkdir()
                ground_dir = child_dir/'ground'
                if not ground_dir.exists():
                    ground_dir.mkdir()

                ############# if there are pressures in 101L that are smaller than 2m pressure, then set them constant ###############
                np.savetxt(atm_dir/'p.txt',p101,fmt='%.6e')
                np.savetxt(atm_dir/'t.txt',tmp_l100_interp[:,k],fmt='%.6e')
                np.savetxt(atm_dir/'q.txt',q_l100_interp[:,k],fmt='%.6e')
                resave_2m(ground_dir/'s2m.txt',tmp_2m_interp[k],q_2m_interp[k],p_2m_interp[k],u_2m_interp[k],v_2m_interp[k]) # 2m处气压未算

                if lsmask[k]==1:
                    srftype = 0
                else:
                    srftype = 1

                tmp = np.array([lons_obs[k], lats_obs[k], bt_rational[0,k],bt_rational[1,k],bt_rational[2,k]])[np.newaxis,:]
                obsbt = np.append(obsbt, tmp, axis=0)
                resave_skin(ground_dir/'skin.txt',srftype,tmp_srf_interp[k])
                resave_angles(child_dir/'angles.txt',senzenith[k],azangle[k],solzenith[k],sunazangle[k],lats_obs[k],lons_obs[k],dem[k])
                resave_date(child_dir/'datetime.txt',dodate)
                resave_gas_unit(child_dir/'gas_units.txt',1) # 1=kg/kg, 2=ppmv
                jj += 1
                # print('The %d th profile is done!' % (jj + 1))

    obt = pd.DataFrame(obsbt[1:,:])
    # print(obt)
    obt.to_excel(writer)
    writer.save()
    writer.close()
    return

match_profile()
################################################## Profile selection done! ###########################################################################

################# Editting shell file and input *.txt files to make preparation for runing RTTOV ##############################
# import os
# import subprocess
#
# src = Path('/home/sky/rttov121/rttov_test/profile-datasets/GRAPES/20190517')
# src_dirs = [x for x in src.glob('**/**') if x.is_dir()]
# target = Path('/home/sky/rttov121/rttov_test/tests.0/iras')
# coef_file = 'rtcoef_rttov12/rttov9pred54L/rtcoef_fy3_1_iras.dat'
#
# def save_fcoef(target,fname):
#     '''
#     Save coef file to input profile set
#     :paam target: target path of the coef file
#     :param fname: relative directory of the coefficient file
#     :return: None
#     '''
#     f = open(target,'w')
#     f.write('&coef_nml\n')
#     s = '  '+'defn%f_coef'+' '+'='+' '+coef_file+'\n'
#     f.write(s)
#     f.write('/')
#     f.close()
#     return
#
#
# fshell = r'/home/sky/rttov121/rttov_test/ob_iras.sh'
# f = open(fshell,'r+')
# for k,direct in enumerate(src_dirs):
#     suffix = str(direct).split('/')[-1]
#     tar_dir = target/'suffix'/'in'
#     if tar_dir.exists():
#         pass
#     else:
#         tar_dir.mkdir()
#         save_fcoef(tar_dir/'coef.txt',coef_file)
#         npro = len([x for x in direct.glob('**') if x.is_dir()])
#         generate_chpr_txt(tar_dir, channel, npro)
#
#         status = subprocess.call("ln -s {:s} {:s}".format(str(direct),str(tar_dir/'profiles')),shell=True)
#         s = f.read()
#         if k%9!=0:
#             f.write('iras/'+suffix)
#         else:
#             f.write('\n  TEST_LIST = iras/'+suffix)
#
# f.write('\nEOF')
# f.close()
#
# os.chdir('/home/sky/rttov121/rttov_test')
#
# status = subprocess.call(fshell,shell=True)

############################### End! ####################################################################
########################### Save the BT computed by RTTOV ##############################
def save_bt():
    iras_dir = Path('/home/sky/rttov121/rttov_test/my_test.1.gfortran/iras')
    dir_flag = '*grape0517'
    nch = 3
    for k, kdir in enumerate(sorted(iras_dir.glob(dir_flag))):
        out_dir = kdir / 'out' / 'direct'
        if k == 0:
            bt_iras = rf.bt_clear(out_dir / 'radiance.txt', nch)
            emiss = rf.emisi_suf(out_dir /'emissivity_out.txt',nch)
            srf_trans = rf.trans_srf(out_dir/'transmission.txt',nch)
        else:
            bt_iras = np.append(bt_iras, rf.bt_clear(out_dir / 'radiance.txt', nch), axis=0)
            emiss = np.append(emiss,rf.emisi_suf(out_dir / 'emissivity_out.txt', nch),axis=0)
            srf_trans = np.append(srf_trans,rf.trans_srf(out_dir / 'transmission.txt', nch),axis=0)

    return bt_iras,emiss,srf_trans

def save_profile():
    propath = Path('/home/sky/rttov121/rttov_test/profile-datasets/GRAPES/20190517')
    prodirs = [x for x in sorted(propath.glob('**/*grape0517'))]
    freference = Path('/home/sky/rttov121/rtcoef_rttov12/rttov7pred54L/rtcoef_fy3_3_iras.dat')
    fp = Path(r'/mnt/hgfs/DL_transmitance/profiles/model_level_definition.xlsx')
    df = pd.read_excel(fp, sheet_name='101L')
    p = np.array(df['ph[hPa]'].values, dtype=np.float)
    nlv = 101
    k = 0
    for parent in prodirs:
        childs = sorted([x for x in parent.glob('*') if x.is_dir()])
        # print(parent,childs)
        for child in childs:
            # fcc = child/'atm'/'cc.txt'
            # subprocess.call(['rm', '-rf', str(fcc)])
            if k==0:
                profiles = rf.read_profile(child)
            else:
                profiles = np.vstack((profiles,rf.read_profile(child)))
            k += 1

    n = profiles.shape[0]
    o3 = np.zeros((1, 4), dtype=np.float)
    with open(freference,'r') as f:
        lines = f.readlines()
        o3_pro = lines[239:293]
        for o in o3_pro:
            o3 = np.vstack((o3,np.array([float(x) for x in o.split()])))

    interpolator = interp1d(o3[:,0],o3[:,2],fill_value='extrapolate')  # pressure and ozone mixing ratio(ppmv)
    o3_interp = np.tile(interpolator(p),[n,1])
    profiles = np.concatenate((profiles[:,nlv:nlv*3],ppmv2kg(o3_interp,47.9982),profiles[:,nlv*3:]),axis=1)

    bt_iras, emiss, srf_trans = save_bt()
    fsave = Path('/mnt/hgfs/DL_transmitance/O-B_validation/xbt_rttov4obs.HDF')
    with h5py.File(fsave,'w') as f:
        x = f.create_dataset('X',data=profiles,compression='gzip')
        x.attrs['name'] = 'GFS clear-sky profiles, no ozone'
        f.create_dataset('BT_rttov',data=bt_iras,compression='gzip')
        f.create_dataset('emissivity',data=emiss,compression='gzip')
        f.create_dataset('trans_srf',data=srf_trans,compression='gzip')
    return
# save_profile()





