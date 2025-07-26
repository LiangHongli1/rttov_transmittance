from numpy import NaN
from numpy import *
from netCDF4 import Dataset
import numpy as np
import os
import h5py
import datetime
import math
import sys
import glob
import time
import pygrib

times = sys.argv[1]
instrument = sys.argv[2]
dates = datetime.datetime.strptime(str(times),'%Y%m%d')
year = dates.strftime('%Y')
month = dates.strftime('%m')
day = dates.strftime('%d')

def generate_nwp_times(nwptimer):
    dhour = datetime.timedelta(hours=12)
    date = nwptimer.strftime('%Y%m%d')
    if (nwptimer.hour>9):
        fcsttime = '12'
        tnwptime = nwptimer-dhour
    else:
        fcsttime = '00'
        tnwptime = nwptimer
    hour = tnwptime.strftime('%H')
    nwptime=date+fcsttime+'0'+hour
    return nwptime

def make_predictor(nwpfile,ncfile):
    P=np.array(['1000','975','950','925','900','850','800','750','700','650','600','550','500','450','400','350','300','250','200','150','100','70','50','30','20','10','7','5','4','3','2','1D5','1','0D5','0D2','0D1'])
    Pf=np.array([1000,975,950,925,900,850,800,750,700,650,600,550,500,450,400,350,300,250,200,150,100,70,50,30,20,10,7,5,4,3,2,1.5,1,0.5,0.2,0.1])
    R = 287.5
    nlevel = P.shape[0]
    if not os.path.exists(ncfile+'_H.nc'):
        os.system('./T639/wgrib2 -match ":HGT:" -match "mb:" %s -no_header -netcdf %s' % (nwpfile,ncfile+'_H.nc'))
    if not os.path.exists(ncfile+'_q.nc'):
        os.system('./T639/wgrib2 -match ":SPFH:" -match "mb:" %s -no_header -netcdf %s' % (nwpfile,ncfile+'_q.nc'))
    if not os.path.exists(ncfile+'_T.nc'):
        os.system('./T639/wgrib2 -match ":TMP:" -match "mb:" %s -no_header -netcdf %s' % (nwpfile,ncfile+'_T.nc'))
    if not os.path.exists(ncfile+'_Ts.nc'):
        os.system('./T639/wgrib2 -match ":TMP:" -match "surface:" %s -no_header -netcdf %s' % (nwpfile,ncfile+'_Ts.nc'))

    ncH = Dataset(ncfile+'_H.nc',mode='r')
    H1000 = ncH.variables['HGT_1000mb'][:]
    H300 = ncH.variables['HGT_300mb'][:]
    H200 = ncH.variables['HGT_200mb'][:]
    H50 = ncH.variables['HGT_50mb'][:]
    ncH.close()
    nlat = H50.shape[1]
    nlon = H50.shape[2]
    X1=np.array([[0.]*nlon]*nlat)
    X2=np.array([[0.]*nlon]*nlat)
    X3=np.array([[0.]*nlon]*nlat)
    X4=np.array([[0.]*nlon]*nlat)
    X1 = H300-H1000
    X2 = H50-H200

    ncq =  Dataset(ncfile+'_q.nc',mode='r')
    q = np.array([[[0.]*nlevel]*nlon]*nlat)
    for i in range(nlevel): 
        varname='SPFH_'+P[i]+'mb'
        q[:,:,i] = ncq.variables[varname][:]
    ncq.close()

    ncT =  Dataset(ncfile+'_T.nc',mode='r')
    T = np.array([[[0.]*nlevel]*nlon]*nlat)
    for i in range(nlevel): 
        varnamet='TMP_'+P[i]+'mb'
        T[:,:,i] = ncT.variables[varnamet][:]
    ncT.close()

    ncTs = Dataset(ncfile+'_Ts.nc',mode='r')
    Ts = ncTs.variables['TMP_surface'][:]
    ncTs.close()
    X3 = Ts
   
    for i in range(nlevel-1):
        X4=X4+(q[:,:,i]+q[:,:,i+1])*0.5*(Pf[i]-Pf[i+1])/9.8

    return X1,X2,X3,X4,nlat,nlon

def interpolation(fgeo1,fgeo2,fgeo3,fgeo4,indexl0,indexp0,indexl1,indexp1,Xor):
    #print '++++',fgeo1.shape
    #print '****',indexl0.shape
    #print '===',Xor.shape
    X = Xor[indexl0,indexp0]*fgeo1+Xor[indexl0,indexp1]*fgeo2+Xor[indexl1,indexp1]*fgeo3+Xor[indexl1,indexp0]*fgeo4
    return X

dtime = datetime.timedelta(hours=3)
hday = datetime.timedelta(hours=12)
starttime = datetime.datetime.strptime('200001011200','%Y%m%d%H%M')
hftime = dtime.seconds
#print 'hftime=',hftime

satpath = '/gpfs1/work/data/SAT/NSMC/FY3D/'+instrument+'/L1/'+year+'/'+month
nwppath = '/gpfs1/work/data/NWP/T639/'+year+'/'+month
outpath = './outdata/'+instrument+'/'+year+'/'+month+'/'
T639path = './T639/'+year+'/'+month+'/'+day+'/'
if not os.path.exists(outpath):
    os.makedirs(outpath)
if not os.path.exists(T639path):
    os.makedirs(T639path)

satfilelist = glob.glob(satpath+'/FY3D_'+instrument+'X_GBAL_L1_'+times+'_*_*KM_MS.HDF')
satfilelist.sort()

for satfile in satfilelist:
    fsat = h5py.File(satfile,'r')
    lat = fsat['/Geolocation/Latitude'].value
    lon = fsat['/Geolocation/Longitude'].value
    lon[lon<0] = lon[lon<0]+360.
    idlon = np.argwhere(np.isnan(lon))
    idlat = np.argwhere(np.isnan(lat))
    lat[idlat] = 100.
    lon[idlon] = 380.
    daycnt = fsat['/Geolocation/Scnlin_daycnt'].value
    mscnt = fsat['/Geolocation/Scnlin_mscnt'].value
    fsat.close()
    #print 'lat.shape=',lat.shape
    nscanl = daycnt.shape[0]
    nscanp = lat.shape[1]
    print 'nscanl=',nscanl,'nscanp=',nscanp

    ddscnt0 = datetime.timedelta(days=int(daycnt[0]))
    dmscnt0 = datetime.timedelta(hours=float(mscnt[0]/1000./60./60.))
    sattime = starttime+ddscnt0+dmscnt0
    #print 'ddscnt0=',ddscnt0,'dmscnt0=',dmscnt0,'hday=',hday
    #print 'sattime=',sattime
    timepos = satfile.index('KM_MS')
    hhmm = satfile[timepos-8:timepos-4]
    #print 'hhmm',hhmm
    yyyymmdd0000 = times+'0000'
    yyyymmddhhmm = datetime.datetime.strftime(sattime,'%Y%m%d%H%M')
    print 'start time=',yyyymmddhhmm
    nwptimepre = datetime.datetime.strptime(yyyymmdd0000,'%Y%m%d%H%M')
    sathour = sattime.hour
    nh = sathour/3
    outputfile = outpath+'FY3D_'+instrument+'X_GBAL_L1_'+times+hhmm+'_Air_Mass.HDF' 
    if os.path.exists(outputfile):
        continue

    nwptime1 = nwptimepre+dtime*nh
    nwptime2 = nwptime1+dtime
    nwptime3 = nwptime2+dtime
    
    #print '&&&&',nwptime1,nwptime2,nwptime3
    strtime1 = generate_nwp_times(nwptime1)
    strtime2 = generate_nwp_times(nwptime2)
    strtime3 = generate_nwp_times(nwptime3)

    #print strtime1,strtime2,strtime3

    nwpfile1 = nwppath+'/gmf.639.'+strtime1+'.grb2'
    nwpfile2 = nwppath+'/gmf.639.'+strtime2+'.grb2'
    nwpfile3 = nwppath+'/gmf.639.'+strtime3+'.grb2'
    ncfile1 = T639path+'/'+strtime1
    ncfile2 = T639path+'/'+strtime2
    ncfile3 = T639path+'/'+strtime3
    X1a,X2a,X3a,X4a,nlat,nlon=make_predictor(nwpfile1,ncfile1)
    X1b,X2b,X3b,X4b,nlat,nlon=make_predictor(nwpfile2,ncfile2)
    X1c,X2c,X3c,X4c,nlat,nlon=make_predictor(nwpfile3,ncfile3)
   
    fractime1 = np.array([[0.]*nscanp]*nscanl)
    fractime2 = np.array([[0.]*nscanp]*nscanl)
    X1 = np.array([[0.]*nscanp]*nscanl)
    X2 = np.array([[0.]*nscanp]*nscanl)
    X3 = np.array([[0.]*nscanp]*nscanl)
    X4 = np.array([[0.]*nscanp]*nscanl)
    indexl0 = np.array([[0.]*nscanp]*nscanl)
    indexp0 = np.array([[0.]*nscanp]*nscanl)
    s1 = np.array([[0.]*nscanp]*nscanl)
    s2 = np.array([[0.]*nscanp]*nscanl)
    s3 = np.array([[0.]*nscanp]*nscanl)
    s4 = np.array([[0.]*nscanp]*nscanl)
    fgeo1 = np.array([[0.]*nscanp]*nscanl)
    fgeo2 = np.array([[0.]*nscanp]*nscanl)
    fgeo3 = np.array([[0.]*nscanp]*nscanl)
    fgeo4 = np.array([[0.]*nscanp]*nscanl)

    indexl0 = ((lat+90)/0.28125).astype(np.int32)
    indexp0 = (abs((lon-0.14))/0.28125).astype(np.int32)
    indexl0[lat>90] = 0
    indexp0[lon>360] = 0

    LATP = np.array([[0.]*nlon]*nlat)
    LONP = np.array([[0.]*nlon]*nlat)
    #print 'lat.shape',lat.shape
    #print 'index.shape',indexl0.shape
    #print 'nlat=',nlat,'nlon=',nlon
    for i in range(nlat):
        LATP[i,:] = i*0.28125-90
    for j in range(nlon):
    	LONP[:,j] = 0.14+j*0.28125
    indexl1 = indexl0+1
    indexp1 = indexp0+1
    #print indexl0
    #print indexp0
    indexl1[indexl1>=nlat-1]=nlat-2
    indexp1[indexp1>=nlon-1]=nlon-1
    s1 = sqrt((lat-LATP[indexl0,indexp0])**2+(lon-LONP[indexl0,indexp0])**2)
    s2 = sqrt((lat-LATP[indexl0,indexp1])**2+(lon-LONP[indexl0,indexp1])**2)
    s3 = sqrt((lat-LATP[indexl1,indexp1])**2+(lon-LONP[indexl1,indexp1])**2)
    s4 = sqrt((lat-LATP[indexl1,indexp0])**2+(lon-LONP[indexl1,indexp0])**2)
    sumgeo=1/s1+1/s2+1/s3+1/s4
    fgeo1 = 1/s1/sumgeo
    fgeo2 = 1/s2/sumgeo
    fgeo3 = 1/s3/sumgeo
    fgeo4 = 1/s4/sumgeo
    X1t1 = interpolation(fgeo1,fgeo2,fgeo3,fgeo4,indexl0,indexp0,indexl1,indexp1,X1a[0,:,:])
    X1t2 = interpolation(fgeo1,fgeo2,fgeo3,fgeo4,indexl0,indexp0,indexl1,indexp1,X1b[0,:,:])
    X1t3 = interpolation(fgeo1,fgeo2,fgeo3,fgeo4,indexl0,indexp0,indexl1,indexp1,X1c[0,:,:])
    X2t1 = interpolation(fgeo1,fgeo2,fgeo3,fgeo4,indexl0,indexp0,indexl1,indexp1,X2a[0,:,:])
    X2t2 = interpolation(fgeo1,fgeo2,fgeo3,fgeo4,indexl0,indexp0,indexl1,indexp1,X2b[0,:,:])
    X2t3 = interpolation(fgeo1,fgeo2,fgeo3,fgeo4,indexl0,indexp0,indexl1,indexp1,X2c[0,:,:])
    X3t1 = interpolation(fgeo1,fgeo2,fgeo3,fgeo4,indexl0,indexp0,indexl1,indexp1,X3a[0,:,:])
    X3t2 = interpolation(fgeo1,fgeo2,fgeo3,fgeo4,indexl0,indexp0,indexl1,indexp1,X3b[0,:,:])
    X3t3 = interpolation(fgeo1,fgeo2,fgeo3,fgeo4,indexl0,indexp0,indexl1,indexp1,X3c[0,:,:])
    X4t1 = interpolation(fgeo1,fgeo2,fgeo3,fgeo4,indexl0,indexp0,indexl1,indexp1,X4a)
    X4t2 = interpolation(fgeo1,fgeo2,fgeo3,fgeo4,indexl0,indexp0,indexl1,indexp1,X4b)
    X4t3 = interpolation(fgeo1,fgeo2,fgeo3,fgeo4,indexl0,indexp0,indexl1,indexp1,X4c)
    #print '%%%%',X3t1[0:5,0]
    
    for i in range(nscanl):
        ddscnt = datetime.timedelta(days=int(daycnt[i]))
        dmscnt = datetime.timedelta(hours=float(mscnt[i]/1000./60./60.))
        pointtime = starttime+ddscnt+dmscnt
        dptime1 = (pointtime-starttime).seconds-(nwptime1-starttime).seconds
        dptime2 = (pointtime-starttime).seconds-(nwptime2-starttime).seconds
        dptime3 = (pointtime-starttime).seconds-(nwptime3-starttime).seconds
        #print '%%%',dptime1,dptime2,dptime3,hftime
        #print '===',pointtime,nwptime1,nwptime2,nwptime3
        if(dptime2<0):
            fractime2[i,:] = dptime1/hftime
            fractime1[i,:] = abs(dptime2/hftime)
            X1[i,:] = X1t1[i,:]*fractime1[i,:]+X1t2[i,:]*fractime2[i,:]
            X2[i,:] = X2t1[i,:]*fractime1[i,:]+X2t2[i,:]*fractime2[i,:]
            X3[i,:] = X3t1[i,:]*fractime1[i,:]+X3t2[i,:]*fractime2[i,:]
            X4[i,:] = X4t1[i,:]*fractime1[i,:]+X4t2[i,:]*fractime2[i,:]
        else:
            fractime2[i,:] = dptime2/hftime
            fractime1[i,:] = abs(dptime3/hftime)
            X1[i,:] = X1t2[i,:]*fractime1[i,:]+X1t3[i,:]*fractime2[i,:]
            X2[i,:] = X2t2[i,:]*fractime1[i,:]+X2t3[i,:]*fractime2[i,:]
            X3[i,:] = X3t2[i,:]*fractime1[i,:]+X3t3[i,:]*fractime2[i,:]
            X4[i,:] = X4t2[i,:]*fractime1[i,:]+X4t3[i,:]*fractime2[i,:]  
    X1[idlat]=np.NaN
    X1[idlon]=np.NaN
    X2[idlat]=np.NaN
    X2[idlon]=np.NaN
    X3[idlat]=np.NaN
    X3[idlon]=np.NaN
    X4[idlat]=np.NaN
    X4[idlon]=np.NaN

 
    fout = h5py.File(outputfile,'w')
    fout.create_dataset('H300-H1000',data=X1)
    fout.create_dataset('H50-H200',data=X2)
    fout.create_dataset('Ts',data=X3)
    fout.create_dataset('TPW',data=X4)
    fout.create_dataset('Latitude',data=lat)
    fout.create_dataset('Longitude',data=lon)
    fout.close()
    
    
    
