# -*- coding: utf-8 -*-
"""
@time: 2019/6/18
@author: LiangHongli
functions: convolving mono radiances of IASI to specific channel;
plank function
"""
from scipy.interpolate import interp1d
import numpy as np
import scipy
from itertools import permutations
import random

h = 6.6262 * 10**-34 #普朗克常量，unit=Js
c = 3 * 10**8 #光速
k = 1.3806 * 10**-23 #玻尔兹曼常数，unit=JK-1
c1 = 1.191042953E-05 # first radiation constant(mW/(m2.sr.cm-4))
c2 = 1.4387774 # second radiation constant(cm.K)

def convolv(monorad,monospec,spec,srf):
    '''
    插值光谱响应函数，卷积radiance
    :param monorad: 单色radiance（IASI）
    :param lamda: 单色的波长
    :param spec：光谱响应函数所对应的波数
    :param srf: 光谱响应函数
    :return: convolved_rad, 卷积后的辐射率
    '''
    inter = interp1d(spec,srf,kind='cubic',bounds_error=False,fill_value='extrapolate',assume_sorted=False)
    asrf = inter(monospec)
    convolved_rad = scipy.trapz(monorad*asrf,monospec,dx=0.25)/scipy.trapz(asrf,monospec,dx=0.25) #卷积到IRAS通道12

    return convolved_rad

def plank(t,scale,offset,cv):
    '''
    利用普朗克公式计算辐射率radiance
    :param t: 温度
    :param scale: for band correction
    :oaram offset: for band correction
    :param cv: central wavenumber
    :return: 辐亮度radiance
    '''
    # x = np.exp(h*c/(lamda*k*t)) - 1
    # y = 2*h*c**2/lamda**5/x
    # 对用于辐亮度计算的温度做通道订正,通道12
    tt = t*scale+offset
    c11 = c1*cv**3
    c22 = c2*cv
    return c11/(np.exp(c22/tt)-1)

def plank_inv(rad,scale,offset,cv):
    '''
    普朗克逆函数，根据波长和radiance求黑体温度（亮温）
    :param rad: 辐亮度
    :param scale: for band correction
    :oaram offset: for band correction
    :param cv: central wavenumber
    :return: 亮温T
    '''
    # x = np.log(2*h*c*c/(lamda**5*rad)+1)
    # y = h*c/(k*lamda*x)
    c11 = c1*cv**3
    c22 = c2*cv
    tt = c22/(np.log(c11/rad+1))
    # 做通道订正
    bt = (tt-offset)/scale
    return bt

def generate_chpr_txt(fpath,channel,npro):
    '''
    指定需要计算的仪器通道和大气廓线的文本文件，以及需要保存的路径
    :param fpath: 函数生成的txt文本需要保存的路径
    :param channel: RTTOV需要计算的通道，整数，标量或数组
    :param npro: 需要计算的廓线条数
    :return: None
    '''
    if isinstance(channel,int):
        cha = channel
        pro = np.arange(1,npro+1).reshape([-1,1])
        nch = 1
    else:
        cha = np.array(channel)
        pro = np.arange(1,npro+1).reshape([-1,1])
        nch = len(cha)
    channels = np.tile(cha,(npro,1))
    profiles = np.tile(pro,(1,nch))
    chname = fpath/'channels.txt'
    pname = fpath/'lprofiles.txt'
    np.savetxt(chname,channels,fmt='%d')
    np.savetxt(pname,profiles,fmt='%d')
    return

# Calculating the 2m pressure through pressure-height function
def calc_p_2m(p_srf,tmp_srf,q_srf):
    gamma = 0.0342 # unit=K/m, Virtual temperature declining rate
    tv = tmp_srf*(1+0.608*q_srf) # virtual temperature, unit=K
    return p_srf*(tv-gamma*2)/tv
# Converting the units from kg/kg to ppmv and converting back
def kg2ppmv(x,M):
    return x*1.0E6*28.9647/M
def ppmv2kg(x,M):
    return x*1.0E-6*M/28.9647
###########################################################################################
def profile_selection(profiles,N,n):
    '''
    根据欧式距离，进行第一轮廓线筛选，选出和已有廓线集的廓线相比、欧式距离最大的廓线
    :param profile: array,shape=(101,3)，已加入备案的廓线
    :param N: int, the number of profiles that will be selected
    :param n: int, iterations of the selection
    :return: distance，如果返回true，则返回profile2的标准差
    '''
    r,c = profiles.shape
    # combinations = list(permutations(range(r),N))
    random.seed(2020)
    seeds = random.sample(range(10000),n)
    distances = []
    for j in seeds:
        random.seed(j)
        index = random.sample(range(r),N)
        x = profiles[index,:]
        dist = 0
        for k in range(c):
            bins,_ = np.histogram(x[:,k],N)
            dist += np.sum(np.abs(bins-1))

        distances.append(dist)
    print(distances)
    dmin_index = np.argsort(np.array(distances))[0]
    dmin_seed = seeds[dmin_index]
    random.seed(dmin_seed)
    index = random.sample(range(r),N)

    return index,distances[dmin_index]

def stat_month(y):
    x = np.arange(1, 13)
    months = np.zeros(12)
    for k in x:
        months[k - 1] = len(y[y == k])

    return months

def rh2q(rh,temperature,pressure):
    '''
    Convert relative humidity(%) to specific humidity(kg/kg)
    :param rh: array, float, relativu humidity
    :return: specific humidity, unit=kg/kg
    '''
    Rair = 287
    Rvapor = 461
    dens_air = pressure/temperature/Rair  # P=ρRT
    t = temperature-273.15
    es = 6.112*np.exp(17.67*t/(t+243.5)) # contruated water vapor pressure
    epres = es*rh
    dense_vapor = epres/temperature/Rvapor
    return 0.622*rh*dense_vapor/(dens_air-dense_vapor)
#########################################################################################
##################### interpolation functions ##########################################
def bilinear_interp(v1,v2,v3,v4,x1,x2,y1,y2,x,y):
    """
    bilinear interpolation for O-B( interpolate gridded data to ungridded data)
    :param v1: value of the first point, coordinate=(x1,y1), scalar or vector
    :param v2: value of the second point, coordinate=(x1,y2)
    :param v3: value of the third point, coordinate=(x2,y1)
    :param v4: value of the fourth point, coordinate=(x2,y2)
    :param x1: coordinate, latitude
    :param x2: coordinate, latitude
    :param y1: coordinate, longitude
    :param y2: coordinate, longitude
    :param x: coordinate, latitude
    :param y: coordinate, longitude
    :return: the interpolated value(s)
    """
    dx = x2-x1
    dy = y2-y1
    f1 = (v1*(x2-x)+v2*(x-x1))/dx
    f2 = (v3*(x2-x)+v4*(x-x1))/dx
    return (f1*(y2-y)+f2*(y-y1))/dy

def idw_interp(v1,v2,v3,v4,x1,x2,y1,y2,x,y):
    d1 = np.sqrt((x-x1)**2+(y-y1)**2)
    d2 = np.sqrt((x-x1)**2+(y-y2)**2)
    d3 = np.sqrt((x-x2)**2+(y-y1)**2)
    d4 = np.sqrt((x-x2)**2+(y-y2)**2)
    return (v1/d1+v2/d2+v3/d3+v4/d4)/(1/d1+1/d2+1/d3+1/d4)

def interp_profile(p_source,v_source,p_target):
    '''
    interpolate profiles with linear method
    :param p_source: pressure, unit=hPa, vector
    :param v_source: variable, vector
    :param p_target:
    :return: interpolated variable
    '''
    interp = interp1d(p_source,v_source,kind='linear',fill_value='extrapolate')
    return interp(p_target)

