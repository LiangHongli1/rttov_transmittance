################################ this code is from a company ######################
from numpy import NaN
from numpy import *
import numpy as np


def BilinearInterpo(v1,v2,v3,v4,x1,x2,y1,y2,x,y):
    klev = v1.shape[0]
    vv1 = np.copy([0.]*klev)
    vv2 = np.copy([0.]*klev)
    v = np.copy([0.]*klev)
    vv1[:] = v1[:]+(v2[:]-v1[:])/(x2-x1)*(x-x1)
    vv2[:] = v3[:]+(v4[:]-v3[:])/(x2-x1)*(x-x1)
    v[:]   = vv1[:]+(vv2[:]-vv1[:])/(y2-y1)*(y-y1)
    return v

def InterpProfile(presi,presf,vari):
    lpresi = np.copy(presi)
    lpresf = np.copy(presf)
    lpresi = np.log(presi)
    lpresf = np.log(presf)
    klevi = presi.shape[0]
    klevf = presf.shape[0]
    varf = np.array([0.]*klevf)
    for jkf in range(klevf):
        for jki in range(klevi-1):
            p1 = np.copy(presi[jki])
            p2 = np.copy(presi[jki+1])
            lp1 = np.copy(lpresi[jki])
            lp2 = np.copy(lpresi[jki+1])
            if(presf[jkf]<p1)&(presf[jkf]>p2):
                t1 = vari[jki]
                t2 = vari[jki+1]         
                if(t2==0):
                    slope = 0.
                else:
                    slope = (t1-t2)/(lp1-lp2)
                varf[jkf] = t1 + slope*(lpresf[jkf]-lp1)
            else:
                if(jki==klevi-2)&(presf[jkf]<=p2):
                    varf[jkf] = vari[klevi-1]
                if (jki==0)&(presf[jkf]>=p1):
                    varf[jkf] = vari[jki]
    return varf


 





