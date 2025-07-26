# -*- coding: utf-8 -*-
"""
@ Time: 2019-11-24
@ author: LiangHongli
@ Mail: Helen_Liang1@outlook.com
Some constants about spectral feature or thresholds.
"""
import numpy as np
import my_funcs as mf

# class _const:
#     class ConstError(TypeError): pass
#     class ConstCaseError(ConstError): pass
#
#     def __setattr__(self, name, value):
#         if name in self.__dict__:
#             raise self.ConstError("can't change const %s" % name)
#         if not name.isupper():
#             raise self.ConstCaseError('const name "%s" is not all uppercase' % name)
#         self.__dict__[name] = value
#
# const = _const()
# const.PI = 3.14

## Hard limits for control of input profile, unit(T)=K,unit(q)=kg/kg,unit(p)=hPa
consts = {'tmax':400.0,
          'tmin':90.0,
          'qmax_ppmv':0.60E+06,
          'qmin':0.1E-10,
          'qmax_kg':0.373,
          'pmax':1100.0,
          'pmin':400.0,
          'ozmax_ppmv':1000,
          'ozmin':0.1E-10,
          'ozmax_kg':1.657E-3,
          'elemax':20,
          'zenmax':85.3}

############ IRAS channel information #############################################

ch_info = {
    'ch1':{'wn':np.arange(654.5,686.8,0.25),'range':np.arange(1,131),'cwl':'14.95μm',
            'scale':0.9999998083E+00,'offset':-0.3184942663E-02,'cv':0.6690863333E+03},
    'ch3':{'wn':np.arange(668,718.3,0.25),'range':np.arange(55,257),'cwl':'14.49μm',
            'scale':0.9999885010E+00,'offset':-0.1124297000E-01,'cv':0.6911181365E+03},
    'ch4':{'wn':np.arange(677.0,729.1,0.25),'range':np.arange(91,300),'cwl':'14.22μm',
            'scale':0.9999839601E+00,'offset':-0.9516182137E-02,'cv':0.7031029997E+03},
    'ch5':{'wn':np.arange(680.25,749.35,0.25),'range':np.arange(104,381),'cwl':'13.97μm',
            'scale':0.9999763995E+00,'offset':-0.9315612816E-02,'cv':0.7151573157E+03},
    'ch6':{'wn':np.arange(704.25,763.6,0.25),'range':np.arange(200,438),'cwl':'13.64μm',
            'scale':0.9999678630E+00,'offset':-0.7189500551E-02,'cv':0.7323479969E+03},
    'ch7':{'wn':np.arange(712.5,791.1,0.25),'range':np.arange(233,548),'cwl':'13.35μm',
            'scale':0.9999438863E+00,'offset':-0.7245686884E-02,'cv':0.7484521317E+03},
    'ch8':{'wn':np.arange(722.75,890.85,0.25),'range':np.arange(274,947),'cwl':'12.47μm',
            'scale':0.9997914422E+00,'offset':0.3325896357E-02,'cv':0.8032521854E+03},
    'ch9':{'wn':np.arange(811.0,1002.6,0.25),'range':np.arange(627,1394),'cwl':'11.11μm',
            'scale':0.9995977979E+00,'offset':0.4527192411E-01,'cv':0.8997182588E+03},
    'ch10':{'wn':np.arange(933.,1107.85,0.25),'range':np.arange(1115,1815),'cwl':'9.71μm',
            'scale':0.9997836032E+00,'offset':0.3768100558E-01,'cv':0.1033146922E+04},
    'ch11':{'wn':np.arange(1157.75,1526.3,0.25),'range':np.arange(0,1475),'cwl':'7.43μm',
            'scale':0.9990157476E+00,'offset':0.2490877067E+00,'cv':0.1339938208E+04},
    'ch12':{'wn':np.arange(1257.5,1490.8,0.25),'range':np.arange(399,1333),'cwl':'7.33μm',
            'scale':0.9994915277E+00,'offset':0.1314627912E+00,'cv':0.1363129419E+04},
    'ch13':{'wn':np.arange(1315.5,1699.5,0.25),'range':np.arange(631,2167),'cwl':'6.52μm',
            'scale':0.9990453787E+00,'offset':0.2749910485E+00,'cv':0.1528937180E+04},
    'ch14':{'wn':np.arange(2136.75,2247.0,0.25),'range':np.arange(129,570),'cwl':'4.57μm',
            'scale':0.9999156986E+00,'offset':0.3360951630E-01,'cv':0.2190051002E+04},
    'ch15': {'wn': np.arange(2104.5, 2290.35, 0.25), 'range': np.arange(0, 744),'cwl':'4.52μm',
            'scale': 0.9999002412E+00, 'offset': 0.4023568283E-01, 'cv': 0.2212570457E+04},
    }

#### feature names for the GBT input #############
feature_names101 = []
nlv = 101
for k in range(nlv*3):
    if k//nlv==0:
        name = 't_lev'+str(k+1)
        feature_names101.append(name)
    elif k//nlv==1:
        name = 'q_lev'+str(k%nlv+1)
        feature_names101.append(name)
    elif k//nlv==2:
        name = 'o3_lev' + str(k%nlv + 1)
        feature_names101.append(name)

for s in ['t_2m','t_srf','srf_type','qr2','qw','qw2','qr_tdif','qr21','qr41','qr','qr3',
          'qr4','qr_tdif2','qr21_tdif','qr2_div_qtw','qr21qr_div_qtw','tr','tr2','tfw','tfu','o3r',
          'o3r21','o3r_tdif','o3r21_tdif','o3r2_ow','o3r23_div_ow','o3r_ow','o3r_ow21','ow','ow2']:
    feature_names101.append(s)


# 'srf_type','elevation',
######################### 137L #########################################
feature_names137 = []
nlv = 137
for k in range(nlv*8):
    if k//nlv==0:
        name = 'p_lev'+str(k+1)
        feature_names137.append(name)
    elif k//nlv==1:
        name = 't_lev'+str(k%nlv+1)
        feature_names137.append(name)
    elif k//nlv==2:
        name = 'q_lev' + str(k%nlv + 1)
        feature_names137.append(name)
    elif k//nlv==3:
        name = 'o3_lev' + str(k%nlv + 1)
        feature_names137.append(name)

for s in ['t_srf','srf_type','elevation']:
    feature_names137.append(s)

nlayer = 136
for k in range(nlayer*8):
    if k//nlayer==0:
        name = 'wv_r2'+str(k%nlayer+1)
        feature_names137.append(name)
    elif k//nlayer==1:
        name = 't_r2'+str(k%nlayer+1)
        feature_names137.append(name)
    elif k//nlayer==2:
        name = 't_r'+str(k%nlayer+1)
        feature_names137.append(name)
    elif k//nlayer==3:
        name = 'wv_r05'+str(k%nlayer+1)
        feature_names137.append(name)
    elif k//nlayer==4:
        name = 'wv_r025'+str(k%nlayer+1)
        feature_names137.append(name)
    elif k//nlayer==5:
        name = 'wv_r'+str(k%nlayer+1)
        feature_names137.append(name)
    elif k//nlayer==6:
        name = 'wv_r3'+str(k%nlayer+1)
        feature_names137.append(name)
    elif k//nlayer==7:
        name = 'wv_r4'+str(k%nlayer+1)
        feature_names137.append(name)
################## transmittance threshold ####################
trans_limit = 1.E-12


