# -*- coding: utf-8 -*-
"""
@ Time: 2019/12/31
@ author: LiangHongli
@ Mail: Helen_Liang1@outlook.com
Construct and calculate new predictors for GBT to regress level-to-space transmittance.
Note: 1. new predictors are primarily mimicked from the RTTOV optical depth predictors.
2. There is no pressure variables in the input profile.
"""
import numpy as np

def var_star(profile,nlevel):
    '''
    Compute a mean profile and then compute layer-based variables
    :param profile: array, shape=[n_sample,n_level*n_feature], pressue should not be included in the features
    :param nlevel: int, scalar, number of atmospheric level
    :return: a vector presenting the mean layer-based profile, shape=(nfeature,nlayer)
    '''
    refer = np.mean(profile,axis=0)
    n_feature = refer.reshape(-1,nlevel).shape[0]
    refer = refer.reshape(n_feature,-1)
    n_feature = refer.shape[0]
    _,n = profile.shape
    var = np.zeros((n_feature,nlevel-1),dtype=np.float)
    for k in range(1,nlevel):
        var[:,k-1] = (refer[:,k]+refer[:,k-1])/2

    # print(var.reshape((n_feature,-1)))
    return var

def layer_profile(profile,nlevel):
    '''
    var_layer[j] = (var_level[j]+var_level[j-1])/2, compute layer variables
    :param profile: array, shape=[n_sample,n_level*n_feature]
    :param nlevel: int, scalar, number of atmospheric level
    :return: array, layer-based variables,shape=(nsample,nfeature,nlayer)
    '''
    n_feature = profile[0,:].reshape(-1,nlevel).shape[0]
    r,n = profile.shape
    prof = profile.reshape((r,n_feature,-1))
    var = np.zeros((r,n_feature,nlevel-1),dtype=np.float)
    for k in range(1,nlevel):
        var[:,:,k-1] = (prof[:,:,k]+prof[:,:,k-1])/2

    return var

def predictors(refer_prof,layer_prof,plevel,nlayer):
    '''
    Take a page from the RTTOV optical depth predictors, just some wv and two CO2 predictors for now
    :param refer_prof: a vector, the mean layer-based profile
    :param layer_prof: array, layer-based profile, shape=(nsample,nfeature,nlayer)
    :param nlevel: int, scalar, number of level
    :return: array, shape=(n_sample,n_feature,n_layer), some new predictors, hopefully it could do a favor for the GBT regression
    '''
    # print(refer_prof)
    xr = layer_prof/refer_prof
    ###################### fixed gas predictors #################
    tr = xr[:,0,:]
    tr2 = tr**2
    tdif = layer_prof[:,0,:]-refer_prof[0,:]
############# water vapor predictors #############
    qr = xr[:,1,:]
    qr2 = qr**2
    qr_tdif = qr*tdif
    qr21 = qr**0.5
    qr41 = qr**0.25
    qr3 = qr**3
    qr4 = qr**4
    qr_tdif2 = qr*tdif*np.abs(tdif)
    qr21_tdif = qr21*tdif
############### o3 predictors ######################3
    o3r = xr[:,2,:]
    o3r21 = o3r**0.5
    o3r_tdif = o3r*tdif
    o3r2 = o3r**2
    o3r21_tdif = o3r21*tdif
################## complex predictors ########################
    wvars = np.zeros((layer_prof.shape[0],4,layer_prof.shape[-1]))
    wvars_star = np.zeros((4,layer_prof.shape[-1]))
    wpreds = np.zeros((layer_prof.shape[0],6,layer_prof.shape[-1]))
    ## wvars: [tpk, qpk, tqpk, opk]
    ## wvars_star: [tp_stark, qp_stark, tqp_stark, op_stark]
    ## wpreds: [tw,tfu,tfw,qw,qtw,ow]

    for k in range(nlayer):
        pw = plevel[k+1]*(plevel[k+1]-plevel[k])
        # print(pw,refer_prof[:,k],layer_prof[:,:,k])
        wvars[:,:3,k] = pw*layer_prof[:,:,k]
        wvars_star[:3,k] = pw*refer_prof[:,k]
        wvars[:,3,k] = pw*layer_prof[:,0,k]*layer_prof[:,1,k]
        wvars_star[3,k] = pw*refer_prof[0,k]*refer_prof[1,k]
        if k==0:
            wpreds[:,0,k] = wvars[:,0,k]/wvars_star[0,k]
            wpreds[:,1,k] = layer_prof[:,0,k]/refer_prof[0,k]
            wpreds[:,3,k] = wvars[:,1,k]/wvars_star[1,k]
            wpreds[:,4,k] = wvars[:,2,k]/wvars_star[2,k]
            wpreds[:,5,k] = wvars[:,3,k]/wvars_star[3,k]
        else:
            wpreds[:,0,k] = np.sum(wvars[:,0,:k],axis=1)/np.sum(wvars_star[0,:k])
            wpreds[:,1,k] = np.sum(layer_prof[:,0,:k],axis=1)/np.sum(refer_prof[0,:k])
            wpreds[:,3,k] = np.sum(wvars[:,1,:k],axis=1)/np.sum(wvars_star[1,:k])
            wpreds[:,4,k] = np.sum(wvars[:,2,:k],axis=1)/np.sum(wvars_star[2,:k])
            wpreds[:,5,k] = np.sum(wvars[:,3,:k],axis=1)/np.sum(wvars_star[3,:k])

        if k==0 or k==1:
            wpreds[:,2,k] = layer_prof[:,0,k]/refer_prof[0,k]
        else:
            wpreds[:,2,k] = np.sum(layer_prof[:,0,1:k],axis=1)/np.sum(refer_prof[0,1:k])
#################################################################################
    qr2_div_qtw = (qr2/wpreds[:,4,:])[:,np.newaxis,:]
    qr21qr_div_qtw = (qr21*qr/wpreds[:,4,:])[:,np.newaxis,:]
    qr2 = qr2[:,np.newaxis,:]
    qw = wpreds[:,3,:][:,np.newaxis,:]
    qw2 = (wpreds[:,3,:]**2)[:,np.newaxis,:]
    qr_tdif = qr_tdif[:,np.newaxis,:]
    qr21 = qr21[:,np.newaxis,:]
    qr41 = qr41[:,np.newaxis,:]
    qr = qr[:,np.newaxis,:]
    qr3 = qr3[:,np.newaxis,:]
    qr4 = qr4[:,np.newaxis,:]
    qr_tdif2 = qr_tdif2[:,np.newaxis,:]
    qr21_tdif = qr21_tdif[:,np.newaxis,:]

#####################################################################
    o3r = o3r[:,np.newaxis,:]
    o3r21 = o3r21[:,np.newaxis,:]
    o3r_tdif = o3r_tdif[:,np.newaxis,:]
    o3r2 = o3r2[:,np.newaxis,:]
    o3r21_tdif = o3r21_tdif[:,np.newaxis,:]
    ow = wpreds[:,5,:][:,np.newaxis,:]
    o3r2_ow = o3r2*ow
    o3r23_div_ow = o3r*o3r21/ow
    o3r_ow = o3r*ow
    o3r_ow21 = o3r* ow**0.5
    ow2 = ow**2

########################################################################
    tr = tr[:,np.newaxis,:]
    tr2 = tr2[:,np.newaxis,:]
    tfw = wpreds[:,2,:][:,np.newaxis,:]
    tfu = wpreds[:,1,:][:,np.newaxis,:]
    print(qr2_div_qtw.shape,qr21qr_div_qtw.shape,o3r23_div_ow.shape)
    new_predictors = np.concatenate((qr2,qw,qw2,qr_tdif,qr21,qr41,qr,qr3,qr4,qr_tdif2,qr21_tdif,
                                     qr2_div_qtw,qr21qr_div_qtw,tr,tr2,tfw,tfu,
                                     o3r,o3r21,o3r_tdif,o3r21_tdif,o3r2_ow,o3r23_div_ow,o3r_ow,
                                     o3r_ow21,ow,ow2),axis=1)
    return new_predictors




