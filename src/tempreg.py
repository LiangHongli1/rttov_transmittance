# -*- coding: utf-8 -*-
"""
@ Time: {DATE},{TIME}
@ author: LiangHongli
@ Mail: Helen_Liang1@outlook.com
"""

import numpy as np
import pandas as pd
import random
import time
import h5py
from pathlib import Path
import xgboost as xgb
from sklearn.svm import SVR
from sklearn import preprocessing
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor,AdaBoostRegressor
from sklearn.model_selection import cross_val_score,GridSearchCV,train_test_split
from sklearn.metrics import mean_absolute_error,mean_squared_log_error,mean_squared_error,r2_score
from sklearn.preprocessing import PowerTransformer,StandardScaler,MinMaxScaler
import joblib

from plot_map_funcs import plot_bt,plot_bias_bt,plot_error_level,plot_mre,scatter_line_trans,\
    plot_trans_level,plot_single_var,bar_feature_rank,plot_bias_ratio_lev,plot_std_lev
from calc_BT import bt_GBT

from LUT import ch_info,feature_names101,trans_limit

import my_funcs as mf

def gbt_reg(loss):
    reg = GradientBoostingRegressor(n_estimators=290,subsample=0.8,
                                    learning_rate=0.045,
                                    max_depth = 3,
                                    n_iter_no_change=500,
                                    random_state=2021,loss=loss,
                                    validation_fraction=0.1
                                    )
    # model = MultiOutputRegressor(reg,n_jobs=-1)
    model = reg
    return model
#max_features=100
def xgb_reg(ntree,ndepth):
    #用sklearn的API
    model = xgb.XGBRegressor(objective="reg:squarederror",
                            max_depth=ndepth,
                            learning_rate=0.045,
                            n_estimators=ntree,
                            verbosity=0,
                            booster='gbtree',
                             n_jobs=-1,
                            random_state=2020)
    # model = MultiOutputRegressor(bst,n_jobs=-1)
#        reg_multi.fit(self.trainx,self.trainy,eval_set=[(self.trainx,self.trainy),(self.testx,self.testy)],eval_metric='mae')
##      train_preds = bst.predict(trainx)
##      test_preds = bst.predict(testx)
#       xgb.plot_importance(bst)
#       xgb.plot_tree(bst,num_trees=3)
#       xgb.to_graphviz(bst,num_trees=3)
    return model
#gamma=0.001,
def svr_reg():
#        params = {"C":[3.,4.,4.5,5.,8,10]}
    reg = SVR(C=2.,epsilon=0.0001,
              kernel='rbf',gamma=0.0001,
              max_iter=-1,shrinking=True,
              tol=0.0001)

    model = reg
    # model = MultiOutputRegressor(reg,n_jobs=-1)
#        reg_multi = RandomizedSearchCV(estimator=MultiOutputRegressor(reg),param_distributions=params,n_iter=10,
#                                       cv=5,n_jobs=-1,refit=True,return_train_score=True)
    return model

def rf_reg():
    model = RandomForestRegressor(n_estimators=300,max_depth=5,
                                random_state=2020,criterion='mse')
#        reg_multi = RandomizedSearchCV(clf,param_distributions=param_list,n_iter=10,cv=5)
    return model

def training(estimator,trainx,trainy,testx,testy):
    t1 = time.perf_counter()
    estimator.fit(trainx,trainy)
    # ytrain = -np.log(trainy)
    # estimator.fit(trainx,ytrain)
    t2_train = time.perf_counter()-t1
    # print('Fitting GBT costs %.4f seconds.' % t2_train)
    train_preds = estimator.predict(trainx)
    t1 = time.perf_counter()

    test_preds = estimator.predict(testx)
    t2_test = time.perf_counter()-t1

    # train_preds = np.exp(-train_preds)
    # test_preds = np.exp(-test_preds)
    # print('Testing GBT costs %.4f seconds.' % t2_test)
    # print('min_testy=%.4f,min_testpred=%.4f,max_testy=%.4f,max_testpred=%.4f' % (np.min(testy),np.min(test_preds),np.max(testy),np.max(test_preds)))
    train_preds[train_preds<0] = trans_limit
    test_preds[test_preds<0] = trans_limit
    feature_rank = estimator.feature_importances_

    # train_abe = mean_absolute_error(trainy,train_preds)
    # test_abe = mean_absolute_error(testy,test_preds)
    # train_rmse = mean_squared_error(trainy,train_preds)**0.5
    # test_rmse = mean_squared_error(testy,test_preds)**0.5
    # train_rmse = r2_score(trainy,train_preds)**0.5
    # test_rmse = r2_score(testy,test_preds)**0.5
    # train_msle = 1.0E04*mean_squared_log_error(1+trainy,1+train_preds)
    # test_msle = 1.0E04*mean_squared_log_error(1+testy,1+test_preds)
    return train_preds,test_preds,feature_rank,t2_train,t2_test#,train_abe,test_abe,train_rmse,test_rmse,train_msle,test_msle,feature_rank

def cv(estimator,x_train,y_train):
    cv = cross_val_score(estimator,
                         x_train,
                         y_train,
                         scoring='neg_mean_squared_error',
                         n_jobs=-1,
                         cv = 5)
    return cv

def norm(x11,x22):
    scaler = preprocessing.StandardScaler().fit(x11)
    x1 = scaler.transform(x11)
    x2 = scaler.transform(x22)
    return x1,x2,scaler.mean_,scaler.scale_

def calc_err(x1,x2):
    bias_levs = np.mean(x1-x2,axis=0)
    rmse_levs = np.average((x1-x2)**2,axis=0)**0.5
    #每一个样本在所有气压层的相对误差均值
    mre = np.mean(np.abs(x1-x2)/x2,axis=1)
    sor_mre = np.where(mre==np.max(mre))[0]
    # print('相对误差最大的廓线位置：',sor_mre)
    return bias_levs, rmse_levs, mre

def main():
    xfile = Path(r'G:\DL_transmitance\revised datasets\final\dataset_ifs101L_v3.HDF')
    fp = Path(r'G:\DL_transmitance\profiles\model_level_definition.xlsx')
    # fweight = Path(r'G:\DL_transmitance\profiles\normed_weight.xlsx')
    # ntree = 1000
    # ndepth = 5
    # depth_name = 'depth'+str(ndepth)
    # ntree_name = str(ntree)+'tree'
    # figpath1 = Path(r'G:\DL_transmitance\figures\101L\final\reg_result')/depth_name/ntree_name
    # figpath1 = Path(r'G:\DL_transmitance\figures\101L\final\reg_result\depth5\100tree')
    modelpath = Path(r'G:\DL_transmitance\models\XGBoost')

    nlv = 101
    chs = [11,12,13]
    fignum = 1
    method = 'XGBoost'
    # loss = 'ls'
    # if method=='SVR':
    #     model = svr_reg()
    # elif method=='RF':
    #     model = rf_reg()
    # elif method=='GBT':
    #     model = gbt_reg(loss)
    # elif method=='XGBoost':
    #     model = xgb_reg(ntree,ndepth)
    ######### load the emissivity and surface2space transmittance etc. ################
    with h5py.File(xfile,'r') as f:
        X = f['X'][:,:]
        Y = f['Y'][:,:,:-1]
        tao_rttov = f['transmission_RTTOV'][:]
        bt_rttov = f['BT_RTTOV'][:]
        bt_true = f['BT_true'][:]
        emiss = f['emissivity'][:]
        new_preds = f['predictor_RTTOV'][:]
    df = pd.read_excel(fp,sheet_name='101L')
    p = np.array(df['ph[hPa]'].values,dtype=np.float)
    print('ifs-137: land %d, ocean %d' % (sum(X[:,308]==0),sum(X[:,308]==1)))
    # surftype = 1
    # if surftype==1:
    #     srf = ' ocean'
    # else:
    #     srf = ' land'
    # condition = np.where(X[:,308]==surftype)[0] # 1 = ocean, 0 = land
    # X = X[condition,:]
    # Y = Y[:,condition,:]
    # tao_rttov = tao_rttov[condition,:,:]
    # bt_rttov = bt_rttov[condition,:]
    # bt_true = bt_true[condition,:]
    # emiss = emiss[condition,:]
    # new_preds = new_preds[condition,:,:]
    # condition = X[:,201]<0.01
    # kick_index = np.where(condition)[0]
    # kick_index = 393
    # X = np.delete(X,kick_index,axis=0)
    lon_lat = X[:,[309,310]]
    X = np.delete(X,[303,305,306,309,310,311,312],axis=1) # kick out 2m variables and longitude, latitude, srf_type, elevation
    npro = X.shape[0]

    print('There are %d profiles' % npro)
    # rand_num = np.random.randn(npro,1)
    # X = np.concatenate((X,rand_num),axis=1)

    # Y = np.delete(Y,kick_index,axis=1)
    # emiss = np.delete(emiss,kick_index,axis=0)
    # tao_rttov = np.delete(tao_rttov,kick_index,axis=0)
    # bt_rttov = np.delete(bt_rttov,kick_index,axis=0)
    # bt_true = np.delete(bt_true,kick_index,axis=0)
    # new_preds = np.delete(new_preds,kick_index,axis=0)
    #
################################# only ocean ##############################3
    ############ Calculate new predictors and concatenate them to the current X ############
    seed = 1269
    bt_train = np.zeros((1124,9),dtype=float)
    bt_test = np.zeros((282,9),dtype=float)
    # sample_ratio = [0.25,0.5,0.7,0.9,1]
    sample_ratio = [1]
    height = [[44,89,97],[44,85,97],[44,75,97]] #100,200,300,
    ntrees = [300]
    ndepths = [3]
    for ndepth in ndepths:
        depth_name = 'depth'+str(ndepth)
        for ntree in ntrees:
            ntree_name = str(ntree)+'tree'
            figpath1 = Path(r'G:\DL_transmitance\figures\101L\final\reg_result')/depth_name/ntree_name
            tlog = open(figpath1/'time_log_xgboost_.txt','w')
    ###################################################################################
            model = xgb_reg(ntree,ndepth)
            for j,ch in enumerate(chs):
                channel = 'ch'+str(ch)
                title = ch_info[channel]['cwl']
                # df_weight = pd.read_excel(fweight,sheet_name=channel)
                # print(df_weight.values)
                # weights = np.array(df_weight.values[:,1],dtype=np.float)
                jheight = height[j]
                for ratio in sample_ratio:
                    random.seed(2020)
                    index = random.sample(range(npro),int(npro*ratio))
                    Xr = X[index,:]
                    Yr = Y[:,index,:]
                    tao_rttovr = tao_rttov[index,:,:]
                    bt_rttovr = bt_rttov[index,:]
                    bt_truer = bt_true[index,:]
                    emissr = emiss[index,:]
                    new_predsr = new_preds[index,:,:]

                    X_train,X_test,Y_train,Y_test = train_test_split(Xr,Yr[j,:,:],test_size=0.2,random_state=seed)
                    tsrf_train,tsrf_test = X_train[:,303],X_test[:,303]
                    tskin_train,tskin_test = X_train[:,304],X_test[:,304]
                    taosrf_train,taosrf_test,emis_train,emis_test = train_test_split(tao_rttovr[:,j,-1],emissr[:,j],test_size=0.2,random_state=seed)
                    tao_rttov_train,tao_rttov_test,bt_rttov_train,bt_rttov_test = train_test_split(tao_rttovr[:,j,:-1],
                                                                                                   bt_rttovr[:,j],test_size=0.2,random_state=seed)
                    lonlat_train,lonlat_test,bt_true_train,bt_true_test = train_test_split(lon_lat,bt_truer[:,j],test_size=0.2,random_state=seed)

                    trans_preds_train = np.ones((X_train.shape[0],nlv),dtype=np.float) ## save every laye's predicted transmittance to calculate BT
                    trans_preds_test = np.ones((X_test.shape[0],nlv),dtype=np.float) ## same as above
                    # trans_preds_independent = np.ones((XX.shape[0],nlv),dtype=np.float)
                    figpath = figpath1/method/str(ratio)
                    if figpath.exists():
                        pass
                    else:
                        figpath.mkdir(parents=True)

                    logname = method+'_'+channel+'.txt'
                    flog = open(figpath/logname,'w')       ### save the log of every layer's and channel's training
                    # X_train = np.delete(X_train,757,axis=0)
                    # X_test = np.delete(X_test,283,axis=0)
                    ############### Editted on 2019-1-2, do layer-by-layer regression, for feature selection ###############################
                    time_train = 0
                    time_test = 0
                    for ii in range(nlv):
                        trainy = Y_train[:,ii]
                        testy = Y_test[:,ii]

                        if ii>0:
                            newtrain,newtest,_,_ = train_test_split(new_predsr[:,:,ii-1],emissr[:,j],test_size=0.2,random_state=seed)
                            # trainx = np.concatenate((X_train[:,[ii,ii+101,ii+202]],newtrain),axis=1)
                            # testx = np.concatenate((X_test[:,[ii,ii+101,ii+202]],newtest),axis=1)
                            # XXii = np.concatenate((XX[:,[ii,ii+101,ii+202]],new_preds2[:,:,ii-1]),axis=1)
                            trainx = np.concatenate((X_train,newtrain),axis=1)
                            testx = np.concatenate((X_test,newtest),axis=1)
                            # XXii = np.concatenate((XX,new_preds2[:,:,ii-1]),axis=1)
                        else:
                            trainx = X_train
                            testx = X_test
                            # XXii = XX

                        # print(trainx.shape,testx.shape,XXii.shape)
                        # scaler = StandardScaler().fit(trainx)
                        # scaler2 = StandardScaler().fit(testx)
                        # trainx = scaler.transform(trainx)
                        # testx = scaler.transform(testx)
                        # XXii = scaler.transform(XXii)

                        # plt.figure()
                        # plt.plot(scaler.mean_[-27:],scaler2.mean_[-27:])
                        # plt.xlabel('mean value of trainx')
                        # plt.ylabel('mean value of testx')
                        # plt.show()
                        # plt.figure()
                        # plt.plot(scaler.scale_[-27:],scaler2.scale_[-27:])
                        # plt.xlabel('std value of trainx')
                        # plt.ylabel('std value of testx')
                        # plt.show()

                        # model = gbt_reg(loss)
                        # train_preds,test_preds,train_mae,test_mae,train_rmse,test_rmse,train_msle,test_msle,rank = training(model,trainx,trainy,testx,testy)
                        train_preds,test_preds,rank,ttrain,ttest = training(model,trainx,trainy,testx,testy)
                        trans_preds_test[:,ii] = test_preds
                        trans_preds_train[:,ii] = train_preds
                        time_train += ttrain
                        time_test += ttest
                        # trans_preds_independent[:,ii] = model.predict(XXii)
                        # trans_preds_independent[:,ii] = np.exp(-model.predict(XXii))

                        model_name = 'XGBoost_'+channel+'_level_'+str(ii)+'.m'
                        joblib.dump(model,modelpath/model_name)

                        ########### Plot feature ranking(top15) ####################
                        # if ii in jheight:
                        #     fignum += 1
                        #     plt.figure(fignum,dpi=300)
                        #     index = np.argsort(-rank)
                        #     # print(rank.shape,len(feature_names101))
                        #     features = []
                        #     for s in index[0:30]:
                        #         features.append(feature_names101[index[s]])
                        #
                        #     # print(features)
                        #     bar_feature_rank(rank[index[0:30]],features,title)
                        #     figname = channel+'feature_rank_layer'+str(ii+1)+'.png'
                        #     plt.savefig(figpath/figname)

                    # print('training time={:.4f} seconds, test time={:.4f}'.format(time_train,time_test))
                    tlog.write('training time={:.4f} seconds, test time={:.4f}'.format(time_train,time_test))
                    ############################## scatter pictures ##################################
                    bias_levs_train, rmse_levs_train, mre_train = calc_err(trans_preds_train,Y_train)
                    bias_levs_test, rmse_levs_test, mre_test = calc_err(trans_preds_test,Y_test)
                    bias_levs_train_rttov,rmse_levs_train_rttov,mre_train_rttov = calc_err(tao_rttov_train[:,:],Y_train)
                    bias_levs_test_rttov,rmse_levs_test_rttov,mre_test_rttov = calc_err(tao_rttov_test[:,:],Y_test)
                    fignum += 1
                    plt.figure(fignum,dpi=300)
                    # print(bias_levs_train.shape,rmse_levs_train.shape,bias_levs_train_rttov.shape,rmse_levs_train_rttov.shape,p.shape)
                    plot_error_level(bias_levs_train, rmse_levs_train,bias_levs_train_rttov,rmse_levs_train_rttov,p,title,method)
                    figname = channel+method+'_train_bias_nonorm.png'
                    plt.savefig(figpath/figname,dpi=300)
        
                    fignum += 1
                    plt.figure(fignum,dpi=300)
                    ratio_gbt = bias_levs_train/np.mean(Y_train,axis=0)
                    ratio_rttov = bias_levs_train_rttov/np.mean(Y_train,axis=0)
                    plot_bias_ratio_lev(ratio_gbt,ratio_rttov,p,title,method)
                    figname = channel+method+'biasratio_train.png'
                    plt.savefig(figpath/figname)
                    fignum += 1
                    plt.figure(fignum,dpi=300)
                    ratio_gbt = np.average((trans_preds_train-np.mean(trans_preds_train,axis=0))**2,axis=0)**0.5
                    ratio_rttov = np.average((tao_rttov_train-np.mean(tao_rttov_train,axis=0))**2,axis=0)**0.5
                    plot_std_lev(ratio_gbt,ratio_rttov,p,title,method)
                    figname = channel+method+'std_train.png'
                    plt.savefig(figpath/figname)
        
                    fignum += 1
                    plt.figure(fignum,dpi=300)
                    plot_mre(mre_train,mre_train_rttov,title,method)
                    figname = channel+method+'_train_mre_nonorm.png'
                    plt.savefig(figpath/figname)
                    # plt.close()
                    fignum += 1
                    plt.figure(fignum,dpi=300)
                    plot_error_level(bias_levs_test[:], rmse_levs_test[:],bias_levs_test_rttov[:],rmse_levs_test_rttov[:],p,title,method)
                    figname = channel+method+'_test_bias_nonorm.png'
                    plt.savefig(figpath/figname)
                    # plt.close()
        
                    fignum += 1
                    plt.figure(fignum,dpi=300)
                    ratio_gbt = bias_levs_test/np.mean(Y_test,axis=0)
                    ratio_rttov = bias_levs_test_rttov/np.mean(Y_test,axis=0)
                    plot_bias_ratio_lev(ratio_gbt,ratio_rttov,p,title,method)
                    figname = channel+method+'biasratio_test.png'
                    plt.savefig(figpath/figname)
                    fignum += 1
                    plt.figure(fignum,dpi=300)
                    ratio_gbt = np.average((trans_preds_test-np.mean(trans_preds_test,axis=0))**2,axis=0)**0.5
                    ratio_rttov = np.average((tao_rttov_test-np.mean(tao_rttov_test,axis=0))**2,axis=0)**0.5
                    plot_std_lev(ratio_gbt,ratio_rttov,p,title,method)
                    figname = channel+method+'std_test.png'
                    plt.savefig(figpath/figname)
        
                    fignum += 1
                    plt.figure(fignum,dpi=300)
                    plot_mre(mre_test,mre_test_rttov,title,method)
                    figname = channel+method+'_test_mre_nonorm.png'
                    plt.savefig(figpath/figname)
        #
        # ###################### independent profiles validation ###########################3
        #             # bias_levs, rmse_levs, mre = calc_err(trans_preds_independent,YY[j,:,:])
        #             # bias_levs_rttov,rmse_levs_rttov,mre_rttov = calc_err(tao_rttov2[:,j,:],YY[j,:,:])
        #             # fignum += 1
        #             # plt.figure(fignum,dpi=300)
        #             # plot_error_level(bias_levs, rmse_levs,bias_levs_rttov,rmse_levs_rttov,p,title)
        #             # figname = channel+method+'_independent_bias.png'
        #             # plt.savefig(figpath/figname)
        #             # # plt.close()
        #             # fignum += 1
        #             # plt.figure(fignum,dpi=300)
        #             # plot_mre(mre,mre_rttov,title)
        #             # figname = channel+method+'_independent_mre.png'
        #             # plt.savefig(figpath/figname)
        # ##############################################################################################
        #             # rmse_levs_train2 = np.sum(weights*rmse_levs_train[1:])
        #             # rmse_levs_train_rttov2 = np.sum(weights*rmse_levs_train_rttov[1:])
        #             # rmse_levs_test2 = np.sum(weights*rmse_levs_test[1:])
        #             # rmse_levs_test_rttov2 = np.sum(weights*rmse_levs_test_rttov[1:])
        #             # print('sum(rmse_levs_train)=%.4f' % rmse_levs_train2)
        #             # print('sum(rmse_levs_train_rttov)=%.4f' % rmse_levs_train_rttov2)
        #             # print('sum(rmse_levs_test)=%.4f' % rmse_levs_test2)
        #             # print('sum(rmse_levs_test_rttov)=%.4f' % rmse_levs_test_rttov2)
        #             # flog.write('sum(rmse_levs_train)=%.4f' % rmse_levs_train2)
        #             # flog.write('sum(rmse_levs_train_rttov)=%.4f' % rmse_levs_train_rttov2)
        #             # flog.write('sum(rmse_levs_test)=%.4f' % rmse_levs_test2)
        #             # flog.write('sum(rmse_levs_test_rttov)=%.4f' % rmse_levs_test_rttov2)
        #
                    train_mae = mean_absolute_error(Y_train,trans_preds_train)
                    test_mae = mean_absolute_error(Y_test,trans_preds_test)
                    train_rmse = mean_squared_error(Y_train,trans_preds_train)**0.5
                    test_rmse = mean_squared_error(Y_test,trans_preds_test)**0.5
                    # train_rmse = r2_score(Y_train,trans_preds_train)**0.5
                    # test_rmse = r2_score(Y_test,trans_preds_test)**0.5
                    train_msle = 1.0E04*mean_squared_log_error(1+Y_train,1+trans_preds_train)
                    test_msle = 1.0E04*mean_squared_log_error(1+Y_test,1+trans_preds_test)

                    # pro_test_mae = np.sum(trans_preds_test-Y_test,axis=1)
                    # pro_test_rmse = np.sum((trans_preds_test-Y_test)**2,axis=1)
                    # print('The index of maximum mae profile:',np.argsort(-pro_test_mae),np.argsort(-pro_test_rmse))

                    bias_rttov_test = mean_absolute_error(Y_test, tao_rttov_test[:,:])
                    rmse_rttov_test = mean_squared_error(Y_test, tao_rttov_test[:,:]) ** 0.5
                    # rmse_rttov_test = r2_score(Y_test, tao_rttov_test[:,:]) ** 0.5
                    msle_rttov_test = 1.0E04 * mean_squared_log_error(1+Y_test, 1+tao_rttov_test[:,:])

                    bias_rttov_train = mean_absolute_error(Y_train, tao_rttov_train[:,:])
                    rmse_rttov_train = mean_squared_error(Y_train, tao_rttov_train[:,:]) ** 0.5
                    # rmse_rttov_train = r2_score(Y_train, tao_rttov_train[:,:]) ** 0.5
                    msle_rttov_train = 1.0E04 * mean_squared_log_error(1 + Y_train, 1 + tao_rttov_train[:,:])
        #
                    fignum += 1
                    plt.figure(fignum,dpi=300)
                    scatter_line_trans(Y_train,trans_preds_train,tao_rttov_train[:,:],title,train_mae,train_rmse,
                                       train_msle,bias_rttov_train,rmse_rttov_train,msle_rttov_train,method)
                    figname = channel+method+'_train_trans_all'+'.png'
                    plt.savefig(figpath/figname)
                    # plt.close()
                    fignum += 1
                    plt.figure(fignum,dpi=300)
                    scatter_line_trans(Y_test,trans_preds_test,tao_rttov_test[:,:],title,test_mae,test_rmse,
                                       test_msle,bias_rttov_test,rmse_rttov_test,msle_rttov_test,method)
                    figname = channel+method+'_test_trans_all'+'.png'
                    plt.savefig(figpath/figname)
                    plt.close()
        #
        # ################################ independent profiles test ###################################################
        #             # mae = mean_absolute_error(YY[j,:,:],trans_preds_independent)
        #             # rmse = mean_squared_error(YY[j,:,:],trans_preds_independent)**0.5
        #             # msle = 1.0E04*mean_squared_log_error(1+YY[j,:,:],1+trans_preds_independent)
        #             # bias_rttov = mean_absolute_error(YY[j,:,:],tao_rttov2[:,j,:])
        #             # rmse_rttov = mean_squared_error(YY[j,:,:],tao_rttov2[:,j,:]) ** 0.5
        #             # msle_rttov = 1.0E04 * mean_squared_log_error(1+YY[j,:,:], 1+tao_rttov2[:,j,:])
        #             #
        #             # fignum += 1
        #             # plt.figure(fignum,dpi=300)
        #             # scatter_line_trans(YY[j,:,:],trans_preds_independent,tao_rttov2[:,j,:],title,mae,rmse,msle,bias_rttov,rmse_rttov,msle_rttov)
        #             # figname = channel+method+'_independent_trans_all'+'.png'
        #             # plt.savefig(figpath/figname)
        # ##############################################################################################################
                    ################# Calculating BT with RT function #################################################

                    ############################################################################################
                    tlevel_train = X_train[:,:nlv]
                    # taosrf_train = tao_rttov_train[:,-1] #trans_preds_train[:,-1]

                    bt_preds_train,btair_preds_train = bt_GBT(trans_preds_train[:,:],tlevel_train,emis_train,tsrf_train,tskin_train,taosrf_train,
                                            ch_info[channel]['scale'],ch_info[channel]['offset'],ch_info[channel]['cv'])
                    bt_train_ytrain,btair_train_ytrain = bt_GBT(Y_train[:,:],tlevel_train,emis_train,tsrf_train,tskin_train,taosrf_train,
                                               ch_info[channel]['scale'],ch_info[channel]['offset'],ch_info[channel]['cv'])
                    _,btair_train_rttov = bt_GBT(tao_rttov_train[:,:],tlevel_train,emis_train,tsrf_train,tskin_train,taosrf_train,
                                               ch_info[channel]['scale'],ch_info[channel]['offset'],ch_info[channel]['cv'])
                    # 画亮温曲线图
                    bias_pred_train = bt_preds_train-bt_true_train
                    bias_rtt_train = bt_rttov_train-bt_true_train

                    # index = np.where(np.abs(bias_pred_train)>=0.4)[0]
                    # print('The index of training profiles whose bias are greater than 0.4:',index)
                    x = np.arange(len(bias_pred_train))
                    landmask = X_train[:,305]==0
                    seamask = X_train[:,305]==1
                    xland = np.arange(sum(landmask))
                    xsea = np.arange(sum(seamask))
                    fignum += 1
                    plt.figure(fignum,dpi=300)
                    plt.subplot(121)
                    ax = np.concatenate((btair_preds_train[landmask]-btair_train_ytrain[landmask],btair_train_rttov[landmask]-btair_train_ytrain[landmask],
                                         btair_preds_train[seamask]-btair_train_ytrain[seamask],btair_train_rttov[seamask]-btair_train_ytrain[seamask]))
                    ymin = np.min(ax)
                    ymax = np.max(ax)
                    plot_bias_bt(xland,btair_preds_train[landmask]-btair_train_ytrain[landmask],btair_train_rttov[landmask]-btair_train_ytrain[landmask],
                                 title+' bias of BT_air(land)',method)
                    plt.ylabel('Deviation of BT_air(K)',fontsize=12)
                    plt.ylim([ymin,ymax])
                    plt.subplot(122)
                    plot_bias_bt(xsea,btair_preds_train[seamask]-btair_train_ytrain[seamask],btair_train_rttov[seamask]-btair_train_ytrain[seamask],
                                 title+' bias of BT_air(ocean)',method)
                    plt.yticks([])
                    plt.ylim([ymin,ymax])
                    figname = 'BTair_bias_train'+channel+'.png'
                    plt.savefig(figpath/figname)

                    fignum += 1
                    ax = np.concatenate((bias_pred_train[landmask],bias_rtt_train[landmask],bias_pred_train[seamask],bias_rtt_train[seamask]))
                    ymin = np.min(ax)
                    ymax = np.max(ax)
                    plt.figure(fignum,dpi=300)
                    plt.subplot(121)
                    plot_bias_bt(xland,bias_pred_train[landmask],bias_rtt_train[landmask],title+' land',method)
                    plt.ylabel('Deviation of BT(K)',fontsize=12)
                    plt.ylim([ymin,ymax])
                    plt.subplot(122)
                    plot_bias_bt(xsea,bias_pred_train[seamask],bias_rtt_train[seamask],title+' ocean',method)
                    plt.ylim([ymin,ymax])
                    plt.yticks([])
                    figname = 'BT_bias_train'+channel+'.png'
                    plt.savefig(figpath/figname)

                    mae_gbt = mean_absolute_error(bt_true_train,bt_preds_train)
                    mae_rtt = mean_absolute_error(bt_true_train,bt_rttov_train)
                    # mae_gbt = np.mean(bt_true_train-bt_preds_train)
                    # mae_rtt = np.mean(bt_true_train-bt_rttov_train)
                    rmse_gbt = mean_squared_error(bt_true_train,bt_preds_train)**0.5
                    rmse_rtt = mean_squared_error(bt_true_train,bt_rttov_train)**0.5
                    std_gbt = np.std(bt_preds_train)
                    std_rtt = np.std(bt_rttov_train)
                    # bt_gbtlog = 'train: GBT results:mae={:.4f}, rmse={:.4f}'.format(mae_gbt,rmse_gbt)
                    # bt_rttovlog = 'train: RTTOV results:mae={:.4f}, rmse={:.4f}'.format(mae_rtt,rmse_rtt)
                    bt_gbtlog = '{:.4f} {:.4f} {:.4f}'.format(mae_gbt,rmse_gbt,std_gbt)
                    bt_rttovlog = '{:.4f} {:.4f} {:.4f}'.format(mae_rtt,rmse_rtt,std_rtt)
                    flog.write(bt_gbtlog)
                    flog.write('\n')
                    flog.write(bt_rttovlog)
                    flog.write('\n')
        ############################################## Plot good and bad performance profile ###########
                    # bias_rank = np.argsort(np.abs(bias_pred_train))
                    # fignum += 1
                    # plt.figure(dpi=300)
                    # plot_single_var(X_train[bias_rank[0],:nlv],p,'temperature','t training good profile')
                    # figname = channel+'train_t_good.png'
                    # plt.savefig(figpath/figname)
                    # fignum += 1
                    # plt.figure(dpi=300)
                    # print('The index of training bad profile: %d' % bias_rank[-1])
                    # plot_single_var(X_train[bias_rank[-1],:nlv],p,'temperature','t training bad profile')
                    # figname = channel+'train_t_bad.png'
                    # plt.savefig(figpath/figname)
                    # fignum += 1
                    # plt.figure(dpi=300)
                    # plot_single_var(X_train[bias_rank[0],nlv:nlv*2],p,'water vapor','q training good profile')
                    # figname = channel+'train_q_good.png'
                    # plt.savefig(figpath/figname)
                    # fignum += 1
                    # plt.figure(dpi=300)
                    # plot_single_var(X_train[bias_rank[-1],nlv:nlv*2],p,'water vapor','q traning bad profile')
                    # figname = channel+'train_q_bad.png'
                    # plt.savefig(figpath/figname)
                    #
                    # fignum += 1
                    # plt.figure(dpi=300)
                    # plot_trans_level(Y_train[bias_rank[0],:],trans_preds_train[bias_rank[0],:],tao_rttov_train[bias_rank[0],:],p,'predicted transmittane','training good profile')
                    # figname = channel+'train_trans_good.png'
                    # plt.savefig(figpath/figname)
                    # fignum += 1
                    # plt.figure(dpi=300)
                    # plot_trans_level(Y_train[bias_rank[-1],:],trans_preds_train[bias_rank[-1],:],tao_rttov_train[bias_rank[-1],:],p,'predicted transmittane','training bad profile')
                    # figname = channel+'train_trans_bad.png'
                    # plt.savefig(figpath/figname)

                    ################################# test BT########################
                    tlevel_test = X_test[:,:nlv]
                    # taosrf_test = tao_rttov_test[:,-1] #trans_preds_test[:,-1]
                    bt_preds_test,btair_preds_test = bt_GBT(trans_preds_test[:,:],tlevel_test,emis_test,tsrf_test,tskin_test,taosrf_test,
                                                            ch_info[channel]['scale'],ch_info[channel]['offset'],ch_info[channel]['cv'])

                    bt_test_ytest,btair_test_ytest = bt_GBT(Y_test[:,:],tlevel_test,emis_test,tsrf_test,tskin_test,taosrf_test,
                                               ch_info[channel]['scale'],ch_info[channel]['offset'],ch_info[channel]['cv'])
                    _,btair_test_rttov = bt_GBT(tao_rttov_test[:,:],tlevel_test,emis_test,tsrf_test,tskin_test,taosrf_test,
                                               ch_info[channel]['scale'],ch_info[channel]['offset'],ch_info[channel]['cv'])

                    # 画亮温曲线图
                    bias_pred_test = bt_preds_test-bt_true_test
                    bias_rtt_test = bt_rttov_test-bt_true_test
                    # index = np.where(np.abs(bias_pred_train)>=0.4)[0]
                    # print('The index of training profiles whose bias are greater than 0.4:',index)
                    x = np.arange(len(bias_pred_test))
                    landmask = X_test[:,305]==0
                    seamask = X_test[:,305]==1
                    xland = np.arange(sum(landmask))
                    xsea = np.arange(sum(seamask))

                    fignum += 1
                    ax = np.concatenate((btair_preds_test[landmask]-btair_test_ytest[landmask],btair_test_rttov[landmask]-btair_test_ytest[landmask],
                                         btair_preds_test[seamask]-btair_test_ytest[seamask],btair_test_rttov[seamask]-btair_test_ytest[seamask]))
                    ymin = np.min(ax)
                    ymax = np.max(ax)
                    plt.figure(fignum,dpi=300)
                    plt.subplot(121)
                    plot_bias_bt(xland,btair_preds_test[landmask]-btair_test_ytest[landmask],btair_test_rttov[landmask]-btair_test_ytest[landmask],
                                 title+' bias of BT_air(land)',method)
                    plt.ylabel('Deviation of BT_air(K)',fontsize=12)
                    plt.ylim([ymin,ymax])
                    plt.subplot(122)
                    plot_bias_bt(xsea,btair_preds_test[seamask]-btair_test_ytest[seamask],btair_test_rttov[seamask]-btair_test_ytest[seamask],
                                 title+' bias of BT_air(ocean)',method)
                    plt.yticks([])
                    plt.ylim([ymin,ymax])
                    figname = 'BTair_bias_test'+channel+'.png'
                    plt.savefig(figpath/figname)

                    # index = np.where(np.abs(bias_pred_test)>=0.4)[0]
                    # print('The index of testing profiles whose bias are greater than 0.4:',index)
                    x = np.arange(len(bias_pred_test))
                    # print(x.shape,bias_pred_test.shape,bias_rtt_test.shape)
                    fignum += 1
                    ax = np.concatenate((bias_pred_test[landmask],bias_rtt_test[landmask],bias_pred_test[seamask],bias_rtt_test[seamask]))
                    ymin = np.min(ax)
                    ymax = np.max(ax)
                    plt.figure(fignum,dpi=300)
                    plt.subplot(121)
                    plot_bias_bt(xland,bias_pred_test[landmask],bias_rtt_test[landmask],title+' land',method)
                    plt.ylabel('Deviation of BT(K)',fontsize=12)
                    plt.ylim([ymin,ymax])
                    plt.subplot(122)
                    plot_bias_bt(xsea,bias_pred_test[seamask],bias_rtt_test[seamask],title+' ocean',method)
                    plt.yticks([])
                    plt.ylim([ymin,ymax])
                    figname = 'BT_bias_test'+channel+'.png'
                    plt.savefig(figpath/figname)
            ##################################### independent profiles #################################
                    # taosrf = tao_rttov2[:,j,-1] #trans_preds_test[:,-1]
                    # bt_preds,btair_preds = bt_GBT(trans_preds_independent,XX[:,:nlv],emiss2[:,j],XX[:,303],XX[:,304],taosrf,
                    #                                         ch_info[channel]['scale'],ch_info[channel]['offset'],ch_info[channel]['cv'])
                    #
                    # bias_pred = bt_preds-bt_true2[:,j]
                    # bias_rtt = bt_rttov2[:,j]-bt_true2[:,j]
                    # x = np.arange(len(bias_pred))
                    # fignum += 1
                    # plt.figure(fignum,dpi=300)
                    # plot_bias_bt(x,bias_pred,bias_rtt,title)
                    # figname = 'BT_bias_independent'+channel+'.png'
                    # plt.savefig(figpath/figname)

        ###############################################################################################################
                    # bias_rank = np.argsort(np.abs(bias_pred_test))
                    # fignum += 1
                    # plt.figure(dpi=300)
                    # plot_single_var(X_test[bias_rank[0],:nlv],p,'temperature','t testing good profile')
                    # figname = channel+'test_t_good.png'
                    # plt.savefig(figpath/figname)
                    # fignum += 1
                    # print('The index of testing bad profile: %d' % bias_rank[-1])
                    # plt.figure(dpi=300)
                    # plot_single_var(X_test[bias_rank[-1],:nlv],p,'temperature','t testing bad profile')
                    # figname = channel+'test_t_bad.png'
                    # plt.savefig(figpath/figname)
                    # fignum += 1
                    # plt.figure(dpi=300)
                    # plot_single_var(X_test[bias_rank[0],nlv:nlv*2],p,'water vapor','q testing good profile')
                    # figname = channel+'test_q_good.png'
                    # plt.savefig(figpath/figname)
                    # fignum += 1
                    # plt.figure(dpi=300)
                    # plot_single_var(X_test[bias_rank[-1],nlv:nlv*2],p,'water vapor','q testing bad profile')
                    # figname = channel+'test_q_bad.png'
                    # plt.savefig(figpath/figname)
                    #
                    # fignum += 1
                    # plt.figure(dpi=300)
                    # plot_trans_level(Y_test[bias_rank[0],:],trans_preds_test[bias_rank[0],:],tao_rttov_test[bias_rank[0],:],p,'predicted transmittance','testing good profile')
                    # figname = channel+'test_trans_good.png'
                    # plt.savefig(figpath/figname)
                    # fignum += 1
                    # plt.figure(dpi=300)
                    # plot_trans_level(Y_test[bias_rank[-1],:],trans_preds_test[bias_rank[-1],:],tao_rttov_test[bias_rank[-1],:],p,'predicted transmittance','testing bad profile')
                    # figname = channel+'test_t_bad.png'
                    # plt.savefig(figpath/figname)
        ##########################################################################################

                    if np.isnan(bt_preds_test).any():
                        print('There is wrong BT value')
                        continue
                    mae_gbt = mean_absolute_error(bt_true_test,bt_preds_test)
                    mae_rtt = mean_absolute_error(bt_true_test,bt_rttov_test)
                    # mae_gbt = np.mean(bt_true_test-bt_preds_test)
                    # mae_rtt = np.mean(bt_true_test-bt_rttov_test)
                    rmse_gbt = mean_squared_error(bt_true_test,bt_preds_test)**0.5
                    rmse_rtt = mean_squared_error(bt_true_test,bt_rttov_test)**0.5
                    std_gbt = np.std(bt_preds_test)
                    std_rtt = np.std(bt_rttov_test)
                    # bt_gbtlog = 'test: GBT results:mae={:.4f}, rmse={:.4f}'.format(mae_gbt,rmse_gbt)
                    # bt_rttovlog = 'test: RTTOV results:mae={:.4f}, rmse={:.4f}'.format(mae_rtt,rmse_rtt)
                    bt_gbtlog = '{:.4f} {:.4f} {:.4f}'.format(mae_gbt,rmse_gbt,std_gbt)
                    bt_rttovlog = '{:.4f} {:.4f} {:.4f}'.format(mae_rtt,rmse_rtt,std_rtt)

        ############################# independent profiles ########################################
                    # print(bt_preds)
                    # mae_gbt2 = mean_absolute_error(bt_true2[:,j],bt_preds)
                    # mae_rtt2 = mean_absolute_error(bt_true2[:,j],bt_rttov2[:,j])
                    # rmse_gbt2 = mean_squared_error(bt_true2[:,j],bt_preds)**0.5
                    # rmse_rtt2 = mean_squared_error(bt_true2[:,j],bt_rttov2[:,j])**0.5
                    # bt_gbtlog2 = 'test: GBT results:mae={:.4f}, rmse={:.4f}'.format(mae_gbt2,rmse_gbt2)
                    # bt_rttovlog2 = 'test: RTTOV results:mae={:.4f}, rmse={:.4f}'.format(mae_rtt2,rmse_rtt2)
                    ################################################################################################
                    bt_train[:,j*3:j*3+3] = np.concatenate((bt_preds_train.reshape(-1,1),bt_rttov_train.reshape(-1,1),bt_true_train.reshape(-1,1)),axis=1)
                    bt_test[:,j*3:j*3+3] = np.concatenate((bt_preds_test.reshape(-1,1),bt_rttov_test.reshape(-1,1),bt_true_test.reshape(-1,1)),axis=1)
                    ############################### Save results to specific files ################################################
                    # writer = pd.ExcelWriter(figpath/'feature_ranks.xlsx')
                    # df = pd.DataFrame(ranks,dtype=float)
                    # df.to_excel(writer)
                    # writer.save()
                    # writer.close()
                    flog.write(bt_gbtlog)
                    flog.write('\n')
                    flog.write(bt_rttovlog)
                    flog.write('\n')

            tlog.close()
    writer = pd.ExcelWriter(Path(r'G:\DL_transmitance\revised datasets\final')/'bt_results_xgboost.xlsx')
    df = pd.DataFrame(bt_train,dtype=float)
    df.to_excel(writer,sheet_name='bt_train')
    df = pd.DataFrame(bt_test,dtype=float)
    df.to_excel(writer,sheet_name='bt_test')
    df = pd.DataFrame(lonlat_train,dtype=float)
    df.to_excel(writer,sheet_name='lonlat_train')
    df = pd.DataFrame(lonlat_test,dtype=float)
    df.to_excel(writer,sheet_name='lonlat_test')
    writer.save()
    writer.close()


if __name__=="__main__":
    main()
