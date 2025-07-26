# -*- coding: utf-8 -*-
"""
@ Time: 2020-1-18
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
from sklearn.metrics import mean_absolute_error,mean_squared_log_error,mean_squared_error
from sklearn.preprocessing import PowerTransformer,MinMaxScaler
import joblib

from plot_map_funcs import plot_bt,plot_bias_bt,plot_error_level,plot_mre,scatter_line_trans,plot_trans_level,plot_single_var,bar_feature_rank
from calc_BT import bt_GBT
from LUT import ch_info,feature_names101,trans_limit

import my_funcs as mf

def svr_reg():
#        params = {"C":[3.,4.,4.5,5.,8,10]}
    reg = SVR(C=5,epsilon=0.005,
              kernel='rbf',gamma=1,
              max_iter=-1,shrinking=True,
              tol=0.0001)

    model = reg
    # model = MultiOutputRegressor(reg,n_jobs=-1)
#        reg_multi = RandomizedSearchCV(estimator=MultiOutputRegressor(reg),param_distributions=params,n_iter=10,
#                                       cv=5,n_jobs=-1,refit=True,return_train_score=True)
    return model

def rf_reg():
#        param_list = {"n_estimators":[80,100,150],
#                  "max_depth":[3,5,10],
#                  "bootstrap":[True,False]}
    model = RandomForestRegressor(n_estimators=300,max_depth=6,
                                random_state=2020,criterion='mse')
#        reg_multi = RandomizedSearchCV(clf,param_distributions=param_list,n_iter=10,cv=5)
    return model

def gbt_reg(loss):
    reg = GradientBoostingRegressor(n_estimators=250,subsample=0.7,
                                    learning_rate=0.045,
                                    max_depth = 3,
                                    n_iter_no_change=500,
                                    random_state=2019,loss=loss,
                                    validation_fraction=0.1,
                                    max_features=250)
    # model = MultiOutputRegressor(reg,n_jobs=-1)
    model = reg
    return model

def xgb_reg():
    #用sklearn的API
    bst = xgb.XGBRegressor(objective="reg:squarederror",
                            max_depth=3,
                            learning_rate=0.05,
                            n_estimators=200,
                            verbosity=0,
                            booster='gbtree',
                            gamma=0.001,
                            n_jobs=-1,
                            reg_alpha=0.01,reg_lambda=0.01,random_state=1)
    model = bst
    # model = MultiOutputRegressor(bst,n_jobs=-1)
#        reg_multi.fit(self.trainx,self.trainy,eval_set=[(self.trainx,self.trainy),(self.testx,self.testy)],eval_metric='mae')
##      train_preds = bst.predict(trainx)
##      test_preds = bst.predict(testx)
#       xgb.plot_importance(bst)
#       xgb.plot_tree(bst,num_trees=3)
#       xgb.to_graphviz(bst,num_trees=3)
    return model

def mlp_reg():
    model = MLPRegressor(hidden_layer_sizes=(330,300),
                         activation='tanh',solver='adam',
                         alpha=0.01,
                        batch_size=100,
                         learning_rate='adaptive',
                         learning_rate_init=0.01,
                        shuffle=False,max_iter=10000,
                         tol=1e-7,verbose=False,early_stopping=False,
                         validation_fraction=0.25)
    return model

def training(estimator,trainx,trainy,testx,testy):
    t1 = time.perf_counter()
    estimator.fit(trainx,trainy)
    t2 = time.perf_counter()-t1
    # print('Fitting GBT costs %.4f seconds.' % t2)
    train_preds = estimator.predict(trainx)
    t1 = time.perf_counter()

    test_preds = estimator.predict(testx)
    t2 = time.perf_counter()-t1

    # train_preds = np.exp(-train_preds)
    # test_preds = np.exp(-test_preds)
    # print('Testing GBT costs %.4f seconds.' % t2)
    # print('min_testy=%.4f,min_testpred=%.4f,max_testy=%.4f,max_testpred=%.4f' % (np.min(testy),np.min(test_preds),np.max(testy),np.max(test_preds)))
    train_preds[train_preds<0] = trans_limit
    test_preds[test_preds<0] = trans_limit
    # feature_rank = estimator.feature_importances_

    train_abe = mean_absolute_error(trainy,train_preds)
    test_abe = mean_absolute_error(testy,test_preds)
    train_rmse = mean_squared_error(trainy,train_preds)**0.5
    test_rmse = mean_squared_error(testy,test_preds)**0.5
    train_msle = 1.0E04*mean_squared_log_error(1+trainy,1+train_preds)
    test_msle = 1.0E04*mean_squared_log_error(1+testy,1+test_preds)
    return train_preds,test_preds,train_abe,test_abe,train_rmse,test_rmse,train_msle,test_msle

def cv(estimator,x_train,y_train):
    cv = cross_val_score(estimator,
                         x_train,
                         y_train,
                         scoring='neg_mean_squared_error',
                         n_jobs=-1,
                         cv = 5)
    return cv

def params_random_search(estimator,param_grid):
    # rsearch = RandomizedSearchCV(estimator,param_grid,
    #                             scoring='neg_mean_squared_error',
    #                             n_jobs=-1,cv=4)
    rsearch = GridSearchCV(estimator,param_grid,
                                scoring='neg_mean_squared_error',
                                n_jobs=-1,cv=4)
    # print("Best score: {:.4f}".format(rsearch.best_score_))
    best_params = rsearch.best_estimator_.get_paramds()
    print('Best parameters:')
    for param_name in sorted(best_params.keys()):
        print("\t{}: {}".format(param_name,best_params[param_name]))

def loadata(datafile):
    # 选择所需通道的数据做训练
    with h5py.File(datafile,'r') as f:
        X = f['X'][:-1,:]
        Y = f['Y'][:-1,:]
        tao_rttov = f['iras_rttov'][:-1,:]

    return X,Y,tao_rttov

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
    xfile = Path(r'G:\DL_transmitance\revised datasets\dataset_101L_ifsec.HDF')
    # femis = Path(r'G:\DL_transmitance\revised datasets\dataset_101L_Y_0-13.HDF')
    # femis2 = Path(r'G:\DL_transmitance\revised datasets\dataset_101L_Y_after13.HDF')
    fp = Path(r'G:\DL_transmitance\profiles\model_level_definition.xlsx')
    nlv = 101
    chs = [11,12,13]
    fignum = 1
    method = 'SVR'
    if method=='SVR':
        model = svr_reg()
    elif method=='RF':
        model = rf_reg()
    elif method=='GBT':
        loss = 'ls'
        model = gbt_reg(loss)
        # param_grid = {'learning_rate': [0.1,0.01],
        #               'max_depth': [2,3]}
    elif method=='XGBoost':
        model = xgb_reg()
    elif method=='MLP':
        model = mlp_reg()
    figpath1 = Path(r'G:\DL_transmitance\figures\101L\reg-results\ifsec')/method
    ######### load the emissivity and surface2space transmittance etc. ################
    # with h5py.File(xfile,'r') as f:
    #     X = f['dependent/X'][:,:]
    #     Y = f['dependent/Y'][:,:,:-1]
    #     tao_rttov = f['dependent/transmission_RTTOV'][:,:,:-1]
    #     bt_rttov = f['dependent/BT_RTTOV'][:]
    #     bt_true = f['dependent/BT_true'][:]
    #     emiss = f['dependent/emissivity'][:]
    #     new_preds = f['dependent/predictor_RTTOV'][:]
    with h5py.File(xfile,'r') as f:
        X = f['X'][:,:]
        Y = f['Y'][:,:,:-1]
        tao_rttov = f['transmission_RTTOV'][:,:,:-1]
        bt_rttov = f['BT_RTTOV'][:]
        bt_true = f['BT_true'][:]
        emiss = f['emissivity'][:]
        new_preds = f['predictor_RTTOV'][:]
    # kick_index = np.where(X[:,100]<280)[0]
    # X = np.delete(X,kick_index,axis=0)
    # Y = np.delete(Y,kick_index,axis=1)
    # emiss = np.delete(emiss,kick_index,axis=0)
    # tao_rttov = np.delete(tao_rttov,kick_index,axis=0)
    # bt_rttov = np.delete(bt_rttov,kick_index,axis=0)
    # bt_true = np.delete(bt_true,kick_index,axis=0)
    # new_preds = np.delete(new_preds,kick_index,axis=0)
    df = pd.read_excel(fp,sheet_name='101L')
    p = np.array(df['ph[hPa]'].values,dtype=np.float)
    # with h5py.File(femis,'r') as f:
    #     emiss = f['emissivity'][:]
    #     # taosrf = f['transmission_RTTOV'][:,11,-1]
    #     bt_true = f['BT_true'][:]
    #     bt_rttov = f['BT_RTTOV'][:]
    #     Y = f['Y'][:]
    #     tao_rttov = f['transmission_RTTOV'][:]
    #
    # with h5py.File(femis2,'r') as f:
    #     emiss2 = f['emissivity'][:]
    #     # taosrf = f['transmission_RTTOV'][:,11,-1]
    #     bt_true2 = f['BT_true'][:]
    #     bt_rttov2 = f['BT_RTTOV'][:]
    #     Y2 = f['Y'][:]
    #     tao_rttov2 = f['transmission_RTTOV'][:]

    # bt_true = np.concatenate((bt_true,bt_true2),axis=0)
    # bt_rttov = np.concatenate((bt_rttov,bt_rttov2),axis=0)
    # Y = np.concatenate((Y,Y2),axis=1)
    # tao_rttov = np.concatenate((tao_rttov,tao_rttov2),axis=0)
    X = np.delete(X,[303,305,306,308,309,310,311],axis=1) # kick out 2m variables and longitude, latitude
    # X = np.delete(X,np.arange(202,303),axis=1)
    # bad_index = []
    # for k in range(n):
    #     if (X[k,nlv*2:nlv*3]<10**-6).any():
    #         bad_index.append(k)
    #     else:
    #         continue
    #
    # X = np.delete(X,bad_index,axis=0)

    npro = X.shape[0]
    # random.seed(2020)
    print('There are %d profiles' % npro)
################################# only ocean ##############################3
    # rr = np.where(X[:,548]==0)[0]
    # emiss = emiss[rr]
    # taosrf = taosrf[rr]
    #
    # bt_rttov = bt_rttov[rr]
    # tsrf = tsrf[rr]
    # X = X[rr]
    # tao_rttov = tao_rttov[rr]
    # Y = Y[rr]
    # bt_true = bt_true[rr]
    ############ Calculate new predictors and concatenate them to the current X ############
    # X = np.concatenate((X,new_preds),axis=1)
    random.seed(2020)
    seeds = random.sample(range(2020),2)
    # loss = 'ls'
    ###################################################################################
    for j,ch in enumerate(chs):
        channel = 'ch'+str(ch)
        title = ch_info[channel]['cwl']
        for s in seeds:
            X_train,X_test,Y_train,Y_test = train_test_split(X,Y[j,:,:],test_size=0.2,random_state=s)
            tsrf_train,tsrf_test = X_train[:,303],X_test[:,303]
            tskin_train,tskin_test = X_train[:,304],X_test[:,304]
            _,_,emis_train,emis_test = train_test_split(X,emiss[:,j],test_size=0.2,random_state=s)
            tao_rttov_train,tao_rttov_test,bt_rttov_train,bt_rttov_test = train_test_split(tao_rttov[:,j,:],
                                                                                           bt_rttov[:,j],test_size=0.2,random_state=s)
            _,_,bt_true_train,bt_true_test = train_test_split(X,bt_true[:,j],test_size=0.2,random_state=s)

            trans_preds_train = np.ones((X_train.shape[0],nlv),dtype=np.float) ## save every laye's predicted transmittance to calculate BT
            trans_preds_test = np.ones((X_test.shape[0],nlv),dtype=np.float) ## same as above
            figpath = figpath1/str(s)
            if figpath.exists():
                pass
            else:
                figpath.mkdir(parents=True)

            logname = method+'_'+channel+'.txt'
            flog = open(figpath/logname,'w')       ### save the log of every layer's and channel's training
            ############### Editted on 2019-1-2, do layer-by-layer regression, for feature selection ###############################
            for ii in range(nlv):
                trainy = Y_train[:,ii]
                testy = Y_test[:,ii]
                # if ii>80:
                #     loss = 'ls'
                # else:
                #     loss = 'ls'
                # model = gbt_reg(loss)
                if ii>0:
                    newtrain,newtest,_,_ = train_test_split(new_preds[:,:,ii-1],emiss[:,j],test_size=0.2,random_state=s)
                    trainx = np.concatenate((X_train,newtrain),axis=1)
                    testx = np.concatenate((X_test,newtest),axis=1)
                else:
                    trainx = X_train
                    testx = X_test

                # scaler = PowerTransformer(method='yeo-johnson',standardize=True).fit(trainx)
                # scaler = MinMaxScaler().fit(trainx)
                # trainx = scaler.transform(trainx)
                # testx = scaler.transform(testx)
                train_preds,test_preds,train_mae,test_mae,train_rmse,test_rmse,train_msle,test_msle = training(model,trainx,trainy,testx,testy)
                trans_preds_test[:,ii] = test_preds
                trans_preds_train[:,ii] = train_preds

                # bias_rttov_test = mean_absolute_error(testy, tao_rttov_test[:,ii])
                # rmse_rttov_test = mean_squared_error(testy, tao_rttov_test[:,ii]) ** 0.5
                # msle_rttov_test = 1.0E04 * mean_squared_log_error(1+testy, 1+tao_rttov_test[:,ii])
                #
                # bias_rttov_train = mean_absolute_error(trainy, tao_rttov_train[:,ii])
                # rmse_rttov_train = mean_squared_error(trainy, tao_rttov_train[:,ii]) ** 0.5
                # msle_rttov_train = 1.0E04 * mean_squared_log_error(1 + trainy, 1 + tao_rttov_train[:,ii])
                #
                # trainlog = 'training mae={:.4f}, rmse={:.4f}, msle={:.4f}'.format(train_mae,train_rmse,train_msle)
                # testlog = 'testing mae={:.4f}, rmse={:.4f}, msle={:.4f}'.format(test_mae,test_rmse,test_msle)
                # rttovtrainlog = 'train: RTTOV mae={:.4f}, rmse={:.4f}, msle={:.4f}'.format(bias_rttov_train,rmse_rttov_train,msle_rttov_train)
                # rttovtestlog = 'test: RTTOV mae={:.4f}, rmse={:.4f}, msle={:.4f}'.format(bias_rttov_test,rmse_rttov_test,msle_rttov_test)
                # flog.write(trainlog)
                # flog.write('\n')
                # flog.write(testlog)
                # flog.write('\n')
                # flog.write(rttovtrainlog)
                # flog.write('\n')
                # flog.write(rttovtestlog)
                # flog.write('\n')
                #
                # model_name = 'v7_137L_'+channel+'_sample'+str(npro)+'.m'
                # joblib.dump(model,Path(r'G:/DL_transmitance/models')/model_name)
                # train_preds = train_preds*train_scale+train_mean
                # test_preds = test_preds*train_scale+train_mean
                ################# compute and plot the error: bias, RMSE, MRE ##################################

                ########### Plot feature ranking(top15) ####################
                # if ii>48:
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

            ############################## scatter pictures ##################################
            bias_levs_train, rmse_levs_train, mre_train = calc_err(trans_preds_train,Y_train)
            bias_levs_test, rmse_levs_test, mre_test = calc_err(trans_preds_test,Y_test)
            bias_levs_train_rttov,rmse_levs_train_rttov,mre_train_rttov = calc_err(tao_rttov_train,Y_train)
            bias_levs_test_rttov,rmse_levs_test_rttov,mre_test_rttov = calc_err(tao_rttov_test,Y_test)
            fignum += 1
            plt.figure(fignum,dpi=300)
            # print(bias_levs_train.shape,rmse_levs_train.shape,bias_levs_train_rttov.shape,rmse_levs_train_rttov.shape,p.shape)
            plot_error_level(bias_levs_train, rmse_levs_train,bias_levs_train_rttov,rmse_levs_train_rttov,p,title)
            figname = channel+method+'_train_bias_nonorm.png'
            plt.savefig(figpath/figname,dpi=300)
            # plt.close()
            fignum += 1
            plt.figure(fignum,dpi=300)
            plot_mre(mre_train,mre_train_rttov,title)
            figname = channel+method+'_train_mre_nonorm.png'
            plt.savefig(figpath/figname)
            # plt.close()
            fignum += 1
            plt.figure(fignum,dpi=300)
            plot_error_level(bias_levs_test[:], rmse_levs_test[:],bias_levs_test_rttov[:],rmse_levs_test_rttov[:],p,title)
            figname = channel+method+'_test_bias_nonorm.png'
            plt.savefig(figpath/figname)
            # plt.close()
            fignum += 1
            plt.figure(fignum,dpi=300)
            plot_mre(mre_test,mre_test_rttov,title)
            figname = channel+method+'_test_mre_nonorm.png'
            plt.savefig(figpath/figname)
            train_mae = mean_absolute_error(Y_train,trans_preds_train)
            test_mae = mean_absolute_error(Y_test,trans_preds_test)
            train_rmse = mean_squared_error(Y_train,trans_preds_train)**0.5
            test_rmse = mean_squared_error(Y_test,trans_preds_test)**0.5
            train_msle = 1.0E04*mean_squared_log_error(1+Y_train,1+trans_preds_train)
            test_msle = 1.0E04*mean_squared_log_error(1+Y_test,1+trans_preds_test)

            pro_test_mae = np.sum(trans_preds_test-Y_test,axis=1)
            pro_test_rmse = np.sum((trans_preds_test-Y_test)**2,axis=1)
            # print('The index of maximum mae profile:',np.argsort(-pro_test_mae),np.argsort(-pro_test_rmse))

            bias_rttov_test = mean_absolute_error(Y_test, tao_rttov_test)
            rmse_rttov_test = mean_squared_error(Y_test, tao_rttov_test) ** 0.5
            msle_rttov_test = 1.0E04 * mean_squared_log_error(1+Y_test, 1+tao_rttov_test)

            bias_rttov_train = mean_absolute_error(Y_train, tao_rttov_train)
            rmse_rttov_train = mean_squared_error(Y_train, tao_rttov_train) ** 0.5
            msle_rttov_train = 1.0E04 * mean_squared_log_error(1 + Y_train, 1 + tao_rttov_train)

            fignum += 1
            plt.figure(fignum,dpi=300)
            scatter_line_trans(Y_train,trans_preds_train,tao_rttov_train,title,train_mae,train_rmse,train_msle,bias_rttov_train,rmse_rttov_train,msle_rttov_train)
            figname = channel+method+'_train_trans_all'+'.png'
            plt.savefig(figpath/figname)
            # plt.close()
            fignum += 1
            plt.figure(fignum,dpi=300)
            scatter_line_trans(Y_test,trans_preds_test,tao_rttov_test,title,test_mae,test_rmse,test_msle,bias_rttov_test,rmse_rttov_test,msle_rttov_test)
            figname = channel+method+'_test_trans_all'+'.png'
            plt.savefig(figpath/figname)
            plt.close()

            ################# Calculating BT with RT function #################################################

            ############################################################################################
            tlevel_train = X_train[:,:nlv]
            taosrf_train = tao_rttov_train[:,-1] #trans_preds_train[:,-1]

            bt_preds_train,btair_preds_train = bt_GBT(trans_preds_train[:,:],tlevel_train,emis_train,tsrf_train,tskin_train,taosrf_train,
                                    ch_info[channel]['scale'],ch_info[channel]['offset'],ch_info[channel]['cv'])
            # bt_train_ytrain,btair_train_ytrain = bt_GBT(ytrain[:,:-1],tlevel_train,emis_train,tsrf_train,ytrain[:,-1],
            #                            ch_info[channel]['scale'],ch_info[channel]['offset'],ch_info[channel]['cv'])
            # _,btair_train_rttov = bt_GBT(taorttovtrain[:,:-1],tlevel_train,emis_train,tsrf_train,taorttovtrain[:,-1],
            #                            ch_info[channel]['scale'],ch_info[channel]['offset'],ch_info[channel]['cv'])
            # 画亮温曲线图
            bias_pred_train = bt_preds_train-bt_true_train
            bias_rtt_train = bt_rttov_train-bt_true_train

            index = np.where(np.abs(bias_pred_train)>=0.4)[0]
            # print('The index of training profiles whose bias are greater than 0.4:',index)
            x = np.arange(len(bias_pred_train))

            # fignum += 1
            # plt.figure(fignum,dpi=300)
            # plot_bias_bt(x,btair_preds_train-btair_train_ytrain,btair_train_rttov-btair_train_ytrain,'bias of BT_air')
            # figname = 'BTair_bias_train'+channel+'.png'
            # plt.savefig(figpath/figname)
            #
            fignum += 1
            plt.figure(fignum,dpi=300)
            plot_bias_bt(x,bias_pred_train,bias_rtt_train,title)
            figname = 'BT_bias_train'+channel+'.png'
            plt.savefig(figpath/figname)

            mae_gbt = mean_absolute_error(bt_true_train,bt_preds_train)
            mae_rtt = mean_absolute_error(bt_true_train,bt_rttov_train)
            rmse_gbt = mean_squared_error(bt_true_train,bt_preds_train)**0.5
            rmse_rtt = mean_squared_error(bt_true_train,bt_rttov_train)**0.5
            bt_gbtlog = 'train: GBT results:mae={:.4f}, rmse={:.4f}'.format(mae_gbt,rmse_gbt)
            bt_rttovlog = 'train: RTTOV results:mae={:.4f}, rmse={:.4f}'.format(mae_rtt,rmse_rtt)
            flog.write(bt_gbtlog)
            flog.write('\n')
            flog.write(bt_rttovlog)
############################################## Plot good and bad performance profile ###########
            bias_rank = np.argsort(np.abs(bias_pred_train))
            # fignum += 1
            # plt.figure(dpi=300)
            # plot_single_var(X_train[bias_rank[0],:nlv],p,'temperature','t training good profile')
            # figname = channel+'train_t_good.png'
            # plt.savefig(figpath/figname)
            # fignum += 1
            # plt.figure(dpi=300)
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

            fignum += 1
            plt.figure(dpi=300)
            plot_single_var(trans_preds_train[bias_rank[0],:],p,'predicted transmittane','training good profile')
            figname = channel+'train_trans_good.png'
            plt.savefig(figpath/figname)
            fignum += 1
            plt.figure(dpi=300)
            plot_single_var(trans_preds_train[bias_rank[-1],:],p,'predicted transmittane','training bad profile')
            figname = channel+'train_trans_bad.png'
            plt.savefig(figpath/figname)

            ################################# test BT########################
            tlevel_test = X_test[:,:nlv]
            taosrf_test = tao_rttov_test[:,-1] #trans_preds_test[:,-1]
            bt_preds_test,btair_preds_test = bt_GBT(trans_preds_test[:,:],tlevel_test,emis_test,tsrf_test,tskin_test,taosrf_test,
                                                    ch_info[channel]['scale'],ch_info[channel]['offset'],ch_info[channel]['cv'])

            # 画亮温曲线图
            bias_pred_test = bt_preds_test-bt_true_test
            bias_rtt_test = bt_rttov_test-bt_true_test

            index = np.where(np.abs(bias_pred_test)>=0.4)[0]
            # print('The index of testing profiles whose bias are greater than 0.4:',index)
            x = np.arange(len(bias_pred_test))
            print(x.shape,bias_pred_test.shape,bias_rtt_test.shape)
            fignum += 1
            plt.figure(fignum,dpi=300)
            plot_bias_bt(x,bias_pred_test,bias_rtt_test,title)
            figname = 'BT_bias_test'+channel+'.png'
            plt.savefig(figpath/figname)
            plt.close('all')
            plt.show()

###############################################################################################################
            bias_rank = np.argsort(np.abs(bias_pred_test))
            # fignum += 1
            # plt.figure(dpi=300)
            # plot_single_var(X_train[bias_rank[0],:nlv],p,'temperature','t testing good profile')
            # figname = channel+'test_t_good.png'
            # plt.savefig(figpath/figname)
            # fignum += 1
            # plt.figure(dpi=300)
            # plot_single_var(X_train[bias_rank[-1],:nlv],p,'temperature','t testing bad profile')
            # figname = channel+'test_t_bad.png'
            # plt.savefig(figpath/figname)
            # fignum += 1
            # plt.figure(dpi=300)
            # plot_single_var(X_train[bias_rank[0],nlv:nlv*2],p,'water vapor','q testing good profile')
            # figname = channel+'test_q_good.png'
            # plt.savefig(figpath/figname)
            # fignum += 1
            # plt.figure(dpi=300)
            # plot_single_var(X_train[bias_rank[-1],nlv:nlv*2],p,'water vapor','q testing bad profile')
            # figname = channel+'test_q_bad.png'
            # plt.savefig(figpath/figname)

            fignum += 1
            plt.figure(dpi=300)
            plot_single_var(trans_preds_test[bias_rank[0],:],p,'predicted transmittance','testing good profile')
            figname = channel+'test_trans_good.png'
            plt.savefig(figpath/figname)
            fignum += 1
            plt.figure(dpi=300)
            plot_single_var(trans_preds_test[bias_rank[-1],:],p,'predicted transmittance','testing bad profile')
            figname = channel+'test_t_bad.png'
            plt.savefig(figpath/figname)
##########################################################################################

            if np.isnan(bt_preds_test).any():
                print('There is wrong BT value')
                continue
            mae_gbt = mean_absolute_error(bt_true_test,bt_preds_test)
            mae_rtt = mean_absolute_error(bt_true_test,bt_rttov_test)
            rmse_gbt = mean_squared_error(bt_true_test,bt_preds_test)**0.5
            rmse_rtt = mean_squared_error(bt_true_test,bt_rttov_test)**0.5
            bt_gbtlog = 'test: GBT results:mae={:.4f}, rmse={:.4f}'.format(mae_gbt,rmse_gbt)
            bt_rttovlog = 'test: RTTOV results:mae={:.4f}, rmse={:.4f}'.format(mae_rtt,rmse_rtt)
            ################################################################################################
            # print(np.argsort(-trans_preds_test[2,:]))
            # print(np.argsort(-Y_test[2,:]))
            ############################### Save results to specific files ################################################
            # writer = pd.ExcelWriter(figpath/'feature_ranks.xlsx')
            # df = pd.DataFrame(ranks,dtype=float)
            # df.to_excel(writer)
            # writer.save()
            # writer.close()
            flog.write(bt_gbtlog)
            flog.write('\n')
            flog.write(bt_rttovlog)
            flog.close()


if __name__=="__main__":
    main()
