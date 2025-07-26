# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 19:16:33 2019
用不同模型训练用ISF91廓线得到的透过率
@author: LiangHongli
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
from sklearn.preprocessing import PowerTransformer,StandardScaler,MinMaxScaler
import joblib

from plot_map_funcs import plot_bt,plot_bias_bt,plot_error_level,plot_mre,scatter_line_trans,plot_trans_level,bar_feature_rank
from calc_BT import bt_GBT
from LUT import ch_info,feature_names91,trans_limit
from calc_new_predictor import var_star,layer_profile,predictors

def svr_reg():
#        params = {"C":[3.,4.,4.5,5.,8,10]}
    reg = SVR(C=2.,epsilon=0.0001,
              kernel='rbf',gamma=0.0001,
              max_iter=-1,shrinking=True,
              tol=0.0001)

    model = MultiOutputRegressor(reg,n_jobs=-1)
#        reg_multi = RandomizedSearchCV(estimator=MultiOutputRegressor(reg),param_distributions=params,n_iter=10,
#                                       cv=5,n_jobs=-1,refit=True,return_train_score=True)
    return model

def rf_reg():
#        param_list = {"n_estimators":[80,100,150],
#                  "max_depth":[3,5,10],
#                  "bootstrap":[True,False]}
    model = RandomForestRegressor(n_estimators=250,max_depth=7,
                                random_state=0,criterion='mse',n_jobs=-1)
#        reg_multi = RandomizedSearchCV(clf,param_distributions=param_list,n_iter=10,cv=5)
    return model

def gbt_reg():
    reg = GradientBoostingRegressor(n_estimators=126,subsample=0.8,
                                    learning_rate=0.045,
                                    max_depth = 3,
                                    n_iter_no_change=500,
                                    random_state=0,loss='ls',
                                    validation_fraction=0.1)
    # model = MultiOutputRegressor(reg,n_jobs=-1)
    model = reg
    return model

def xgb_reg():
    #用sklearn的API
    bst = xgb.XGBRegressor(objective="reg:squarederror",
                            max_depth=5,
                            learning_rate=0.05,
                            n_estimators=300,
                            verbosity=0,
                            booster='gbtree',
                            gamma=0.001,
                            n_jobs=-1,
                            reg_alpha=0.01,reg_lambda=0.01,random_state=1)
    model = MultiOutputRegressor(bst,n_jobs=-1)
#        reg_multi.fit(self.trainx,self.trainy,eval_set=[(self.trainx,self.trainy),(self.testx,self.testy)],eval_metric='mae')
##      train_preds = bst.predict(trainx)
##      test_preds = bst.predict(testx)
#       xgb.plot_importance(bst)
#       xgb.plot_tree(bst,num_trees=3)
#       xgb.to_graphviz(bst,num_trees=3)
    return model

def mlp_reg():
    model = MLPRegressor(hidden_layer_sizes=(300,200,200),
                         activation='relu',solver='adam',
                         alpha=0.01,
                        batch_size=10,
                         learning_rate='adaptive',
                         learning_rate_init=0.01,
                        shuffle=False,max_iter=10000,
                         tol=1e-7,verbose=True,early_stopping=False,
                         validation_fraction=0.25)
    return model

def training(estimator,trainx,trainy,testx,testy):
    t1 = time.perf_counter()
    estimator.fit(trainx,trainy)
    t2 = time.perf_counter()-t1
    print('Fitting GBT costs %.4f seconds.' % t2)
    train_preds = estimator.predict(trainx)
    t1 = time.perf_counter()

    test_preds = estimator.predict(testx)
    t2 = time.perf_counter()-t1
    print('Testing GBT costs %.4f seconds.' % t2)
    print(np.min(train_preds),np.min(test_preds),np.max(train_preds),np.max(test_preds))
    train_preds[train_preds<0] = trans_limit
    test_preds[test_preds<0] = trans_limit
    feature_rank = estimator.feature_importances_

    train_abe = mean_absolute_error(trainy,train_preds)
    test_abe = mean_absolute_error(testy,test_preds)
    train_rmse = mean_squared_error(trainy,train_preds)**0.5
    test_rmse = mean_squared_error(testy,test_preds)**0.5
    train_msle = 1.0E04*mean_squared_log_error(1+trainy,1+train_preds)
    test_msle = 1.0E04*mean_squared_log_error(1+testy,1+test_preds)
    return train_preds,test_preds,train_abe,test_abe,train_rmse,test_rmse,train_msle,test_msle,feature_rank

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
    mse_levs = np.sum((x1-x2)**2,axis=0)/len(x1[:,0])
    rmse_levs = mse_levs**0.5
    #每一个样本在所有气压层的相对误差均值
    mre = np.mean(np.abs(x1-x2)/x2,axis=1)
    sor_mre = np.where(mre==np.max(mre))[0]
    print('相对误差最大的廓线位置：',sor_mre)
    return bias_levs, rmse_levs, mre

def main():
    datafile = Path(r'G:\DL_transmitance\revised datasets\training_ch12_IFS91l_894.HDF')
    femis = Path(r'G:\DL_transmitance\revised datasets\ifs91_X999_Y894_v9.HDF')
    # datafile = Path(r'G:\DL_transmitance\revised datasets\dataset_IFS137_ch12_ch15.HDF')
    # femis = Path(r'G:\DL_transmitance\revised datasets\dataset_IFS137_ch1_ch10.HDF')
    pfile = Path(r'G:\DL_transmitance\profiles\model_level_definition.xlsx')

    # npro = 999
    # ch = 12
    nlv = 91
    chs = [12]
    fignum = 1
    X,Y,tao_rttov = loadata(datafile)
    # X = X[:,:nlv*4]
    # contribute_layer_index = [39,59,89,99,109,116,-1,-1,37,104,94,131,122]
    ######### load the emissivity and surface2space transmittance etc. ################
    with h5py.File(femis,'r') as f:
        emiss = f['emissivity'][:-1] #[:,11]
        taosrf = f['transmittance_rttov'][:-1,0,-1] #[:,11,-1]
        bt_true = f['BT_true'][:-1]
        bt_rttov = f['BT_rttov'][:-1] #[:,11]
        profiles = f['profiles'][:] #f['X'][:,:-1]
        # tao_rttov = f['transmission_RTTOV'][:,11,:-1]

    profiles = np.delete(profiles,[38,161,216,239,845],axis=0)
    profiles = np.delete(profiles,np.arange(800,900),axis=0)
    tsrf = profiles[:-1,459] # skin temperature
    cc = profiles[:-1,np.arange(364,455)] # cloud fraction of each level
    r = np.where(np.max(cc,axis=1)<0.3)
    r = r[0]
    print(r)
    emiss = emiss[r]
    taosrf = taosrf[r]
    bt_true = bt_true[r]
    bt_rttov = bt_rttov[r]
    tsrf = tsrf[r]
    X = X[r]
    Y = Y[r]
    tao_rttov = tao_rttov[r]

    # rr = np.where(profiles[r,460]-1<0.000001)[0]
    # print(rr)
    # emiss = emiss[rr]
    # taosrf = taosrf[rr]
    # bt_true = bt_true[rr]
    # bt_rttov = bt_rttov[rr]
    # tsrf = tsrf[rr]
    # X = X[rr]
    # Y = Y[rr]
    # tao_rttov = tao_rttov[rr]

    # with h5py.File(datafile,'r') as f:
    #     Y = f['Y'][10,:,:-1]
    #     bt_true = f['BT_true'][:,10]
    # X = np.delete(profiles,[548,549,550,551,553,554,555],axis=1)

    npro = X.shape[0]
    print('There are %d profiles' % npro)
    ntrain = int(npro*0.8)
    random.seed(2019)
    train_index = random.sample(range(npro),ntrain)
    test_index = [x for x in range(npro) if x not in train_index]
    ############ Calculate new predictors and concatenate them to the current X ############
    # vstar = var_star(X[:,nlv:nlv*3],nlv)
    # layerp = layer_profile(X[:,nlv:nlv*3],nlv)
    # new_preds = predictors(vstar,layerp,nlv)
    # X = np.concatenate((X,new_preds),axis=1)

    # X = X[:,54:]
    trainx = X[train_index,:]
    testx = X[test_index,:]

    # outlier_train = [18,37,109,111,122,134,179]
    # outlier_test = [10,9,20,16,12,4,2,13]
    outlier_train = [94,99,102,140]
    outlier_test = [20,13,25,40,22,30,17,18,3,35,7,31,4,6,38,9,24,19,15,42,41,21,14,12,23]
    # outlier_train = [0]
    # outlier_test = [0]
    trainx = np.delete(trainx,outlier_train,axis=0)
    testx = np.delete(testx,outlier_test,axis=0)
    # trainx,testx,trainy,testy = train_test_split(X,Y,test_size=0.1,random_state=2019)
    pdata = pd.read_excel(pfile,sheet_name='91L')
    p = np.array(pdata['ph [hPa]'].values[1:],dtype=np.float)

    method = 'GBT'
    if method=='SVR':
        model = svr_reg()
    elif method=='RF':
        model = rf_reg()
    elif method=='GBT':
        model = gbt_reg()
        # param_grid = {'learning_rate': [0.1,0.01],
        #               'max_depth': [2,3]}
    elif method=='XGBoost':
        model = xgb_reg()
    elif method=='MLP':
        model = mlp_reg()
    ###################################################################################
    for j,ch in enumerate(chs):
        channel = 'ch'+str(ch)
        title = ch_info[channel]['cwl']
        trans_preds_train = np.ones((trainx.shape[0],nlv),dtype=np.float) ## save every laye's predicted transmittance to calculate BT
        trans_preds_test = np.ones((testx.shape[0],nlv),dtype=np.float) ## same as above
        ranks = np.zeros((nlv-1,X.shape[1]),dtype=np.float) ## save feature ranks whose value is more than 0 of each layer

        figpath = Path(r'G:\DL_transmitance\figures\multi_regressors\layer-by-layer-4feature-selection\ch12\clear-sky-ocean-only\for-reply\nestimator80')
        if figpath.exists():
            pass
        else:
            figpath.mkdir(parents=True)

        logname = method+'_'+'_'+channel+'.txt'
        flog = open(figpath/logname,'w')       ### save the log of every layer's and channel's training

        ############### Editted on 2019-1-2, do layer-by-layer regression, for feature selection ###############################
        for ii in range(nlv):
            trainy = Y[train_index,ii]
            testy = Y[test_index,ii]
            trainy = np.delete(trainy,outlier_train)
            testy = np.delete(testy,outlier_test)
            train_preds,test_preds,train_mae,test_mae,train_rmse,test_rmse,train_msle,test_msle,rank = training(model,trainx,trainy,testx,testy)
            trans_preds_test[:,ii] = test_preds
            trans_preds_train[:,ii] = train_preds
            # ranks[ii-1,:] = rank
            #
            tao_rttov_test = np.delete(tao_rttov[test_index,ii],outlier_test)
            bias_rttov_test = mean_absolute_error(testy, tao_rttov_test)
            rmse_rttov_test = mean_squared_error(testy, tao_rttov_test) ** 0.5
            msle_rttov_test = 1.0E04 * mean_squared_log_error(1+testy, 1+tao_rttov_test)

            tao_rttov_train = np.delete(tao_rttov[train_index,ii],outlier_train)
            bias_rttov_train = mean_absolute_error(trainy, tao_rttov_train)
            rmse_rttov_train = mean_squared_error(trainy, tao_rttov_train) ** 0.5
            msle_rttov_train = 1.0E04 * mean_squared_log_error(1 + trainy, 1 + tao_rttov_train)

            trainlog = 'level={:d},training mae={:.4f}, rmse={:.4f}, msle={:.4f}'.format(ii+1,train_mae,train_rmse,train_msle)
            testlog = 'level={:d},testing mae={:.4f}, rmse={:.4f}, msle={:.4f}'.format(ii+1,test_mae,test_rmse,test_msle)
            rttovtrainlog = 'level={:d},train: RTTOV mae={:.4f}, rmse={:.4f}, msle={:.4f}'.format(ii+1,bias_rttov_train,rmse_rttov_train,msle_rttov_train)
            rttovtestlog = 'level={:d},test: RTTOV mae={:.4f}, rmse={:.4f}, msle={:.4f}'.format(ii+1,bias_rttov_test,rmse_rttov_test,msle_rttov_test)
            flog.write(trainlog)
            flog.write('\n')
            flog.write(testlog)
            flog.write('\n')
            flog.write(rttovtrainlog)
            flog.write('\n')
            flog.write(rttovtestlog)
            flog.write('\n')

            # model_name = 'v7_137L_'+channel+'_sample'+str(npro)+'.m'
            # joblib.dump(model,Path(r'G:/DL_transmitance/models')/model_name)
            # train_preds = train_preds*train_scale+train_mean
            # test_preds = test_preds*train_scale+train_mean
            ################# compute and plot the error: bias, RMSE, MRE ##################################

            # 2019/6/17 save the predictions
            # preds = np.zeros((893,91))
            # for i,k in enumerate(train_index):
            #     preds[k,:] = train_preds[i,:]
            #
            # for i,k in enumerate(test_index):
            #     preds[k,:] = test_preds[i,:]

            # chj = f.create_group(name=channel)
            # chj.create_dataset('trans_preds',data=test_preds,compression='gzip')
            # chj.create_dataset('tans_true',data=testy,compression='gzip')

            ########### Plot feature ranking(top15) ####################
            # fignum += 1
            # plt.figure(fignum,dpi=300)
            # index = np.argsort(-rank)
            # first0 = np.where(rank[index]<=0)[0][0]
            # if first0>30:
            #     first0 = 30
            # else:
            #     pass
            # top_rank = rank[index][:first0]
            #
            # features = []
            # # print(index[:first0],len(feature_names137))
            # for s in index[:first0]:
            #     features.append(feature_names91[s])
            # # print(features)
            # bar_feature_rank(top_rank,features,title)
            # figname = channel+'feature_rank_layer'+str(ii+1)+'.png'
            # plt.savefig(figpath/figname)
            # plt.show()

        ################# Calculating BT with RT function #################################################
        tlevel_train = trainx[:,nlv:nlv*2]
        emis_train = emiss[train_index]
        tsrf_train = tsrf[train_index]
        taosrf_train = taosrf[train_index]

        emis_train = np.delete(emis_train,outlier_train)
        taosrf_train = np.delete(taosrf_train,outlier_train)
        tsrf_train = np.delete(tsrf_train,outlier_train)

        bt_preds_train = bt_GBT(trans_preds_train,tlevel_train,emis_train,tsrf_train,taosrf_train,ch_info[channel]['scale'],ch_info[channel]['offset'],ch_info[channel]['cv'])

        # 画亮温曲线图
        bt_true_train = bt_true[train_index]#[valid_index]
        bt_rttov_train = bt_rttov[train_index]#[valid_index]

        bt_true_train = np.delete(bt_true_train,outlier_train)
        bt_rttov_train = np.delete(bt_rttov_train,outlier_train)

        bias_pred_train = bt_preds_train-bt_true_train
        bias_rtt_train = bt_rttov_train-bt_true_train

        index = np.where(np.abs(bias_pred_train)>=0.4)[0]
        print('The index of training profiles whose bias are greater than 0.4:',index)
        x = np.arange(len(bias_pred_train))

        fignum += 1
        plt.figure(fignum,dpi=300)
        plot_bias_bt(x,bias_pred_train,bias_rtt_train,title)
        figname = 'BT_bias_train'+channel+'.png'
        plt.savefig(figpath/figname)
        # plt.close('all')
        #plt.show()
        if np.isnan(bt_preds_train).any():
            print('There is wrong BT value')
            continue
        mae_gbt = mean_absolute_error(bt_true_train,bt_preds_train)
        mae_rtt = mean_absolute_error(bt_true_train,bt_rttov_train)
        rmse_gbt = mean_squared_error(bt_true_train,bt_preds_train)**0.5
        rmse_rtt = mean_squared_error(bt_true_train,bt_rttov_train)**0.5
        bt_gbtlog = 'train: GBT results:mae={:.4f}, rmse={:.4f}'.format(mae_gbt,rmse_gbt)
        bt_rttovlog = 'train: RTTOV results:mae={:.4f}, rmse={:.4f}'.format(mae_rtt,rmse_rtt)
        flog.write(bt_gbtlog)
        flog.write('\n')
        flog.write(bt_rttovlog)
        ################################# test ########################
        tlevel_test = testx[:,nlv:nlv*2]
        emis_test = emiss[test_index]
        tsrf_test = tsrf[test_index]
        taosrf_test = taosrf[test_index]

        emis_test = np.delete(emis_test,outlier_test)
        taosrf_test = np.delete(taosrf_test,outlier_test)
        tsrf_test = np.delete(tsrf_test,outlier_test)

        bt_preds_test = bt_GBT(trans_preds_test,tlevel_test,emis_test,tsrf_test,taosrf_test,ch_info[channel]['scale'],ch_info[channel]['offset'],ch_info[channel]['cv'])

        # 画亮温曲线图
        bt_true_test = bt_true[test_index]#[valid_index]
        bt_rttov_test = bt_rttov[test_index]#[valid_index]

        bt_true_test = np.delete(bt_true_test,outlier_test)
        bt_rttov_test = np.delete(bt_rttov_test,outlier_test)

        bias_pred_test = bt_preds_test-bt_true_test
        bias_rtt_test = bt_rttov_test-bt_true_test

        index = np.where(np.abs(bias_pred_test)>=0.4)[0]
        print('The index of testing profiles whose bias are greater than 0.4:',index)
        x = np.arange(len(bias_pred_test))

        # fignum += 1
        # plt.figure(fignum,dpi=300)
        # plot_bt(x,bt_true_test,bt_preds,bt_rttov_test,title)
        # figname = 'new_BT_'+channel+'.png'
        # plt.savefig(figpath/figname)
        # plt.close()
        #plt.show()
        fignum += 1
        plt.figure(fignum,dpi=300)
        plot_bias_bt(x,bias_pred_test,bias_rtt_test,title)
        figname = 'BT_bias_test'+channel+'.png'
        plt.savefig(figpath/figname)
        # plt.close('all')
        #plt.show()
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

        ytrain = np.delete(Y[train_index,:],outlier_train,axis=0)
        taorttovtrain = np.delete(tao_rttov[train_index,:],outlier_train,axis=0)
        ytest = np.delete(Y[test_index,:],outlier_test,axis=0)
        taorttovtest = np.delete(tao_rttov[test_index,:],outlier_test,axis=0)
        bias_levs_train, rmse_levs_train, mre_train = calc_err(trans_preds_train,ytrain)
        bias_levs_test, rmse_levs_test, mre_test = calc_err(trans_preds_test,ytest)
        bias_levs_train_rttov,rmse_levs_train_rttov,mre_train_rttov = calc_err(taorttovtrain,ytrain)
        bias_levs_test_rttov,rmse_levs_test_rttov,mre_test_rttov = calc_err(taorttovtest,ytest)
        fignum += 1
        plt.figure(fignum,dpi=300)
        # print(bias_levs_train.shape,rmse_levs_train.shape,bias_levs_train_rttov.shape,rmse_levs_train_rttov.shape,p.shape)
        plot_error_level(bias_levs_train, rmse_levs_train,bias_levs_train_rttov,rmse_levs_train_rttov,p,title)
        figname = method+'_train_bias_nonorm_log.png'
        plt.savefig(figpath/figname,dpi=300)
        # plt.close()
        fignum += 1
        plt.figure(fignum,dpi=300)
        plot_mre(mre_train,mre_train_rttov,title)
        figname = method+'_train_mre_nonorm_log.png'
        plt.savefig(figpath/figname)
        # plt.close()
        fignum += 1
        plt.figure(fignum,dpi=300)
        plot_error_level(bias_levs_test, rmse_levs_test,bias_levs_test_rttov,rmse_levs_test_rttov,p,title)
        figname = method+'_test_bias_nonorm.png'
        plt.savefig(figpath/figname)
        # plt.close()
        fignum += 1
        plt.figure(fignum,dpi=300)
        plot_mre(mre_test,mre_test_rttov,title)
        figname = method+'_test_mre_nonorm.png'
        plt.savefig(figpath/figname)
        ############################## scatter pictures ##################################
        train_mae = mean_absolute_error(ytrain,trans_preds_train)
        test_mae = mean_absolute_error(ytest,trans_preds_test)
        train_rmse = mean_squared_error(ytrain,trans_preds_train)**0.5
        test_rmse = mean_squared_error(ytest,trans_preds_test)**0.5
        train_msle = 1.0E04*mean_squared_log_error(1+ytrain,1+trans_preds_train)
        test_msle = 1.0E04*mean_squared_log_error(1+ytest,1+trans_preds_test)

        pro_test_mae = np.sum(trans_preds_test-ytest,axis=1)
        pro_test_rmse = np.sum((trans_preds_test-ytest)**2,axis=1)
        print('The index of maximum mae profile:',np.argsort(-pro_test_mae),np.argsort(-pro_test_rmse))

        tao_rttov_test = np.delete(tao_rttov[test_index],outlier_test,axis=0)
        bias_rttov_test = mean_absolute_error(ytest, tao_rttov_test)
        rmse_rttov_test = mean_squared_error(ytest, tao_rttov_test) ** 0.5
        msle_rttov_test = 1.0E04 * mean_squared_log_error(1+ytest, 1+tao_rttov_test)

        tao_rttov_train = np.delete(tao_rttov[train_index],outlier_train,axis=0)
        bias_rttov_train = mean_absolute_error(ytrain, tao_rttov_train)
        rmse_rttov_train = mean_squared_error(ytrain, tao_rttov_train) ** 0.5
        msle_rttov_train = 1.0E04 * mean_squared_log_error(1 + ytrain, 1 + tao_rttov_train)

        fignum += 1
        plt.figure(fignum,dpi=300)
        scatter_line_trans(ytrain,trans_preds_train,taorttovtrain,title,train_mae,train_rmse,train_msle,bias_rttov_train,rmse_rttov_train,msle_rttov_train)
        figname = method+'_train_trans_all'+'.png'
        plt.savefig(figpath/figname)
        # plt.close()
        fignum += 1
        plt.figure(fignum,dpi=300)
        scatter_line_trans(ytest,trans_preds_test,taorttovtest,title,test_mae,test_rmse,test_msle,bias_rttov_test,rmse_rttov_test,msle_rttov_test)
        figname = method+'_test_trans_all'+'.png'
        plt.savefig(figpath/figname)
        plt.close()

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
