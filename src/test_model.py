'''
########################### To validate the GBT models with IRAS observations #################################

'''
from pathlib import Path
import h5py
import numpy as np
import joblib
import matplotlib.pyplot as plt
import pandas as pd

from calc_BT import bt_GBT
from LUT import ch_info
from plot_map_funcs import map_scatter,plot_bias_bt,dot_error_bt,dot_error_bt_method
from sklearn.metrics import mean_squared_error

def main():
    methods = ['GBT','XGBoost','RF']
    # model_path = Path(r'G:\DL_transmitance\models')/method
    fx = Path(r'G:\DL_transmitance\O-B_validation\X4O-B-20080714.HDF')
    fbt_rttov = Path(r'G:\DL_transmitance\O-B_validation\bt_rttov_200807140006.xlsx')
    figpath = Path(r'G:\DL_transmitance\figures\101L\o-b')
    
    with h5py.File(fx,'r') as f:
        X = f['X'][:]  # O3 is not available on GFS data, but it's available on ERA
        # emissivity = f['emissivity'][:]
        # trans_srf = f['trans_srf'][:]
        # bt_rttov = f['BT_RTTOV'][:]
        preds = f['predictors'][:]
        obs_bt = f['BT_obs'][:]

    # lon_lat = X[:,[309,310]]
    lon_lat = obs_bt[:,-2:]
    # X = np.delete(X,[303,304,305,306,308,309,310,311],axis=1)
    df_bt = pd.read_excel(fbt_rttov,sheet_name='bt')
    bt_rttov = df_bt.loc[:,['ch11','ch12','ch13']].values  # RTTOV simulated BT of IRAS channel 11 12 13
    df_emiss = pd.read_excel(fbt_rttov,sheet_name='emissivity')
    emissivity = df_emiss.loc[:,['ch11','ch12','ch13']].values
    print(emissivity.shape)
    df_srf_trans = pd.read_excel(fbt_rttov,sheet_name='srf_trans')
    srf_trans = df_srf_trans.loc[:,['ch11','ch12','ch13']].values
    nlv = 101
    nch = 3
    nsample = X.shape[0]
    xx = np.arange(nsample)
    # boundary = [-30,30,np.min(lon_lat[:,0])-0.5,np.max(lon_lat[:,0]+0.5)]
    # trans_atm = np.zeros((3,nsample,nlv))
    # errors = np.zeros((3,4)) # 3 channels, 4 error(mean bias, std of GBT, mean bias, std of RTTOV)
    # for ch in range(3):
    #     channel = 'ch'+str(ch+11)
    #     for k in range(nlv):
    #         model_name = method+'_'+channel+'_level_'+str(k)+'.m'
    #         model = joblib.load(model_path/model_name)
    #         if k==0:
    #             x = X
    #         else:
    #             x = np.concatenate((X,preds[:,:,k-1]),axis=1)
                
    #         trans_atm[ch,:,k] = model.predict(x)
            
            
    #     # print(trans_atm.shape,X[:,:nlv].shape,emissivity.shape)   trans_atm[ch,:,-1]
    #     bt_preds,btair_preds = bt_GBT(trans_atm[ch,:,:],X[:,:nlv],emissivity[:,ch],X[:,303],X[:,304],srf_trans[:,ch],
    #                                 ch_info[channel]['scale'],ch_info[channel]['offset'],ch_info[channel]['cv'])
    #     errors[ch,0] = np.mean(bt_preds-obs_bt[:,ch])
    #     errors[ch,1] = np.std(bt_preds)
    #     errors[ch,2] = np.mean(bt_rttov[:,ch]-obs_bt[:,ch])
    #     # rmse_pred = mean_squared_error(obs_bt[:,ch],bt_preds)
    #     # rmse_rttov = mean_squared_error(obs_bt[:,ch],bt_rttov[:,ch])
    #     errors[ch,3] = np.std(bt_rttov[:,ch])
        # print('Mean bias of GBT predicted BT = %.4f, RMSE = %.4f, std=%.4f' % (np.mean(bias_pred),rmse_pred,std_pred))
        # print('Mean bias of RTTOV calculated BT = %.4f, RMSE = %.4f, std=%.4f' % (np.mean(bias_rttov),rmse_rttov,std_rttov))
        # bias_pred = bt_preds-obs_bt[:,ch]
        # bias_rttov = bt_rttov[:,ch]-obs_bt[:,ch]
        # vmin = np.min(np.vstack((bias_pred,bias_rttov)))
        # vmax = np.max(np.vstack((bias_pred,bias_rttov)))
        # plt.figure(dpi=200)
        # plt.subplot(121)
        # title = method+'-obs '+ch_info[channel]['cwl']
        # map_scatter(lon_lat[:,0],lon_lat[:,1],bt_preds-obs_bt[:,ch],boundary,title,vmin,vmax)
        # # plt.ylabel('Distribution of BT bias(K)')
        # plt.subplots_adjust(left=0.1,right=0.9,wspace=0.002)
        # # plt.tight_layout()
        # plt.subplot(122)
        # title = 'RTTOV-obs '+ch_info[channel]['cwl']
        # map_scatter(lon_lat[:,0],lon_lat[:,1],bt_rttov[:,ch]-obs_bt[:,ch],boundary,title,vmin,vmax,'Bias of Brightness Temperature(K)')
        # figname = method+'20080714_bias_distribution'+channel+'.png'
        # plt.savefig(figpath/figname)
        # plt.close()
        
        # fig,axs = plt.subplots(1,3)
        # plt.figure(dpi=200)
        # vmin = np.min(np.vstack((obs_bt[:,ch],bt_rttov[:,ch],bt_preds)))
        # vmax = np.max(np.vstack((obs_bt[:,ch],bt_rttov[:,ch],bt_preds)))
        # plt.subplot(131)
        # title = 'obs '+ch_info[channel]['cwl']
        # map_scatter(lon_lat[:,0],lon_lat[:,1],obs_bt[:,ch],boundary,title,vmin,vmax)
        # # plt.subplots_adjust(wspace=0.005)
        # plt.subplot(132)
        # title = 'RTTOV '+ch_info[channel]['cwl']
        # map_scatter(lon_lat[:,0],lon_lat[:,1],bt_rttov[:,ch],boundary,title,vmin,vmax,yticks=False)
        # # plt.subplots_adjust(right=0.9)
        # plt.subplot(133)
        # title = method+' '+ch_info[channel]['cwl']
        # map_scatter(lon_lat[:,0],lon_lat[:,1],bt_preds,boundary,title,vmin,vmax,'Brightness Temperature(K)',yticks=False)
        # # axs = plt.axes([0.1,0,0.9,0.1])
        # # cb = plt.colorbar(mappable=axs[0],orientation='horizontal',ax=axs,aspect=10,spacing='uniform') #pad=0.3,fraction=0.8,
        # # cb.set_label('Brightness Temperature(K)',fontsize=10)
        # figname = method+'200807140006_BT_distribution'+channel+'.png'
        # plt.savefig(figpath/figname)
        # plt.close()

    ###############################################################################
    errors = np.zeros((3,3,4)) # 3methods, 3 channels, 4 error(mean bias, std of GBT, mean bias, std of RTTOV)
    all_bt_preds = np.zeros((nsample,3,3)) # 3channels, 3 methods
    for m,method in enumerate(methods):
        model_path = Path(r'G:\DL_transmitance\models')/method
        trans_atm = np.zeros((3,nsample,nlv))
        for ch in range(3):
            channel = 'ch'+str(ch+11)
            for k in range(nlv):
                model_name = method+'_'+channel+'_level_'+str(k)+'.m'
                model = joblib.load(model_path/model_name)
                if k==0:
                    x = X
                else:
                    x = np.concatenate((X,preds[:,:,k-1]),axis=1)
                    
                trans_atm[ch,:,k] = model.predict(x)
                
                
            # print(trans_atm.shape,X[:,:nlv].shape,emissivity.shape)   trans_atm[ch,:,-1]
            bt_preds,btair_preds = bt_GBT(trans_atm[ch,:,:],X[:,:nlv],emissivity[:,ch],X[:,303],X[:,304],srf_trans[:,ch],
                                        ch_info[channel]['scale'],ch_info[channel]['offset'],ch_info[channel]['cv'])
            errors[m,ch,0] = np.mean(bt_preds-obs_bt[:,ch])
            errors[m,ch,1] = np.std(bt_preds)
            errors[m,ch,2] = np.mean(bt_rttov[:,ch]-obs_bt[:,ch])
            # rmse_pred = mean_squared_error(obs_bt[:,ch],bt_preds)
            # rmse_rttov = mean_squared_error(obs_bt[:,ch],bt_rttov[:,ch])
            errors[m,ch,3] = np.std(bt_rttov[:,ch])
            
            #-------------------------------------------------------------
            all_bt_preds[:,ch,m] = bt_preds
    with h5py.File(r'G:\DL_transmitance\O-B_validation\bt_predicted_by_ml.HDF','w') as f:
        f['bt'] = all_bt_preds
        f['bt'].attrs['dimension'] = 'nsample, channel, method'
            #-------------------------------------------------------------
    
    # plt.figure(dpi=200)
    # plt.subplot(121)
    # dot_error_bt(errors[:,0],errors[:,2],title='200807140006',ylabel='Mean bias of BT(K)')
    # plt.subplots_adjust(wspace=0.4)
    # # plt.tight_layout()
    # plt.subplot(122)
    # dot_error_bt(errors[:,1],errors[:,3],title='200807140006',ylabel='STD of BT(K)')
    # # plt.ylabel([])
    # figname = method+'20080714_plot_error_bt.png'
    # plt.savefig(figpath/figname)
    # plt.close()
    
    # plt.figure(dpi=200)
    # plt.subplot(121)
    # dot_error_bt_method(errors[0,:,0],errors[1,:,0],errors[2,:,0],title='200807140006',ylabel='Mean bias of BT(K)')
    # plt.subplots_adjust(wspace=0.4)
    # plt.subplot(122)
    # dot_error_bt_method(errors[0,:,1],errors[1,:,1],errors[2,:,1],title='200807140006',ylabel='Mean bias of BT(K)')
    # figname = 'all_method20080714_plot_error_bt.png'
    # plt.savefig(figpath/figname)
    # plt.close()
    
    return

if __name__ == "__main__":
    main()