# -*- coding: utf-8 -*-
"""
@ Time: {DATE},{TIME}
@ author: LiangHongli
@ Mail: Helen_Liang1@outlook.com
"""
from pathlib import Path
from calc_BT import bt_GBT

fx9 = Path(r'G:\DL_transmitance\revised datasets\dataset_101L_rttov7.HDF')
fx7 = Path(r'G:\DL_transmitance\revised datasets\dataset_101L_rttov7_iasi7.HDF')


bt_train_ytrain,btair_train_ytrain = bt_GBT(Y_train[:,:],tlevel_train,emis_train,tsrf_train,taosrf_train,
                                       ch_info[channel]['scale'],ch_info[channel]['offset'],ch_info[channel]['cv'])
_,btair_train_rttov = bt_GBT(tao_rttov_train[:,:],tlevel_train,emis_train,tsrf_train,taosrf_train,
                           ch_info[channel]['scale'],ch_info[channel]['offset'],ch_info[channel]['cv'])

