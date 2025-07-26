# -*- coding: utf-8 -*-
"""
@ Time: 2020-4-13
@ author: LiangHongli
@ Mail: Helen_Liang1@outlook.com
"""

from pathlib import Path
import read_files as rf
import numpy as np
import pandas as pd

def save_bt(bt_dir, dir_flag,nch):
    for k, kdir in enumerate(sorted(bt_dir.glob(dir_flag))):
        out_dir = kdir / 'out' / 'direct'
        if k == 0:
            bt_iras = rf.bt_clear(out_dir / 'radiance.txt', nch)
            emiss = rf.emisi_suf(out_dir / 'emissivity_out.txt', nch)
            srf_trans = rf.trans_srf(out_dir / 'transmission.txt', nch)
        else:
            bt_iras = np.append(bt_iras, rf.bt_clear(out_dir / 'radiance.txt', nch), axis=0)
            emiss = np.append(emiss, rf.emisi_suf(out_dir / 'emissivity_out.txt', nch), axis=0)
            srf_trans = np.append(srf_trans, rf.trans_srf(out_dir / 'transmission.txt', nch), axis=0)

    return bt_iras, emiss, srf_trans

def main():
    iras_dir = Path('/home/sky/rttov121/rttov_test/my_test.1.gfortran/iras')
    dir_flag = 'ERA*'
    nch = 3
    bt_iras, emiss, srf_trans = save_bt(iras_dir,dir_flag,nch)
    fsave = Path('/mnt/hgfs/DL_transmitance/O-B_validation/bt_rttov_200807140006.xlsx')
    writer = pd.ExcelWriter(fsave)
    df = pd.DataFrame(bt_iras,columns=['ch11','ch12','ch13'])
    df.to_excel(writer,sheet_name='bt')
    df = pd.DataFrame(emiss,columns=['ch11','ch12','ch13'])
    df.to_excel(writer,sheet_name='emissivity')
    df = pd.DataFrame(srf_trans,columns=['ch11','ch12','ch13'])
    df.to_excel(writer,sheet_name='srf_trans')
    writer.save()
    writer.close()


if __name__ == '__main__':
    main()