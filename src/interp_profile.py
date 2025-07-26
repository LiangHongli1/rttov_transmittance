# -*- coding: utf-8 -*-
"""
@ Time: 2020-1-13
@ author: LiangHongli
@ Mail: Helen_Liang1@outlook.com
"""

import numpy as np
from pathlib import Path
from scipy.interpolate import interp1d
import pandas as pd

def main():
    prodir = Path(r'')
    pfile = Path(r'G:\DL_transmitance\profiles\model_level_definition.xlsx')
    pdata = pd.read_excel(pfile,sheet_name='101L')
    pp = np.array(pdata['ph[hPa]'].values,dtype=np.float)
    for dd in sorted(prodir.glob('*')):
        atmdir = dd/'atm'
        p = np.loadtxt(atmdir/'p.txt')
        t = np.loadtxt(atmdir/'t.txt')
        q = np.loadtxt(atmdir/'q.txt')
        o3 = np.loadtxt(atmdir/'o3.txt')
        tinter = interp1d(p,t,kind='spine')
        tt = tinter(pp)
        qinter = interp1d(p,q,kind='spine')
        qq = qinter(pp)
        o3inter = interp1d(p,o3,kind='spine')
        o33 = o3inter(pp)
        np.savetxt(atmdir/'p.txt',pp.reshape(-1,1),fmt='%.6E')
        np.savetxt(atmdir/'t.txt',tt.reshape(-1,1),fmt='%.6E')
        np.savetxt(atmdir/'q.txt',qq.reshape(-1,1),fmt='%.6E')
        np.savetxt(atmdir/'o3.txt',o33.reshape(-1,1),fmt='%.6E')


if __name__ == '__main__':
    main()
