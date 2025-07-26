# -*- coding: utf-8 -*-
"""
@ Time: 2019/10/15
@ author: LiangHongli
@ Mail: Helen_Liang1@outlook.com
Extract data required for RTTOV input from grib2 files, and resave the needed variables to nc file for futue use.
!!!Notice: This code requires Nio module to run. Thus it has to be in linux's ncl-to-python environment.
The way to enter ncl-to-python:
On the linux platform, enter miniconda/bin directory, input "chmod 777 activate", then "source activate ncl-to-python"
"""
# import eccodes as ec
import Nio
from pathlib import Path

variable_list_forecast = {
    'HGT_P0_L100_GLL0',
    'TMP_P0_L100_GLL0',
    'SPFH_P0_L100_GLL0',
    'UGRD_P0_L103_GLL0',
    'VGRD_P0_L103_GLL0',
    'TMP_P0_L103_GLL0',
    'TMP_P0_L1_GLL0',
    'PRES_P0_L1_GLL0',
    'SPFH_P0_L103_GLL0',
    'TCDC_P0_L2_GLL0',
    'HGT_P0_L1_GLL0',
    'lv_ISBL0',
    'lv_ISBL2',
    'lat_0',
    'lon_0'
}

variable_list_analysis = {
    'lv_ISBL0',
    'lv_ISBL4',
    'lat_0',
    'lon_0',
    'PRES_P0_L1_GLL0',
    'SPFH_P0_L103_GLL0',
    'RH_P0_L100_GLL0',
    'TMP_P0_L103_GLL0',
    'TMP_P0_L100_GLL0',
    'TMP_P0_L1_GLL0',
    'UGRD_P0_L103_GLL0',
    'VGRD_P0_L103_GLL0'
}
# 'Geopotential_height_isobaric',
# 'Geopotential_height_surface',


fpath = Path('/mnt/hgfs/DL_transmitance/O-B_validation/2008_GFS/20080709')
savepath = Path('/mnt/hgfs/DL_transmitance/O-B_validation/200807_grb2_to_nc')
file_list = [x for x in fpath.glob('*.grb2')]

dodate = ['20080709']
for file in file_list:
    f = Nio.open_file(str(file),'r')
    fname = file.name[:-4] + 'nc'
    if fname.split('_')[2] not in dodate:
        print('Jump')
        continue
    outf_name = savepath/fname
    try:
        outf = Nio.open_file(str(outf_name),'c') # create new file to save the needed variable
    except:
        continue

    keys = list(f.dimensions)
    for key in keys:
        outf.create_dimension(key, f.dimensions[key])

    for var in variable_list_analysis:
        var_obj = f.variables[var]
        outf.create_variable(var,var_obj.typecode(),var_obj.dimensions)

        var_attrs = list(var_obj.__dict__.keys())
        var_attrs.sort()
        for att in var_attrs:
            value = getattr(var_obj,att)
            setattr(outf.variables[var],att,value)

        outf.variables[var].assign_value(var_obj)
        print('Variable "%s" has been written.' % var)

    print('File "%s" has been written.' % str(outf_name))
    outf.close()
    f.close()


