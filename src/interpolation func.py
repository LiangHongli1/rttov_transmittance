##################### interpolation functions ##########################################
def bilinear_interp(v1,v2,v3,v4,x1,x2,y1,y2,x,y):
    """
    bilinear interpolation for O-B( interpolate gridded data to ungridded data)
    :param v1: value of the first point, coordinate=(x1,y1), scalar or vector
    :param v2: value of the second point, coordinate=(x1,y2)
    :param v3: value of the third point, coordinate=(x2,y1)
    :param v4: value of the fourth point, coordinate=(x2,y2)
    :param x1: coordinate, latitude
    :param x2: coordinate, latitude
    :param y1: coordinate, longitude
    :param y2: coordinate, longitude
    :param x: coordinate, latitude
    :param y: coordinate, longitude
    :return: the interpolated value(s)
    """
    dx = x2-x1
    dy = y2-y1
    f1 = (v1*(x2-x)+v2*(x-x1))/dx
    f2 = (v3*(x2-x)+v4*(x-x1))/dx
    return (f1*(y2-y)+f2*(y-y1))/dy

def idw_interp(v1,v2,v3,v4,x1,x2,y1,y2,x,y):
    d1 = np.sqrt((x-x1)**2+(y-y1)**2)
    d2 = np.sqrt((x-x1)**2+(y-y2)**2)
    d3 = np.sqrt((x-x2)**2+(y-y1)**2)
    d4 = np.sqrt((x-x2)**2+(y-y2)**2)
    return (v1/d1+v2/d2+v3/d3+v4/d4)/(1/d1+1/d2+1/d3+1/d4)

def interp_profile(p_source,v_source,p_target):
    '''
    interpolate profiles with linear method
    :param p_source: pressure, unit=hPa, vector
    :param v_source: variable, vector
    :param p_target:
    :return: interpolated variable
    '''
    interp = interp1d(p_source,v_source,kind='linear',fill_value='extrapolate')
    return interp(p_target)