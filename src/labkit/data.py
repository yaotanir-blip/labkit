#read off data frmo a given file, or generate arbitrary data from a given scenario 

import numpy as np 

def fileread(filepath='',generate_data = False,skiprows=1,delimiter=',',n=50):
    """
    reads a csv file located in filepath.
    generate_data generates arbitrary data with n rows 

    use example - a = fileread("path")
    returns a n x m single dictionary array with column titles as keys
    """
    if not generate_data:
        headers = getheader(filepath=filepath)
        data = np.loadtxt(filepath,skiprows=skiprows,unpack=True,delimiter=delimiter,encoding="latin-1")
        cols = {clean(h):col for h, col in (zip(headers,data))}
        return cols
    else:
        return data_gen(n)

def getheader(filepath='',delimiter=','):
    return np.loadtxt(filepath,max_rows=1,dtype=str,unpack=True,delimiter=delimiter,encoding="latin-1")
        

def clean(h):
    return h.split('(')[0].strip().lower().replace(' ','_')

def data_gen(n=50):
    x_ar = np.linspace(1,100,num=n)
    dx = np.ones(len(x_ar))*0.02
    dy = np.ones(len(x_ar))*0.4
    y_ar = x_ar * 0.34 + np.random.normal(0, dy)
    headers = ['x','dx','y','dy']
    data = [x_ar,dx,y_ar,dy]
    cols = {clean(h):col for h, col in (zip(headers,data))}
    return cols