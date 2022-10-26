
from matplotlib import colors
import matplotlib.pyplot as plt
import numpy as np
import time
import scipy
from scipy import interpolate


def check_local_max_min(imf, t, thr):
    dt = t[1] - t[0]
    x = imf
    upper_t = []
    upper_x = []
    lower_t = []
    lower_x = []
    #find peaks and dips
    for idx_x in range(len(x)-2):
        if (x[idx_x] - x[idx_x-1]) > 0  and (x[idx_x+1] - x[idx_x]) < 0:
            upper_t.append(t[idx_x])
            upper_x.append(x[idx_x])
        if (x[idx_x] - x[idx_x-1]) < 0  and (x[idx_x+1] - x[idx_x]) > 0:
            lower_t.append(t[idx_x])
            lower_x.append(x[idx_x])
    idx_head_t = int(max(upper_t[0], lower_t[0])/dt+1)
    idx_tail_t = int((min(upper_t[-1], lower_t[-1])/dt)-1)
    #connect local max and min point
    f_upper = interpolate.interp1d(upper_t, upper_x, kind='cubic', fill_value="extrapolate")
    f_upper_inerpolate = f_upper(t[idx_head_t:idx_tail_t])
    f_lower = interpolate.interp1d(lower_t, lower_x, kind='cubic', fill_value="extrapolate")
    f_lower_inerpolate = f_lower(t[idx_head_t:idx_tail_t])
    h = (f_upper_inerpolate+f_lower_inerpolate)/2
    # python cubic function calculate error?
    print(abs(np.sum(h)))
    
def claculate_imf(signal, t, thr):
    # try:
    keep = 1
    dt = t[1] - t[0]
    x = signal
    upper_t = []
    upper_x = []
    lower_t = []
    lower_x = []
    #find peaks and dips
    for idx_x in np.arange(1,len(x)-1):
        if (x[idx_x] - x[idx_x-1]) > 0  and (x[idx_x+1] - x[idx_x]) < 0:
            upper_t.append(t[idx_x])
            upper_x.append(x[idx_x])
            plt.scatter(t[idx_x], x[idx_x], color = 'r')
        if (x[idx_x] - x[idx_x-1]) < 0  and (x[idx_x+1] - x[idx_x]) > 0:
            lower_t.append(t[idx_x])
            lower_x.append(x[idx_x])
            plt.scatter(t[idx_x], x[idx_x], color = 'b')
    
    if len(upper_t) > 1 or len(lower_t) > 1:
        idx_head_t = int(max(upper_t[0], lower_t[0])/dt+1)
        idx_tail_t = int((min(upper_t[-1], lower_t[-1])/dt)-1)
        #connect local max and min point
        f_upper = interpolate.interp1d(upper_t, upper_x, kind='cubic', fill_value="extrapolate")
        f_upper_inerpolate = f_upper(t[idx_head_t:idx_tail_t])
        f_lower = interpolate.interp1d(lower_t, lower_x, kind='cubic', fill_value="extrapolate")
        f_lower_inerpolate = f_lower(t[idx_head_t:idx_tail_t])
        # compute mean and residual
        h = (f_upper_inerpolate+f_lower_inerpolate)/2
        imf_fun = x[idx_head_t:idx_tail_t]-h
        check_local_max_min(imf_fun, t[idx_head_t:idx_tail_t], thr)
        plt.plot(t[idx_head_t:idx_tail_t], imf_fun)
        plt.show()
        return imf_fun, h, t[idx_head_t:idx_tail_t], keep
    # check extreme point no more than 1 
    else:
        keep = 0
        plt.plot(t, signal)
        plt.show()
        return None, signal, t, keep

def hht(x, t, thr):
    dict_component = {}
    iter = 1
    keep = 1
    signal = x
    seleted_t = t
    while iter<5 and keep:
        key_name = "iter_%s"%(iter)
        imf_fun, signal, seleted_t, keep = claculate_imf(signal, seleted_t, thr)
        if keep == 1:
            print()
            dict_component.update({key_name: imf_fun})
        elif keep == 0:
            dict_component.update({key_name: signal})
        iter += 1
    return dict_component    

dt = 0.01
t = np.arange(0, 10.01, dt)
x = 0.2*t + np.cos(2*np.pi*t) + 0.4*np.cos(10*np.pi*t)
thr = 0.2
y = hht(x, t, thr)