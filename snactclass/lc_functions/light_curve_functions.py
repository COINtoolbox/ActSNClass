import numpy as np
import scipy.interpolate as interp 

def karpenka(time, A, B, t0, t1, t_fall, t_rise):
    """
    Double peaked exponential light curve function of Karpenka et al. (2013).
    Setting B to zero is the functional equivalent of Bazin (2011)

    :param time: the time in MJD
    :param A: normalization of first pulse
    :param B: normaliztion of second pulse
    :param t0: zero time of the pulse
    :param t1: zero time of the second pulse
    :param t_fall: decay tiem constant
    :param t_rise: rise time constant
    """
    t1m = time-t1
    t0m = time-t0

    return A*(1+ (B *t1m*t1m) )* np.exp(-t0m/t_fall )/ (1+ np.exp(-t0m/t_rise))

def sanders(time, Yb, Mp, alpha,  beta1,  beta2, betadN, betadC, t0,t1,tp,t2,td):
    """
    Type IIb/c supernova function from Sanders et al. (2014). These times are defined as durations
    not points.

    :param time: time in MJD
    :param Yb: background level
    :param alpha: first power law index
    :param beta1: second decay constant
    :param beta2: third decay constant
    :param betadC: forth decay constant
    :param betadN: fifth decay constant
    :param t0: start time of pulse
    :param t1: duration of rise
    :param tp: duration to peak
    :param t2: duration to decay
    :param td: duration to end

    """
    # normalize the constants
    # note: the paper has this wrong
    
    
    t = time - t0
    
    M1 = Mp/np.exp(beta1*tp)
    M2 = Mp/(np.exp(beta2*t2))
    Md = M2/np.exp(betadN*td)

    # select the time segments
    
    idx1=  (t>=0.) & (t<t1)
    idx2 = (t1<=t) & (t<t1+tp)
    idx3 = (t1+tp<=t) & (t<t1+tp+t2)
    idx4 = (t1+tp+t2<=t) & (t<t1+tp+t2+td)
    idx5 = t >= t1+tp+t2+td
    
    # calculate the power laws
    
    arg1 = M1*np.power(t[idx1]/t1,alpha)
    arg2 = M1*np.exp(beta1*(t[idx2]-t1))
    arg3 = Mp * np.exp(-beta2*(t[idx3]-(tp+t1)))
    arg4 = M2 * np.exp(-betadN*(t[idx4]-(t2+tp+t1)))
    arg5 = Md * np.exp(-betadC * (t[idx5] - (td+t2+tp+t1)))

    # now create an array that is just background
    
    out = np.zeros_like(t) + Yb

    # set the segments
    

    out[idx1] += arg1
    out[idx2] += arg2
    out[idx3] += arg3
    out[idx4] += arg4
    out[idx5] += arg5
    
  
    return out


def newling(time, A, phi, psi, k, sigma):
    """
    SN lightcurve function from Newling et al. (2011). 
    
    :param time: time in MJD
    :param A: Normalization
    :param phi: start time of pulse
    :param psi: tail value of pulse
    :param k: temporal streching
    :param sigma: temporal stretching
    """
    
    # calculate the spline function
    
    # create an empty array

    out = np.zeros_like(time)
    idx = time>=phi

    
    tail = np.zeros_like(time)

    # compute tau
    tau = k*sigma+phi
    
    # figure out where on the x-axis we compute things
    idx_spline = (phi<time) & (time<tau)
    idx_end = tau <= time
    
    # add the constants on
    tail[idx_end] = psi

    if (~idx_spline).sum()>=2:
        # create a spline in the regions between phi and tau    
        spline = interp.CubicSpline(time[~idx_spline],tail[~idx_spline])

        tail = spline(time)
   
    
    
    # now get the rest
    arg = (time[idx]-phi)/sigma
    
    out[idx] = A * np.power(arg,k) * np.exp(-arg) *np.power(k,-k) * np.exp(k)

    return out+tail


def bazin(time, A,t0,t_fall,t_rise):

    return karpenka(time,A,0.,t0,0.,t_fall,t_rise)






