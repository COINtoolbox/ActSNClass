import numpy as np
import scipy.interpolate as interp 
import theano.tensor as tt


def karpenka_theano(time, A, B, t0, t1, t_fall, t_rise):
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
    
    return A*(1+B *t1m*t1m)* tt.exp(-t0m/t_fall )/ (1+ tt.exp(-t0m/t_rise))




def sanders_theano(time, Yb,Mp, alpha,  beta1,  beta2, betadN, betadC, t0,t1,tp,t2,td):
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
    # precompute the sums

    t = time-t0
    
    sum1 = t1
    sum2 = sum1+tp
    sum3 = sum2+t2
    sum4 = sum3+td

    # normalize the segments
    # note these are wrong in the paper
    
    M1 = Mp/tt.exp(beta1*tp)
    M2 = Mp/tt.exp(beta2*t2)
    Md = M2/tt.exp(betadN*td)

    # compute the arguments
    arg1 = M1*tt.power(t/t1,alpha)
    arg2 = M1*tt.exp(beta1*(t-t1))
    arg3 = Mp*tt.exp(-beta2*(t-(tp+t1)))
    arg4 = M2*tt.exp(-betadN*(t-(t2+tp+t1)))
    arg5 = Md*tt.exp(-betadC*(t- (td+t2+tp+t1)))

    # theano piecewise
    
    return tt.switch(tt.lt(t,0.),0.,
             tt.switch(tt.lt(t,sum1),arg1,
                       tt.switch(tt.lt(t,sum2),arg2,
                           tt.switch(tt.lt(t,sum3),arg3,
                                     tt.switch(tt.lt(t,sum4),arg4,arg5))))) + Yb

