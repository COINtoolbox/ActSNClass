from snactclass.lc_functions.light_curve_functions import sanders, karpenka

import matplotlib.pyplot as plt
import numpy as np



def plot_karpenka(trace,flux,burnin=1000,thin=20,alpha=.09):
    
   
    fig, ax = plt.subplots()

    for point in trace[burnin::thin]:

        for band, color in zip(['g','r','i','z'],["#dd0100","#fac901","#225095",'#9b4800']): 
            
            err = np.array(flux[band]['FLUXCALERR'])
            idx = np.isfinite(err)
            time = np.array(flux[band]['MJD'] - flux[band]['MJD'].min())[idx]
            
            pseudo_time = np.linspace(time.min(),time.max(),50)

            pred = karpenka(pseudo_time,
                          10**point['logA_%s'%band],
                          10**point['logB_%s'%band],
                          point['t0_%s'%band],
                          point['t1_%s'%band],
                          point['t_fall_%s'%band],
                          point['t_rise_%s'%band]
                         )


            ax.plot(psudo_time,pred,color,alpha=.09,zorder=-33)


    for band, color in zip(['g','r','i','z'],["#dd0100","#fac901","#225095",'#9b4800']):    


        err = np.array(flux[band]['FLUXCALERR'])
        idx = np.isfinite(err)
        err = err[idx]
        tdata = np.array(flux[band]['FLUXCAL'])[idx]
        time = np.array(flux[band]['MJD'] - flux[band]['MJD'].min())[idx]


        ax.errorbar(time,tdata,yerr=err,color=color,fmt='o',zorder=-1,label='%s data'%band)
        
        
    ax.set_xlabel('time')
    ax.set_ylabel('flux')
    ax.legend()
    
    return fig



def plot_sanders(trace, flux, burnin=1000,thin=20,alpha=.09):
    
   
    fig, ax = plt.subplots()

    for point in trace[burnin::thin]:

        for band, color in zip(['g','r','i','z'],["#dd0100","#fac901","#225095",'#9b4800']): 
            
            err = np.array(flux[band]['FLUXCALERR'])
            idx = np.isfinite(err)
            time = np.array(flux[band]['MJD'] - flux[band]['MJD'].min())[idx]
            
            pseudo_time = np.linspace(time.min(),time.max(),50)

            pred = sanders(pseudo_time,
                  10**point['logYb_%s'%band],
                  10**point['logMp_%s'%band],
                  10**point['logalpha_%s'%band],
                  10**point['logbeta1_%s'%band],
                  10**point['logbeta2_%s'%band],
                  10**point['logbetadn_%s'%band],
                  10**point['logbetadc_%s'%band],
                  point['t0_%s'%band],
                  point['t1_%s'%band],
                  point['tp_%s'%band],
                  point['t2_%s'%band],
                  point['td_%s'%band]
                 )


            ax.plot(psudo_time,pred,color,alpha=.09,zorder=-33)


    for band, color in zip(['g','r','i','z'],["#dd0100","#fac901","#225095",'#9b4800']):    


        err = np.array(flux[band]['FLUXCALERR'])
        idx = np.isfinite(err)
        err = err[idx]
        tdata = np.array(flux[band]['FLUXCAL'])[idx]
        time = np.array(flux[band]['MJD'] - flux[band]['MJD'].min())[idx]


        ax.errorbar(time,tdata,yerr=err,color=color,fmt='o',zorder=-1,label='%s data'%band)
        
        
    ax.set_xlabel('time')
    ax.set_ylabel('flux')
    ax.legend()
    
    return fig
