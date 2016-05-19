import scipy.optimize as O
import scipy.interpolate as I
import numpy as N
import scipy.special as S
import matplotlib.pyplot as plt





#wpherical bessel function of order n at z (a vector)
def besph(n, z):
    ans=S.jn(n+0.5, z)*N.sqrt(N.pi/(2*z))
    if n==0:
        ans[N.where(z==0)]=1
    return ans



def make_template(pkwig="", pknowig="", sigma=8.0):
    #wiggles and no wiggles spectra
    wiggles=N.loadtxt(pkwig)
    pkwig=wiggles[:,1]
    klin=wiggles[:,0]
    nowiggles=N.loadtxt(pknowig)
    pknowig=nowiggles[:,1]
    #smooth BAO bump with a 8 Mpc Gaussian
    ptemp=(pkwig-pknowig)*N.exp(-klin**2*sigma**2/2.0)+pknowig
    pk2=I.interp1d(klin, ptemp, kind='cubic')
    L=1000
    deltak=2*N.pi/L
    kk=N.arange(deltak, deltak*L, deltak)
    #smooth with a Gaussian with r=1.0 to damp high-k tail (this is done in Xu et al)
    w=N.exp(-kk*kk/2.0);
    r=N.arange(0.1, 500, 0.5)
    jrk=besph(0, N.outer(r,kk))
    xit=N.sum((w*w*pk2(kk)*kk**2*jrk*deltak)/(2*N.pi**2), 1)
    return r, xit
    




def chi_sq(x, *args):
    
    rm=args[0]
    mock=args[1]
    tempI=args[2]
    cinv=args[3]
    
    alpha, logb=x
    
    bsq=N.exp(logb)
    
    #compute the bias, which is the ratio of the mock to the template at r=50 (as per Xu et al)
    mockI=I.interp1d(rm, mock)
    bias=mockI(50.0)/tempI(50.0)
    
    
    xi_m=bias*tempI(rm*alpha)

    y=mock-xi_m*bsq
    p=N.polyfit(1.0/rm, y, 2)
    bf=p[0]/rm**2+p[1]/rm+p[2]+xi_m*bsq
    
    
    #bf = best fit correlation function for this set of (alpha, logb2)
    
    f=N.dot(N.transpose(mock-bf), N.dot(cinv, mock-bf))
    return f


def get_inv_covariance(cov_file, i_low=0, i_high=-1):
    cov=N.loadtxt(cov_file)
    cov=cov[i_low:i_high, i_low:i_high]
    return N.linalg.inv(cov)
    
    
    
    
def template_fit(numruns, cf_file="", cov_file="", rmin=30, rmax=200, pkwig="", pknowig="", sigma=8.0):
    #numruns is the number of realizations to analyze
    #cf_file should be the prefix for the files where the correlation functions are stored
    #they will be appended with cf_file+### for each realization
    #cinv_file is the file where the inverse covariance matrix to be used is stored
    #rmin and rmax are the min and max distances used in the analysis (covariance function should only include these distances)
    #pkwig and pknowig are the files with the linear power spectra with and without wiggles (used to make the template correlation function)
    #sigma is the amount by which the template BAO feature is smoothed (in Xu et al, they use values from 6.6-8.1 without reconstruction)
    
    #make template from the linear power spectra and sigma specified
    r, template=make_template(pkwig=pkwig, pknowig=pknowig, sigma=sigma)
    tempI=I.interp1d(r, template)
    
    #read in covariance matrix, invert the relevant part (needs cf file to get the r column)
    cinv=get_inv_covariance(cf_file, cov_file, rmin=rmin, rmax=rmax)
    
    
    
    alpha=N.zeros(numruns, dtype=N.float)
    logb2=N.zeros(numruns, dtype=N.float)
    
    alpha_mean=N.zeros(numruns, dtype=N.float)
    sigma_alpha=N.zeros(numruns, dtype=N.float)
    
    #x0 is initial guess for alpha and log B^2
    x0=[1.0, 0.0]
    
    for i in N.arange(numruns):
        realization="%03i" % i
        cfunc=N.loadtxt(cf_file+realization)
        #assumes that the columns of the correlation function file are [r, xi(r)]
        mock=cfunc[:,1]
        rm=cfunc[:,0]
        
        #clip the correlation function to just the r-range we are interested in
        i_low=N.where(rm>=rmin)[0][0]
        i_high=N.where(rm>=rmax)[0][0]
        rm=rm[i_low:i_high]
        mock=mock[i_low:i_high]
        
        if (i==0):
            #read in covariance matrix, invert the relevant part
            cinv=get_inv_covariance(cov_file, i_low=i_low, i_high=i_high)
        
        
        #minimize chi^2 for the given mock
        xopt=O.fmin(chi_sq, x0, args=(rm, mock, tempI, cinv), maxiter=500, disp=False)
        
        #these are the alpha and logb2 at the minimum chi^2
        alpha[i]=xopt[0]
        logb2[i]=xopt[1]
        
        #This computes the probability distribution of alpha from the chi^2 values in the alpha range (?)
        ai=N.arange(0.6, 1.4, 0.005)
        pal=N.zeros_like(ai)
        for j in N.arange(ai.size):
            pal[j]=N.exp(-xi_sq([ai[j], logb2[i]], rm, mock, tempI, cinv)/2.0)
        pal=pal/N.sum(pal)
        alpha_mean[i]=N.sum(ai*pal)
        sigma_alpha[i]=N.sqrt(N.sum((ai-alpha_mean[i])**2*pal))
        
    return alpha, logb2, alpha_mean, sigma_alpha




















