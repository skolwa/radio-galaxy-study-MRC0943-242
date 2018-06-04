#S.N. Kolwa (2018)
#functions.py

import numpy as np

#define Gaussian models
def gauss(x, amp, wid, g_cen, cont):
	gauss = (amp/(np.sqrt(2*np.pi)*wid)) * np.exp(-(x-g_cen)**2 /(2*wid**2))
	return gauss + cont

def gauss_nocont(x, amp, wid, g_cen):
	gauss = (amp/(np.sqrt(2*np.pi)*wid)) * np.exp(-(x-g_cen)**2 /(2*wid**2))
	return gauss

def dgauss(x, amp1, wid1, g_cen1, amp2, wid2, g_cen2, cont):
	gauss1 = (amp1/(np.sqrt(2*np.pi)*wid1)) * np.exp(-(x-g_cen1)**2 / (2*wid1**2))
	gauss2 = (amp2/(np.sqrt(2*np.pi)*wid2)) * np.exp(-(x-g_cen2)**2 / (2*wid2**2))
	return gauss1 + gauss2 + cont

def dgauss_nocont(x, amp1, wid1, g_cen1, amp2, wid2, g_cen2):
	gauss1 = (amp1/(np.sqrt(2*np.pi)*wid1)) * np.exp(-(x-g_cen1)**2 / (2*wid1**2))
	gauss2 = (amp2/(np.sqrt(2*np.pi)*wid2)) * np.exp(-(x-g_cen2)**2 / (2*wid2**2))
	return gauss1 + gauss2 

def subcube_mask(cube,radius): #masks optimally (QfitsView style) for R=3
	N = radius
	x1 = 2*N - 2
	x2 = 2*N - 1
	x3 = 2*N
	x4 = 2*N + 1 

	cube.data[:, x3:x4 , x3:x4] = np.ma.masked
	cube.data[:, x2:x3 , x3:x4] = np.ma.masked
	cube.data[:, x3:x4 , x2:x3] = np.ma.masked

	cube.data[:, 0:1 , 0:1] = np.ma.masked
	cube.data[:, 1:2 , 0:1] = np.ma.masked
	cube.data[:, 0:1 , 1:2] = np.ma.masked

	cube.data[:,x3:x4 , 0:1] = np.ma.masked
	cube.data[:,x2:x3 , 0:1] = np.ma.masked
	cube.data[:,x3:x4 , 1:2] = np.ma.masked

	cube.data[:,0:1 , x3:x4] = np.ma.masked
	cube.data[:,0:1 , x2:x3] = np.ma.masked
	cube.data[:,1:2 , x3:x4] = np.ma.masked

	cube.data[:, x1:x2 , x3:x4] = np.ma.masked
	cube.data[:, x3:x4 , x1:x2] = np.ma.masked
	
	cube.data[:, 2:3 , 0:1] = np.ma.masked
	cube.data[:, 0:1 , 2:3] = np.ma.masked 
	
	cube.data[:,x1:x2 , 0:1] = np.ma.masked
	cube.data[:,x3:x4 , 2:3] = np.ma.masked
	
	cube.data[:,0:1 , x1:x2] = np.ma.masked
	cube.data[:,2:3 , x3:x4] = np.ma.masked

#convert flux units to Jy
def flux_Jy(wav,flux):
	f = 3.33564095e4*flux*1.e-20*wav**2
	return f

#define y-tick marks in reasonable units
#recover flux in 10^-20 erg/s/cm^2 and convert to microJy
def flux_cgs(wav,flux_Jy):
	f = flux_Jy/(3.33564095e4*wav**2)
	return f*1.e20

# -----------------------------------------------
# 	   	     Voigt-Hjerting Models  	
# -----------------------------------------------
# Voigt-Hjerting function
def H(a, x):
    """ The H(a, u) function of Tepper Garcia 2006
    """
    P = x ** 2
    H0 = sp.e ** (-(x ** 2))
    Q = 1.5 / x ** 2
    H = H0 - a / sp.sqrt(sp.pi) / P * (H0 * H0 * (4 * P * P + 7 * P + 4 + Q) - Q - 1)
    return H

# 1st order polynomial for continuum-fitting
def str_line(x,m,c):
	return m*x + c
