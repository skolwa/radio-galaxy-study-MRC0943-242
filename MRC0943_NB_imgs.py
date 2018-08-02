# S.N. Kolwa (2017)
# MRC0943_NB_imgs.py

# Purpose: 
# - Construct narrow-band images 
# - Rest-frame UV lines detected by  
# 	MUSE in WFM (4800 - 9300 Ang) for z=2.92 source

import matplotlib.pyplot as pl
import numpy as np 

from astropy.io import fits
from astropy.wcs import WCS

import mpdaf.obj as mpdo
import scipy.ndimage as ndimage
import matplotlib.ticker as tk

import warnings
from astropy.utils.exceptions import AstropyWarning

from itertools import chain 

import sys
from time import time

import astropy.cosmology as ac

from functions import*

params = {'legend.fontsize': 18,
          'legend.handlelength': 2}

pl.rcParams.update(params)

pl.rc('text', usetex=True)
pl.rc('font', **{'family':'monospace', 'monospace':['Computer Modern Typewriter']})

t = time()

img_scale = 'linear'

#ignore those pesky warnings
warnings.filterwarnings('ignore', category=UserWarning, append=True)
warnings.simplefilter  ( 'ignore', category=AstropyWarning         )

#custom-selected min,max wavelengths
spec_feat 		= ['HeII','HeII', 'HeII', 'CIV','CIII]','CII]','CII','SiIV','OIII]','NIV]','NV']+['Lya']*4 
lam1 			= [6422., 6396., 6445., 6050., 7455., 9096., 5232., 5472., 6500., 5820., 4848., 4732.5, 4754.3, 4768.8, 4778.8]
lam2 			= [6430., 6406., 6450., 6090., 7485., 9146., 5258., 5533., 6554., 5841., 4892., 4747.5, 4758.8, 4776.2, 4793.8]


#radial velocity (Doppler)
def vel(wav_obs,wav_em,z):
	c = 2.9979245800e5					#km/s
	v = c*((wav_obs/wav_em/(1.+z)) - 1.)
	return v

#display spectral range on the spectrum plot
#for Lya and HeII, different colour regions on same plot hence arrays below

Lya1 = [lam1[11], lam2[11]]
Lya2 = [lam1[12], lam2[12]]
Lya3 = [lam1[13], lam2[13]]
Lya4 = [lam1[14], lam2[14]]

# HeII
z = 2.923
wav_rest = 1640.4

HeII1 = [vel(lam1[0],wav_rest,z), vel(lam2[0],wav_rest,z)]
HeII2 = [vel(lam1[1],wav_rest,z), vel(lam2[1],wav_rest,z)]
HeII3 = [vel(lam1[2],wav_rest,z), vel(lam2[2],wav_rest,z)]


# define region for extraction
center = (48,46)
radius = 3

for spec_feat,lam1,lam2 in zip(spec_feat,lam1,lam2):

	print spec_feat
	#--------------------------
	# LOAD spectral data cubes
	#--------------------------
	fname = "./out/"+spec_feat+"_cs.fits"
	cube 		= mpdo.Cube(fname,ext=1)

	#---------------
	# LINE EMISSION
	#---------------
	#get WCS data from header
	hdu 			= fits.open(fname)[1]
	wcs 			= WCS(hdu.header).celestial

	pl.figure(figsize=(8,8))
	# cont_cube_ap 	= cube.subcube_circle_aperture(center=center, radius=radius,unit_center=None,unit_radius=None)
	# spec = cont_cube_ap.sum(axis=(1,2))

	# #shift centre
	center1 = (center[0]-1,center[1]-1)
	
	cont_cube = cube.subcube(center1, (2*radius+1), unit_center=None, unit_size=None)
	
	subcube_mask(cont_cube,radius)

	spec = cont_cube.sum(axis=(1,2))

	wav 	=  spec.wave.coord()  		# Ang
	flux 	=  spec.data				#1.e-20 erg / s / cm^2 / Ang
	# pl.plot(wav,flux,c='k',drawstyle='steps-mid')

	p1,p2 			= spec.wave.pixel([lam1,lam2], nearest=True)

	line_em 		= cube[p1:p2+1,:,:]		#spectral integration for narrow-band image

	line_em_img 	= line_em.sum(axis=0) 	#create narrow-band image

	ax = pl.gca()

	pl.xlabel(r'Observed Wavelength ($\AA$)', fontsize=18)	
	pl.ylabel(r'Flux Density (10$^{-20}$ erg s$^{-1}$ cm$^{-2}$  $\AA^{-1}$)', fontsize=18)

	for label in ax.xaxis.get_majorticklabels():
		label.set_fontsize(16)

	for label in ax.yaxis.get_majorticklabels():
		label.set_fontsize(16)

	# ax.yaxis.set_major_formatter( tk.FuncFormatter(lambda x,pos: '%.1f'%(x*1.e-2) ) )	

	if spec_feat == 'Lya':
		spec.plot(c='k')

		pl.axvspan(Lya1[0],Lya1[1],ymax=0.98,color='blue',alpha=0.2)
		pl.axvspan(Lya2[0],Lya2[1],ymax=0.98,color='green',alpha=0.2)
		pl.axvspan(Lya3[0],Lya3[1],ymax=0.98,color='orange',alpha=0.2)
		pl.axvspan(Lya4[0],Lya4[1],ymax=0.98,color='red',alpha=0.2)
		# pl.title(spec_feat+r' integrated flux widths')
		pl.savefig('./out/narrow-band/Lya_profile.png')

	elif spec_feat == 'HeII':
		N = len(wav)

		wav_arr = [ wav[i] for i in xrange(N) ]
		vel_arr = [ vel(wav_arr[i], wav_rest, z) for i in xrange(N) ]

		pl.plot(vel_arr, flux, c='k', drawstyle='steps-mid')
		pl.axvspan(HeII1[0],HeII1[1],ymax=0.98,color='green',alpha=0.2)
		pl.axvspan(HeII2[0],HeII2[1],ymax=0.98,color='blue',alpha=0.2)
		pl.axvspan(HeII3[0],HeII3[1],ymax=0.98,color='red',alpha=0.2)
		# pl.title(spec_feat+r' integrated flux widths')
		pl.xlim(-2500.,2500.)
		pl.savefig('./out/narrow-band/HeII_profile.png')
		pl.savefig('./out/narrow-band/HeII_profile.pdf')

	else:
		spec.plot(c='k')

		pl.axvspan(lam1,lam2,ymax=0.98,color='magenta',alpha=0.2)
		# pl.title(spec_feat+r' integrated flux width ('+`int(lam1)`+'-'+`int(lam2)`+'$\AA$)')
		pl.savefig('./out/narrow-band/'+spec_feat+'_profile.png') 
	
	#------------------
	# get VLA CONTOURS
	#------------------
	vla = fits.open('/Users/skolwa/DATA/VLA_DATA/0943C.ICLN')[0]
	wcs_vla = WCS(vla.header).celestial
	
	#define contour parameters
	radio_rms 		= 8.e-5
	
	number_contours =	4
	start_sigma		= 	3
	odd_number 		=   1
	
	start_level 	= radio_rms*start_sigma
	contours 		= np.zeros(number_contours)
	contours[0] 	= start_level
	
	for i in range(1,number_contours):
		contours[i] = start_level*(np.sqrt(2))*odd_number
		odd_number	+= 2
	# 	print odd_number
	
	print contours
	print start_level

	fig = pl.figure()
	fig.add_subplot(111,projection=wcs_vla)
	vla_arr = vla.data[0,0,:,:]

	# pl.contour(vla_arr,levels=contours,colors='white')
	# ax = pl.imshow(vla_arr,origin='lower',cmap='gist_gray',vmin=-50,vmax=1000)
	# pl.colorbar(ax,orientation = 'vertical')
	# pl.savefig("./out/narrow-band/VLA_cont_0943.png")
	# pl.show()
	
	#save VLA .ICLN file in .fits format
	fits.writeto('/Users/skolwa/DATA/VLA_DATA/0943C.fits',vla_arr,clobber=True)

	#5arcsec -> kpc conversion
	dl 	  = ac.Planck15.luminosity_distance(2.923)	#Mpc
	z     = 2.923
	D     = 40.		#kpc

	theta = D*(1.+z)**2/(dl.value*1000.)  #radians
	theta_arc = theta*206265.		#arcsec

	# print "%.2f arcsec = 40 kpc" %theta_arc

	#--------------------------------------------------------------
	# OVERLAY VLA contours on Gaussian-smoothed narrow band images
	#--------------------------------------------------------------
	fig = pl.figure()
	ax1 = fig.add_axes([0.15, 0.15, 0.8, 0.8], projection=wcs)
	line_em_arr = line_em_img.data[:,:]

	# (x,y) are dimensions of image array
	x = len(line_em_arr)	#dimension 1
	y = len(line_em_arr[0])	#dimension 2

	#convert from erg/s/cm^2/Ang/pix to erg/s/cm^2/Ang/arcsec^2: 1pix = 0.04 arcsec^2
	line_em_arr = [ [line_em_arr[i][j]/0.04 for j in range(y)] for i in range(x) ]

	#vmin,vmax auto-scale
	pix = list(chain(*line_em_arr))
	pix_rms = np.sqrt(np.mean(np.square(pix)))
	pix_med = np.median(pix)
	vmax = pix_med + pix_rms
	vmin = pix_med - 0.4*pix_rms
	print vmin,vmax

	ax1.contour(vla_arr,levels=contours,colors='yellow',transform=ax1.get_transform(wcs_vla),linewidths=0.8)
	pl.annotate(s='', xy=(1.4,1.4), xytext=(15.5,1.4), arrowprops=dict(arrowstyle='->',ec='red'))		#horizontal
	pl.annotate(s='', xy=(15.,15.), xytext=(15.,0.8), arrowprops=dict(arrowstyle='->',ec='red'))		#vertical

	pl.annotate(s='', xy=(98.,1.4), xytext=(75.,1.4), arrowprops=dict(arrowstyle='<->',ec='red'))
	pl.text(0.8,0.05,'40 kpc',fontsize=10,color='red', transform=ax1.transAxes)
	pl.text(13.5,15.,'N',color='red')
	pl.text(-0.5,0.,'E',color='red')
	pl.xlabel('RA (J2000)', fontsize=16)
	pl.ylabel('DEC (J2000)', fontsize=16)
	
	gs_img = ndimage.gaussian_filter(line_em_arr, sigma=(1, 1), order=0)
	figure = ax1.imshow(gs_img,cmap='gray_r',origin='lower',interpolation='nearest',\
		transform=ax1.get_transform(wcs), vmin=vmin, vmax=vmax)
	cb = pl.colorbar(figure, orientation = 'vertical')
	#check these
	cb.set_label(r'10$^{-20}$ erg s$^{-1}$ cm$^{-2}$ arcsec$^{-2}$',rotation=90, fontsize=16)
	# pl.title(spec_feat+r' M$_0$ map ('+`int(lam1)`+'-'+`int(lam2)`+r'$\AA$)')

	if lam1 == 4732.5:
		# pl.savefig("./out/narrow-band/"+spec_feat+"1 grey VLA.png")
		pl.savefig("./out/narrow-band/"+spec_feat+"1_VLA.pdf")

	elif lam1 == 4754.3:
		# pl.savefig("./out/narrow-band/"+spec_feat+"2 grey VLA.png")
		pl.savefig("./out/narrow-band/"+spec_feat+"2_VLA.pdf")

	elif lam1 == 4768.8:
		figure = ax1.imshow(gs_img,cmap='gray_r',origin='lower',interpolation='nearest',\
			transform=ax1.get_transform(wcs), vmin=vmin, vmax=0.2*vmax)
		# pl.savefig("./out/narrow-band/"+spec_feat+"3 grey VLA.png")
		pl.savefig("./out/narrow-band/"+spec_feat+"3_VLA.pdf")

	elif lam1 == 4778.8:
		# pl.savefig("./out/narrow-band/"+spec_feat+"4 grey VLA.png") 
		pl.savefig("./out/narrow-band/"+spec_feat+"4_VLA.pdf")

	elif lam1 == 6396.:
		# pl.savefig("./out/narrow-band/"+spec_feat+"_blue.png") 
		pl.savefig("./out/narrow-band/"+spec_feat+"_blue_VLA.pdf")

	elif lam1 == 6422.:
		# pl.savefig("./out/narrow-band/"+spec_feat+" grey VLA.png") 
		pl.savefig("./out/narrow-band/"+spec_feat+"_VLA.pdf")

	elif lam1 == 6445.:
		# pl.savefig("./out/narrow-band/"+spec_feat+"_red.png") 
		pl.savefig("./out/narrow-band/"+spec_feat+"_red_VLA.pdf")

	else:
		figure = ax1.imshow(gs_img,cmap='gray_r',origin='lower',interpolation='nearest',\
			transform=ax1.get_transform(wcs), vmin=0.2*vmin, vmax=vmax)
		pl.savefig("./out/narrow-band/"+spec_feat+"_grey_VLA.pdf")
		# pl.savefig("./out/narrow-band/"+spec_feat+" grey VLA.png")	

time_elapsed = time() - t
print "build time: %4.2f s" %time_elapsed	