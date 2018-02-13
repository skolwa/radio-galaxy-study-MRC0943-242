# S.N. Kolwa (2017)
# Purpose: 
# 0943_NB_imgs.py
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

t = time()

img_scale = 'linear'

#ignore those pesky warnings
warnings.filterwarnings('ignore', category=UserWarning, append=True)
warnings.simplefilter('ignore', category=AstropyWarning)

#custom-selected min,max wavelengths
# spec_feat 	= ['HeII','CIV','CIII]','CII]','CII','SiIV','OIII]','NIV]','NV']+['Lya']*4 
# lam1 			= [6422.,6050.,7455.,9096.,5232.,5472.,6500.,5820.,4848.,4732.5,4754.3,4768.8,4778.8]
# lam2 			= [6430.,6090.,7485.,9146.,5258.,5533.,6554.,5841.,4892.,4747.5,4758.8,4776.2,4793.8]

spec_feat 	= ['HeII']
lam1 		= [6400.]
lam2 		= [6410.]

#define region for extraction
# center = (88,58)
# radius = 4

center = (29,26)
radius = 4

for spec_feat,lam1,lam2 in zip(spec_feat,lam1,lam2):
	print spec_feat
	#--------------------------
	# LOAD spectral data cubes
	#--------------------------
	fname = "./out/"+spec_feat+"_cs.fits"
	cube 			= mpdo.Cube(fname)

	#---------------
	# LINE EMISSION
	#---------------
	#adjust the FOV
	# subcube 		= cube[:,:,20:150] 
	subcube = cube[:,60:110,50:100]			

	subcube_fname = './out/subcube_optim_cstretch_NB_imgs.fits'
	subcube.write(subcube_fname)

	cube = mpdo.Cube(subcube_fname)
	print cube

	#get WCS data from header
	hdu 			= fits.open(subcube_fname)[1]
	wcs 			= WCS(hdu.header).celestial

	pl.figure()
	cont_cube_ap 	= cube.subcube_circle_aperture(center=center,radius=radius,unit_center=None,unit_radius=None)
	spec = cont_cube_ap.sum(axis=(1,2))
	spec.plot(color='black',zorder=1)
	p1,p2 			= spec.wave.pixel([lam1,lam2], nearest=True)
	line_em 		= cube[p1:p2+1,:,:]
	
	#save spectrum with summed line emission shown
	Lya1 = [4732.5, 4747.5]
	Lya2 = [4754.3, 4758.8]
	Lya3 = [4768.8, 4776.2]
	Lya4 = [4778.8, 4793.8]

	ax = pl.gca()
	ax.yaxis.set_major_formatter( tk.FuncFormatter(lambda x,pos: '%d'%(x*1.e-2) ) )		
	pl.ylabel(r"Flux Density (10$^{-17}$ erg / s / cm$^2$ / $\AA$)")

	# if lam1 == 4768.8:
	# 	pl.axvspan(Lya3[0],Lya3[1],ymax=0.98,color='magenta',alpha=0.2)
	# 	pl.savefig('./out/narrow-band/Lya 3 profile.png')

	if spec_feat == 'Lya':
		pl.axvspan(Lya1[0],Lya1[1],ymax=0.98,color='blue',alpha=0.2)
		pl.axvspan(Lya2[0],Lya2[1],ymax=0.98,color='green',alpha=0.2)
		pl.axvspan(Lya3[0],Lya3[1],ymax=0.98,color='orange',alpha=0.2)
		pl.axvspan(Lya4[0],Lya4[1],ymax=0.98,color='red',alpha=0.2)
		pl.savefig('./out/narrow-band/Lya profile_og.png')

	else:
		pl.axvspan(lam1,lam2,ymax=0.98,color='magenta',alpha=0.2)
		pl.ylabel(r"Flux Density (10$^{-17}$ erg / s / cm$^2$ / $\AA$)")
		pl.title(spec_feat+r' integrated flux width ('+`int(lam1)`+'-'+`int(lam2)`+'$\AA$)')
		pl.savefig('./out/narrow-band/'+spec_feat+' profile.png') 
	
	line_em_img = line_em.sum(axis=0)
	
	#vmin,vmax auto-scale
	fig = pl.figure()
	pix = list(chain(*line_em_img.data))
	pix_rms = np.sqrt(np.mean(np.square(pix)))
	pix_med = np.median(pix)
	vmax = pix_med + pix_rms
	vmin = pix_med - 0.4*pix_rms
	print vmin,vmax
	
	#------------------
	# get VLA CONTOURS
	#------------------
	vla = fits.open('/Users/skolwa/DATA/VLA_DATA/0943C.ICLN')[0]
	wcs_vla = WCS(vla.header).celestial
	
	#define contour parameters
	radio_rms 		= 2.e-4
	
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
	
	# print contours
	# print start_level

	fig = pl.figure()
	fig.add_subplot(111,projection=wcs_vla)
	vla_arr = vla.data[0,0,:,:]

	# pl.contour(vla_arr,levels=contours,colors='white')
	# ax = pl.imshow(vla_arr,origin='lower',cmap='gist_gray',vmin=-50,vmax=1000)
	# pl.colorbar(ax,orientation = 'vertical')
	# pl.savefig("./out/narrow-band/VLA_cont_0943.png")
	# pl.show()
	
	# #save VLA .ICLN file in .fits format
	# fits.writeto('/Users/skolwa/DATA/VLA_DATA/0943C.fits',vla_arr,clobber=True)

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
	ax1 = fig.add_axes([0.15, 0.15, 0.8, 0.75], projection=wcs)
	line_em_arr = line_em_img.data[:,:]
	ax1.contour(vla_arr,levels=contours,colors='red',transform=ax1.get_transform(wcs_vla),linewidths=0.8)
	pl.annotate(s='', xy=(1.4,1.4), xytext=(15.5,1.4), arrowprops=dict(arrowstyle='->',ec='red'))		#horizontal
	pl.annotate(s='', xy=(15.,15.), xytext=(15.,0.8), arrowprops=dict(arrowstyle='->',ec='red'))		#vertical

	pl.annotate(s='', xy=(106.,1.), xytext=(132.,1.), arrowprops=dict(arrowstyle='<->',ec='red'))
	pl.text(110.,2.5,'40 kpc',fontsize=10,color='red')
	pl.text(13.5,15.,'N',color='red')
	pl.text(-0.5,0.,'E',color='red')
	pl.xlabel('RA (J2000)')
	pl.ylabel('DEC (J2000)')

	gs_img = ndimage.gaussian_filter(line_em_arr, sigma=(1, 1), order=0)
	figure = ax1.imshow(gs_img,cmap='gray_r',origin='lower',interpolation='nearest',\
		transform=ax1.get_transform(wcs),vmin=vmin,vmax=vmax)
	cb = pl.colorbar(figure, orientation = 'vertical')
	#check these
	cb.set_label(r'10$^{-20}$ erg/s/cm$^{2}$/pix$^2$',rotation=90)
	pl.title(spec_feat+r' M$_0$ map ('+`int(lam1)`+'-'+`int(lam2)`+r'$\AA$)')

	if spec_feat != 'Lya':
		figure = ax1.imshow(gs_img,cmap='gray_r',origin='lower',interpolation='nearest',\
			transform=ax1.get_transform(wcs),vmin=0.2*vmin,vmax=vmax)
		pl.savefig("./out/narrow-band/"+spec_feat+" merg. comp. grey VLA.pdf")
		pl.savefig("./out/narrow-band/"+spec_feat+" merg. comp. grey VLA.png")		
	elif lam1 == 4732.5:
		pl.savefig("./out/narrow-band/"+spec_feat+"1 grey VLA.png")
		pl.savefig("./out/narrow-band/"+spec_feat+"1 grey VLA.pdf")
	elif lam1 == 4754.3:
		pl.savefig("./out/narrow-band/"+spec_feat+"2 grey VLA.png")
		pl.savefig("./out/narrow-band/"+spec_feat+"2 grey VLA.pdf")
	elif lam1 == 4768.8:
		figure = ax1.imshow(gs_img,cmap='gray_r',origin='lower',interpolation='nearest',\
			transform=ax1.get_transform(wcs),vmin=vmin,vmax=0.2*vmax)
		pl.savefig("./out/narrow-band/"+spec_feat+"3 grey VLA.png")
		pl.savefig("./out/narrow-band/"+spec_feat+"3 grey VLA.pdf")
	elif lam1 == 4778.8:
		pl.savefig("./out/narrow-band/"+spec_feat+"4 grey VLA.png") 
		pl.savefig("./out/narrow-band/"+spec_feat+"4 grey VLA.pdf")
	
time_elapsed = time() - t
print "build time: %4.2f s" %time_elapsed	