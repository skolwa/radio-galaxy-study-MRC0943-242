#S.N. Kolwa (2018)
#MRC0943_SiIV_comparison.py

# Purpose:  
# - Compare SiIV QSO systems (literature) to our 0943 MUSE results

import matplotlib.pyplot as pl
import numpy as np
from operator import itemgetter
from itertools import groupby

direct = '/Users/skolwa/PHD_WORK/catalogues/'

filenames = [ direct+'Songaila1998_abs.txt'] + [direct+'A'+`i`+'_Dodorico_2013_abs.txt' for i in range(1,4,1)] + [direct+'A'+`i`+'_Dodorico_2013_abs.txt' for i in range(5,7,1)]

params = {'legend.fontsize': 14,
          'legend.handlelength': 2}

pl.rcParams.update(params)
pl.rc('text', usetex=True)
pl.rc('font', **{'family':'monospace', 'monospace':['Computer Modern Typewriter']})

pl.figure(figsize=(10,8))
ax = pl.gca()
# N(CIV) vs N(SiIV)
data_0943 = np.genfromtxt('out/line-fitting/4_component_Lya_plus_blue_comp/0943_voigt_fit_0.7.txt',\
	dtype=[('spec_feat', 'S5'), ('wav0', np.float64), ('z_abs', np.float64), ('z_abs_err',np.float64),\
	 ('wav_abs', np.float64), ('b', np.float64), ('b_err', np.float64), ('N', np.float64)\
	 , ('N_err', np.float64), ('vel_abs', np.float64), ('vel_abs_err', np.float64), ('abs_no',np.float64)])

n = len(data_0943)

N_CIV 			= [ data_0943[i][7] for i in range(n) if data_0943[i][0] == 'CIV']
N_CIV_err 		= [ data_0943[i][8] for i in range(n) if data_0943[i][0] == 'CIV']
N_CIV_no 		= [ data_0943[i][11] for i in range(n) if data_0943[i][0] == 'CIV']
z_abs_CIV		= [ data_0943[i][2] for i in range(n) if data_0943[i][0] == 'CIV']
z_abs_err_CIV 	= [ data_0943[i][3] for i in range(n) if data_0943[i][0] == 'CIV']
logN_CIV 		= [ np.log10(N_CIV[i]) for i in range(len(N_CIV)) if  N_CIV_err[i]/N_CIV[i] < 5.]
logN_CIV_err 	= [ N_CIV_err[i]/N_CIV[i] for i in range(len(N_CIV)) if N_CIV_err[i]/N_CIV[i] <5.]
N_CIV_no 		= [ N_CIV_no[i] for i in range(len(N_CIV)) if N_CIV_err[i]/N_CIV[i] < 5.]
z_abs_CIV 		= [ [ z_abs_CIV[i], z_abs_err_CIV[i] ] for i in range(len(N_CIV)) if N_CIV_err[i]/N_CIV[i] < 5. ]

N_SiIV 			= [ data_0943[i][7] for i in range(n) if data_0943[i][0] == 'SiIV']
N_SiIV_err 		= [ data_0943[i][8] for i in range(n) if data_0943[i][0] == 'SiIV']
N_SiIV_no 		= [ data_0943[i][11] for i in range(n) if data_0943[i][0] == 'SiIV']
z_abs_SiIV		= [ data_0943[i][2] for i in range(n) if data_0943[i][0] == 'SiIV']
z_abs_err_SiIV 	= [ data_0943[i][3] for i in range(n) if data_0943[i][0] == 'SiIV']

logN_SiIV 		= [ np.log10(N_SiIV[i]) for i in range(len(N_SiIV)) if N_SiIV_err[i]/N_SiIV[i] < 5.]
logN_SiIV_err 	= [ N_SiIV_err[i]/N_SiIV[i] for i in range(len(N_SiIV)) if N_SiIV_err[i]/N_SiIV[i] < 5. ]
N_SiIV_no 		= [ N_SiIV_no[i] for i in range(len(N_SiIV)) if N_SiIV_err[i]/N_SiIV[i] < 5.]
z_abs_SiIV 		= [  [z_abs_SiIV[i], z_abs_err_SiIV[i]] for i in range(len(N_SiIV)) if N_SiIV_err[i]/N_SiIV[i] < 5. ]

pl.figure(figsize=(10,8))

ax = pl.gca()
n = len(N_CIV_no)
m = len(N_SiIV_no)

for i in range(0,n,2):
	for j in range(0,m,2):
		if N_CIV_no[i] == N_SiIV_no[j]:
			pl.scatter(logN_SiIV[j], logN_CIV[i], c='red', label='MRC 0943-242 (MUSE)' if i==2else "", s=60 )
			if N_CIV_no[i] == 1.1:
				pl.text( 1.004*logN_SiIV[j], 0.997*logN_CIV[i], `int(N_CIV_no[i])`, fontsize=16 )
			elif N_CIV_no[i] == 2.1:
				pl.text( 0.992*logN_SiIV[j], 0.997*logN_CIV[i], `int(N_CIV_no[i])`, fontsize=16 )
			else:
				pl.text( 1.004*logN_SiIV[j], 0.997*logN_CIV[i], `int(N_CIV_no[i])`, fontsize=16 )
			pl.errorbar(logN_SiIV[j], logN_CIV[i], xerr=logN_SiIV_err[j], yerr=logN_CIV_err[i], ecolor='red', capsize=5)
			pl.xlabel(r'$\log{\rm{N}_{SiIV}}$', fontsize=16)
			pl.ylabel(r'$\log{\rm{N}_{CIV}}$', fontsize=16)

ax.tick_params(direction='in', right=1, top=1)

for tick in ax.xaxis.get_major_ticks():
	   tick.label.set_fontsize(18)
	   tick.label.set_family('serif')

for tick in ax.yaxis.get_major_ticks():
	   tick.label.set_fontsize(18)
	   tick.label.set_family('serif')

# Songaila et al (1998) absorption
abs_data = np.genfromtxt( filenames[0],\
 dtype=[ ('m_z', np.float64), ('Quasar', 'S15'), ('no.', np.int64),\
  ('N(CIV)', np.float64), ('N(SiIV)', np.float64) ], usecols=(0,1,2,3,4) )
N_CIV 	= [ np.log10( abs(abs_data[i][3]) ) for i in range(len(abs_data)) ]
N_SiIV 	= [ np.log10( abs(abs_data[i][4]) ) for i in range(len(abs_data)) ]
ax.scatter( N_SiIV, N_CIV, s=40, c='#660022', marker='^', label='Songaila et al (1998)')

# D'Odorico et al (2013) absorption
for k in range(1,len(filenames),1):
	abs_data = np.genfromtxt( filenames[k],\
		dtype=[('System', '|S15'), 	
		('Ion','|S5'), ('z', '<f8'), 
		('z_err','<f8'),  
		('logN','<f8'),('logN_err', '<f8')], skip_header=2)
	
	sources = []
	for key,g in groupby( abs_data, lambda x: x[0] ):
		sources.append( list(g))
	
	sources_ = []
	
	for i in range(len(sources)):
		if len(sources[i]) >= 2.:
			sources_.append(sources[i])

	# only write legend label for first iteration of loop i.e. first data file
	if k == 1:
		for i in range(len(sources_)):
			ax.scatter( sources_[i][1][4], sources_[i][0][4], s=40, c='k', marker='s', label=r"D$'$Odorico et al (2013)" if i==1 else "")
			if sources_[i][1][3] == 0.:
				ax.errorbar (sources_[i][1][4], sources_[i][0][4], xerr= sources_[i][1][5], xuplims=True, yerr= sources_[i][0][5], capsize=5, c='k' )

			else:
				ax.errorbar (sources_[i][1][4], sources_[i][0][4], xerr= sources_[i][1][5], yerr= sources_[i][0][5], capsize=10, c='k')

	else:
		for i in range(len(sources_)):
			ax.scatter( sources_[i][1][4], sources_[i][0][4], s=40, c='k', marker='s')
			if sources_[i][1][3] == 0.:
				ax.errorbar (sources_[i][1][4], sources_[i][0][4], xerr= sources_[i][1][5], xuplims=True, yerr= sources_[i][0][5], capsize=5, c='k')

			else:
				ax.errorbar (sources_[i][1][4], sources_[i][0][4], xerr= sources_[i][1][5], yerr= sources_[i][0][5], capsize=10, c='k')

ax.legend()
pl.savefig('out/QSO_vs_HzRGs/SiIV_CIV_QSO_HzRGs.png')
pl.savefig('/Users/skolwa/PUBLICATIONS/0943_resonant_lines_letter/plots/SiIV_CIV_QSO_HzRGs.pdf')

# N(CIV)/N(SiIV) vs redshift

pl.figure(figsize=(10,8))
ax = pl.gca()

ax.tick_params(direction='in', right=1, top=1)

for tick in ax.xaxis.get_major_ticks():
	   tick.label.set_fontsize(18)
	   tick.label.set_family('serif')

for tick in ax.yaxis.get_major_ticks():
	   tick.label.set_fontsize(18)
	   tick.label.set_family('serif')

p = len(N_CIV_no)
q = len(N_SiIV_no)

for i in range(0,p,2):
	for j in range(0,q,2):
		if N_CIV_no[i] == N_SiIV_no[j]:
			z_abs = (z_abs_CIV[i][0] + z_abs_SiIV[j][0])/2. 			#average of absorber redshifts in SiIV and CIV
			z_abs_err = np.sqrt( (z_abs_CIV[i][1])**2 + (z_abs_SiIV[j][1])**2 )  
			ratio = logN_SiIV[j]/logN_CIV[i]
			pl.scatter( z_abs, ratio, c='red', label='MRC 0943-242 (MUSE)' if i==2 else "", s=60 )
			yerr = ratio*np.sqrt( (logN_CIV_err[i]/logN_CIV[i])**2 + (logN_SiIV_err[j]/logN_SiIV[j])**2 )
			pl.errorbar( z_abs, ratio, c='red', capsize=5, xerr=z_abs_err, yerr=yerr )

#plot ratio vs redshift from the literature

# Songaila et al (1998) absorption
abs_data = np.genfromtxt( filenames[0],\
 dtype=[ ('m_z', np.float64), ('Quasar', 'S15'), ('no.', np.int64),\
  ('N(CIV)', np.float64), ('N(SiIV)', np.float64) ], usecols=(0,1,2,3,4) )
ratio 	= [ np.log10( abs(abs_data[i][3]) )/np.log10( abs(abs_data[i][4]) ) for i in range(len(abs_data)) ]
z 		= [ abs_data[i][0] for i in range(len(abs_data)) ]
ax.scatter( z, ratio, s=40, c='#660022', marker='^', label='Songaila et al (1998)')
pl.xlabel(r'z', fontsize=16)
pl.ylabel(r'$\log{\rm{N}_{CIV}}/\log{\rm{N}_{SiIV}}$', fontsize=16)

# these absorbers are at very high redshifts
for k in range(1,len(filenames),1):
	abs_data = np.genfromtxt( filenames[k],\
		dtype=[('System', '|S15'), 	
		('Ion','|S5'), ('z', '<f8'), 
		('z_err','<f8'),  
		('logN','<f8'),('logN_err', '<f8')], skip_header=2)
	
	sources = []
	for key,g in groupby( abs_data, lambda x: x[0] ):
		sources.append( list(g))
	
	sources_ = []

	for i in range(len(sources)):
		if len(sources[i]) >= 2.:
			sources_.append(sources[i])

	if k == 1:
		for i in range(len(sources_)):
			ratio = sources_[i][1][4]/sources_[i][0][4] 	#N(SiIV)/N(CIV)
			z = sources_[i][0][2]
			z_err = ratio*np.sqrt( sources_[i][1][5]**2 + sources_[i][0][5]**2 )

			ax.scatter( z, ratio, s=40, c='k', marker='s', label=r"D$'$Odorico et al (2013)" if i==1 else "")
			ax.errorbar(z, ratio, xerr=None, yerr=z_err, c='k', capsize=5 )

	else:
		for i in range(len(sources_)):
			ratio = sources_[i][1][4]/sources_[i][0][4] 	#N(SiIV)/N(CIV)
			z = sources_[i][0][2]
			z_err = ratio*np.sqrt( sources_[i][1][5]**2 + sources_[i][0][5]**2 )

			ax.scatter( z, ratio, s=40, c='k', marker='s')
			ax.errorbar( z, ratio, xerr=None, yerr=z_err, c='k', capsize=5)

pl.legend()
pl.savefig('out/QSO_vs_HzRGs/SiIV_CIV_QSO_HzRGs_redshift.png')
pl.savefig('/Users/skolwa/PUBLICATIONS/0943_resonant_lines_letter/plots/SiIV_CIV_QSO_HzRGs_redshift.pdf')
