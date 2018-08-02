#S.N. Kolwa (2018)
#MRC0943_absorber_velocity_diag.py

# Purpose:  
# - Draw absorber velocities
# - Colour-code according to column density
# - Indicate L.O.S. overlaps in absorption from different ions

import numpy as np
from matplotlib import pyplot as pl
import matplotlib.ticker as tk

params = {'legend.fontsize': 14,
          'legend.handlelength': 2}

pl.rcParams.update(params)

pl.rc('font', **{'family':'monospace', 'monospace':['Computer Modern Typewriter']})
pl.rc('text', usetex=True)

data = np.genfromtxt('out/line-fitting/4_component_Lya_plus_blue_comp/0943_voigt_fit_0.7.txt',\
	dtype=[('spec_feat', 'S5'), ('wav0', np.float64), ('z_abs', np.float64), ('z_abs_err', np.float64),\
	 ('wav_abs', np.float64), ('b', np.float64), ('b_err', np.float64), ('N', np.float64)\
	 , ('N_err', np.float64), ('vel_abs', np.float64), ('vel_abs_err', np.float64), ('abs_no', np.float64)])

N = len(data)

vel_abs_Lya 	= [ data[i][9] for i in range(N) if data[i][0] == 'Lya']
vel_abs_Lya_err = [ data[i][10] for i in range(N) if data[i][0] == 'Lya']
abs_no_Lya 		= [ data[i][11] for i in range(N) if data[i][0] == 'Lya' ]

vel_abs_NV 		= [ data[i][9] for i in range(N) if data[i][0] == 'NV']
vel_abs_NV_err 	= [ data[i][10] for i in range(N) if data[i][0] == 'NV']
abs_no_NV 		= [ data[i][11] for i in range(N) if data[i][0] == 'NV' ]

vel_abs_CIV 		= [ data[i][9] for i in range(N) if data[i][0] == 'CIV']
vel_abs_CIV_err 	= [ data[i][10] for i in range(N) if data[i][0] == 'CIV']
abs_no_CIV 			= [ data[i][11] for i in range(N) if data[i][0] == 'CIV' ]

#draw plot
pl.figure(figsize=(6,8))
ax = pl.gca()

pos = 1
for i in range(0,len(vel_abs_Lya),1):
	pos += 3
	pl.scatter(vel_abs_Lya[i], pos,s=50, facecolor='red', edgecolor='black',\
		label=r'Ly$\alpha$' if i==1 else "")
	pl.errorbar(vel_abs_Lya[i], pos, xerr=vel_abs_Lya_err[i], capsize=8, ecolor='red',zorder=-1)
	pl.text(vel_abs_Lya[i]-50., pos+0.3, `int(abs_no_Lya[i])`, family='serif')

pos = 2
for i in range(0,len(vel_abs_NV),2):
	pos += 3
	pl.scatter(vel_abs_NV[i], pos,s=50, facecolor='green', edgecolor='black',\
		label='NV' if i==0 else "")
	pl.errorbar(vel_abs_NV[i], pos, xerr=vel_abs_NV_err[i], capsize=8, ecolor='green',zorder=-1)
	pl.text(vel_abs_NV[i]-50., pos+0.3, `int(abs_no_NV[i])`, family='serif')

pos = 3
for i in range(1,len(vel_abs_CIV),2):
	pos += 3
	pl.scatter(vel_abs_CIV[i], pos,s=50, facecolor='blue', edgecolor='black',\
		label='CIV' if i==1 else "")
	pl.errorbar(vel_abs_CIV[i], pos, xerr=vel_abs_CIV_err[i], capsize=8, ecolor='blue',zorder=-1)
	pl.text(vel_abs_CIV[i]-50., pos+0.3, `int(abs_no_CIV[i])`, family='serif')

for tick in ax.xaxis.get_major_ticks():
	tick.label.set_fontsize(14)

xticks = tk.FuncFormatter( lambda x,pos: '%.0f'%x ) 
ax.xaxis.set_major_formatter(xticks) 
pl.plot([0.,0.],[3.,16.],ls='-.', c='k',lw=0.8)
pl.legend(loc='best')
pl.xlim(-3000, 3000)
pl.ylim(3,16)
pl.tick_params(
	axis='y',
	left=False, 
	labelleft=False)
# pl.title('Absorber Velocities', family='serif')
pl.xlabel(r'$\Delta$v (km/s)', family='serif', fontsize=14)
pl.savefig('./out/line-fitting/4_component_Lya_plus_blue_comp/absorber_velocities.png')
pl.savefig('/Users/skolwa/PUBLICATIONS/0943_resonant_lines_letter/plots/absorber_velocities.pdf')
pl.show()