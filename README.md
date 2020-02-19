# radio_galaxy_phd_project-1

## Background
This repository contains several Python modules that were used to carry out the data wrangling, extraction, analysis and visualisation of an astronomical datacube which is structured as a tensor containing pixel counts from the telescope's detector. 

For this project, the data was acquired using the <a href="https://www.eso.org/public/teles-instr/paranal-observatory/vlt/">Very Large Telescope (VLT)</a> which is operated by the European Southern Obsevatory (ESO) and located in Paranal, Chile. The instrument whence the datacubes come has a nice rockstar name - MUSE - which is an acronym for the <a href="https://www.eso.org/sci/facilities/develop/instruments/muse.html">Multi-unit Spectroscopic Explorer</a>. MUSE is an integral field unit spectrograph that is capable of obtaining a 1D spectrum for every pixel in the field of view imaged by the telescope. 

The datacube is a three dimensional tensor with the co-ordinates mapped in the cartesian plane such that (x,y,z) represent (right-ascension, declination, wavelength). This is shown in the figure below which illustrates how a MUSE datacube is a series of 2D images obtained at different wavelengths. The image also shows the wavelengths of common spectral lines such as [OIII], HeII, HeI and H&#x03B1;.

<p align="center">
<img src="muse_datacube.jpg" height="300x" class="center">
</p>

## The Project

In this project, datacube image was that of radio galaxy named <a href="https://ned.ipac.caltech.edu/byname?objname=MRC%200943-242&hconst=67.8&omegam=0.308&omegav=0.692&wmap=4&corr_z=1">MRC0943-242</a> and the projected area of approximately 500 kpc x 500 kpc around it. The optical data permitted us to constrain the kinematics, mass and structure of ionised gas surrounding the galaxy.


## Usage
The modules are very customised to my specific use-case. However, they can be used as a template for carrying out similar astronomical data analysis where an optical datacube is being used. 

Python libraries used:
- For astronomy: MPDAF, AstroPy, WCS
- For linear algebra, calculus etc: SciPy, NumPy
- For visualization: Matplotlib
- For model fitting: LMFIT
- Others: Warnings, Itertools, Math


A report of the methodology and findings of this data analysis have been published in the journal, Astronomy & Astrophysics. The paper is freely available on the archive as well: https://arxiv.org/abs/1904.05114

[1] Credit: ESO Astronomy/J. Walsh