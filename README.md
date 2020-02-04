# radio_galaxy_phd_project-1

## Background
This repository contains several Python modules that were used to carry out the data wrangling, extraction, analysis and visualisation of an astronomical datacube. The datacubes I worked with in this project are structured as tensors consisting of pixel counts from the detector. 

The telescope used to obtain the data is the <a href="https://www.eso.org/public/teles-instr/paranal-observatory/vlt/">Very Large Telescope (the VLT)</a> located at the ESO Obsevatory in Paranal, Chile. The instrument from whence the datacubes came has a nice rockstar name - MUSE - which is an acronym that reads out fully as <a href="https://www.eso.org/sci/facilities/develop/instruments/muse.html">Multi-unit Spectroscopic Explorer</a>. The instrument is an integral field unit spectrograph that is capable of obtaining a 1D spectrum for every pixel in its field of view. The datacube produced is therefore three dimensional with the spatial co-ordinates mapped such that, (x,y,z) = (right-ascension, declination, redshift) as the image below displays [1]. 

<img src="muse_datacube.jpg" height="300">

The datacube imaged radio galaxy called with the catalogue name, <a href="https://ned.ipac.caltech.edu/byname?objname=MRC%200943-242&hconst=67.8&omegam=0.308&omegav=0.692&wmap=4&corr_z=1">MRC0943-242</a>a>. 


## Usage
The modules are very customised to my specific user case. However, they can be used as a template for carrying out similar astronomical data analysis where an optical datacube is being used. 

Python libraries used:
- For astronomy: MPDAF, AstroPy
- For linear algebra, calculus etc: SciPy, NumPy
- For visualization: Matplotlib
- Others: Warnings, Itertools


A report of the methodology and findings of this data analysis have been published in the journal, Astronomy & Astrophysics. The paper is freely available on the archive as well: https://arxiv.org/abs/1904.05114

[1] Credit: ESO Astronomy/J. Walsh