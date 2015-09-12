# -*- coding: utf-8 -*-
"""
Created on Thu Feb 12 19:46:18 2015

@author: kiransathaye
"""

# import packages
import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import CrustalColumnDiff as CCD # contains functions to run simulation
CCD=reload(CCD);

tmax=1.5e9; # run simulation for 1.5Ga (~crust age in NE New Mexico)
phi=[0,1e-2]; # run for two separate porosities
FracLost=np.zeros(2); #compute fraction vented to atm
#%% Run helium diffusion

He1=CCD.HeDiff(1.5e9,phi[1]); #run Helium with 1% porosity
He0=CCD.HeDiff(1.5e9,phi[0]); # run helium with 0 porosity ( no diffusion)
FracLost[0]=1-np.sum(He1[:,1])/np.sum(He0[:,1]); #compute fraction vented
np.savetxt('He1Percent.csv',He1,delimiter=','); # save output
np.savetxt('He0Percent.csv',He0,delimiter=',');

#%% Run Argon Diffusion
CCD=reload(CCD); #load package

Ar1=CCD.ArgonDiff(1.5e9,phi[1]); #run Argon diffusion for 1% porosity
Ar0=CCD.ArgonDiff(1.5e9,phi[0]);#run Argon diffusion for 0% porosity
FracLost[1]=1-np.sum(Ar1[:,1])/np.sum(Ar0[:,1]); # compute fraction of Argon lost

#%% Plotting
plt.close('all')

plt.subplot(1,2,1)
plt.plot(He1[:,1]*1e3,He1[:,0]*1e-3,c='red',lw=2) #plot 1% porosity case
plt.plot(He0[:,1]*1e3,He0[:,0]*1e-3,c='b',lw=2) #plot pure production case
blue_line = mlines.Line2D([], [], color='blue',label='$\phi$=0',lw=2);#set up legend
red_line = mlines.Line2D([], [], color='red',label='$\phi$=1%',lw=2);
plt.legend(handles=[blue_line,red_line],loc=6); #plot legend
plt.xlabel('$^4$He (10$^{-3}$ mol/m$^3$)');#xlabel with units
plt.ylabel('Depth (km)'); #ylabel of depth in km
plt.xticks([0,20,40,60]);#Xtick locations
plt.gca().invert_yaxis()

plt.subplot(1,2,2)
plt.plot(Ar1[:,1]*1e3,Ar1[:,0]*1e-3,c='red',lw=2); #plot 1% porosity Argon case
plt.plot(Ar0[:,1]*1e3,Ar0[:,0]*1e-3,c='b',lw=2); # plot pure production case
blue_line = mlines.Line2D([], [], color='blue',label='$\phi$=0',lw=2);#set up legend
red_line = mlines.Line2D([], [], color='red',label='$\phi$=1%',lw=2);
plt.legend(handles=[blue_line,red_line],loc=1);#plot legend
plt.xlabel('$^{40}$Ar (10$^{-3}$ mol/m$^3$)'); #plot x axis with units
plt.yticks([0,5,10,15,20,25,30,35,40,45],[]);#remove ytick labels
plt.xticks([0,4,8,12]); #set x ticks
plt.gca().invert_yaxis()

plt.savefig('ArHeDiff.pdf',format='pdf'); #save plot to PDF