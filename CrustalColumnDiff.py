# -*- coding: utf-8 -*-
"""
Created on Sat Jan 17 13:41:10 2015

@author: kiransathaye
"""

import numpy as np
import scipy
import scipy.sparse.linalg as la
import matplotlib.pyplot as pl

def Geotherm(z):
    P=np.array([-.0625,16.298,8.979+273]); #define quadratic geotherm with depth
    z=z*1e-3; # convert m to km
    T=np.polyval(P,z);#output temperature
    return T;

def HeDiff(tmax,phi):
    theta=0; #theta method for implicit/explicit, theta=0=explicit

    tau=np.sqrt(2)**(-1); #crustal tortuosity factor
    years=365.24*24*60*60; # Convert years to seconds in no time!
    tmax=tmax*years; #end time for simulation
    DensityRaw=np.genfromtxt('CenterDense.csv',delimiter=','); #load crustal density profile from USArray
    UFraction=1.5e-6; # Uranium mass fraction in upper crust
    ThFraction=UFraction*3.9; #thorium mass fraction upper crust
    K2OFraction=1.88e-2; #K2O fraction in upper crust

    DensityRaw=DensityRaw[::2]; #only use every other density point to smooth
    numzgrid=100 #number of grid points

    zmax=45e3; #base of continental crust, 45km
    z=np.linspace(0,zmax,numzgrid); #set up grid for surface to base of crust
    Density=np.zeros([len(z),2]); # assign density
    Density[:,0]=z; #assign depth to density array
    Density[:,1]=np.interp(z,1e3*DensityRaw[:,0],DensityRaw[:,1]); # interpolate density points
    X=(-Density[:,1]+3.0)/0.3; # compute fraction of felsic crust

    X[X<0]=0; # set felsic fraction bewtween 0 and 1
    X[X>1]=1;

    Avocado=6.022e23; #mols to atoms

    dz=z[1]-z[0]; # depth discretization
    T=Geotherm(z); # temperature profile

    #set up radioactive elements with depth
    UContent=(UFraction*X+(1-X)*0.2e-6)*(Density[:,1]*1e6)/238; #mols/cubic meter
    ThContent=(ThFraction*X+(1-X)*1.2e-6)*(Density[:,1]*1e6)/232;
    #80% He4 trapped within
    UContent[T<180+273]=UContent[T<180+273]*0.2;
    ThContent[T<180+273]=ThContent[T<180+273]*0.2;

    lambda1 = np.log(2)/(4.468*1e9*years);#% decay rate of U in seconds
    lambda2 = np.log(2)/(1.405e10*years); # decay rate of Th in seconds
    lambda3 = np.log(2)/(703800000*years); #decay rate of 235U in seconds

    R=8.314; # gas constant J /(K mol)
    Nt = 20000; # number of timesteps
    dt = tmax/(Nt-1); # time discretization
    tvec = np.arange(0,tmax,dt); # set up timesteps vector

    #Arrhenius relation for He4 diffusion in water
    D0=1.37e-6;
    Ea=1.3e4;
    D=D0*np.exp(-Ea/(R*T)); # diffusivity varies with depth(Temp)

    # initialize arrays for He4 production with depth
    D1=np.zeros([len(tvec)-1,len(Density)]);  #Uranium 238
    D2=np.zeros([len(tvec)-1,len(Density)]); #Thorium 232
    D3=np.zeros([len(tvec)-1,len(Density)]);#Uranium 235

    for i in range(len(UContent)):
        #set He4 production from each isotope at depth for each timestep
        D1[:,i]=8*UContent[i]*(-np.exp(lambda1*tvec[0:-1])+np.exp(lambda1*tvec[1:]));
        D2[:,i]=6*ThContent[i]*(-np.exp(lambda2*tvec[0:-1])+np.exp(lambda2*tvec[1:]));
        D3[:,i]=7*UContent[i]*(-np.exp(lambda3*tvec[0:-1])+np.exp(lambda3*tvec[1:]))/137.88;


    gstep=D1+D2+D3; # total He4 production each timestep
    gstep=gstep[::-1]; # reverse for greatest production at t=0
    beta=np.ones(len(D))*dt*(D)*phi*tau/(2*dz**2); # beta operator incorporating constant values in PDE
    main=(1+2*beta*theta)*np.ones(len(z)); # main diagonal for Finite Difference
    sub=-theta*beta*np.ones(len(z)); # subdiagonal for central difference

    sub2=-theta*beta*np.ones(len(z)); # superdiagonal for central difference
    # boundary conditions
    sub2[-2]=-1;#-2*beta[-1]*theta;
    main[-1]=1;
    main[0]=1;
    sub[1]=0;

    opD  = scipy.sparse.spdiags([sub2, main, sub],[-1, 0, 1],len(z),len(z)); #sparse operator matrix
    C=np.zeros(len(z)); # concentration vector
    k=1; # iteration variable
    RHS=np.zeros(len(C)); # right hand side of PDE
    TotalProd=0;
    F=np.zeros(len(tvec)); # fraction lost to atm
    Out=np.zeros([len(C),2]);
    for k in range(Nt-2):
        da=gstep[k,:]; # He4 Production term
        RHS[1:-1]=C[1:-1]+da[1:-1]+(1-theta)*beta[1:-1]*(C[0:-2]-2*C[1:-1]+C[2:]); #update right side with new production term
        C = la.dsolve.spsolve(opD, RHS, use_umfpack=False); # solve system with sparse solver
        TotalProd=TotalProd+np.sum(da); # sum total He4 in crustal column
        F[k]=np.sum(C)/TotalProd; #compute fraction lost to atm
        C[0]=0;

    Out[:,0]=Density[:,0]; #store output depth
    Out[:,1]=C; #store output concentration
    return Out;

def ArgonDiff(tmax,phi):

    theta=0.1;  #theta method for implicit/explicit, theta=0=explicit

    tau=np.sqrt(2)**(-1);#crustal tortuosity factor
    years=365.24*24*60*60; # Convert years to seconds in no time!
    tmax=tmax*years;#end time for simulation
    DensityRaw=np.genfromtxt('CenterDense.csv',delimiter=','); #load crustal density from seismic profile
    K2OFraction=1.88e-2; #set upper crust K2O fraction

    DensityRaw=DensityRaw[::2]; #every other density point
    numzgrid=1000#number of z grid points

    zmax=45e3; # base of crust 45km
    z=np.linspace(0,zmax,numzgrid); #set up vector of depths
    Density=np.zeros([len(z),2]); #initialize density array
    Density[:,0]=z; # depth as first col
    Density[:,1]=np.interp(z,1e3*DensityRaw[:,0],DensityRaw[:,1]); #interpolate density from seismic into z array

    X=(-Density[:,1]+3.0)/0.3; # felsic fraction

    X[X<0]=0; #set between 0 and 1
    X[X>1]=1;

    KContent=120e-6*2*(K2OFraction*X+(1-X)*0.6e-6)*(Density[:,1]*1e6)/94; #mols/CM amoung ot potassium40 with depth at present day

    R=8.314;# gas constant
    Nt = 30000; # number timesteps
    dt = tmax/(Nt-1); #delta t for simulation
    tvec = np.arange(0,tmax,dt); #vector of timesteps

    lambdaK40=np.log(2)/(years*1.248e9); #decay constant K40 in seconds

    dz=z[1]-z[0]; # depth discretization
    T=Geotherm(z); # temperature with depth
    KContent[T<260+273]=KContent[T<260+273]*0; # no Argon released in shallow regions below closure temperature

    #set up Arrhenius law from experimental measurements of Ar diffusion in water
    #linear relationship between 1/T and ln(D)
    TAr=np.array([273,278,283,288,293,298,303,308.0]);
    DAr=np.array([.72,.85,.97,1.09,1.23,1.44,1.65,1.82])*1e-9;
    InvT=TAr**-1;
    lnDAr=np.log(DAr);
    mxb=np.polyfit(InvT,lnDAr,1);
    Ea=-mxb[0]*R;
    D0=np.exp(mxb[1]);
    D=D0*np.exp(-Ea/(R*T));

    D1=np.zeros([len(tvec)-1,len(Density)]); # initialize Argon40 production vector
    for i in range(len(KContent)):
        D1[:,i]=0.1072*KContent[i]*(-np.exp(lambdaK40*tvec[0:-1])+np.exp(lambdaK40*tvec[1:]));
    # set up production with depth for each time
    gstep=D1;#total Argon production
    gstep=gstep[::-1]; # production decreases with time

    beta=np.ones(len(D))*dt*(D)*phi*tau/(2*dz**2); #constant PDE terms
    main=(1+2*beta*theta)*np.ones(len(z)); #main diagonal terms
    sub=-theta*beta*np.ones(len(z)); #subdiagonal terms (central difference)

    sub2=-theta*beta*np.ones(len(z));
    #Boundary conditions
    sub2[-2]=-1;
    main[-1]=1;
    main[0]=1;
    sub[1]=0;

    opD  = scipy.sparse.spdiags([sub2, main, sub],[-1, 0, 1],len(z),len(z)); #solve sparse matrix problem
    C=np.zeros(len(z)); # intialize concentration vector
    k=1; # timestep counter
    RHS=np.zeros(len(C)); # initialize right hand side vector
    TotalProd=0; # sum of production across column
    F=np.zeros(len(tvec)); # fraction vented
    Out=np.zeros([len(C),2]); # initialize output array

    for k in range(Nt-2):
        da=gstep[k,:]; #production at time k
        RHS[1:-1]=C[1:-1]+da[1:-1]+(1-theta)*beta[1:-1]*(C[0:-2]-2*C[1:-1]+C[2:]);#right side for matrix solve
        C = la.dsolve.spsolve(opD, RHS, use_umfpack=False); #solve sparse matrix
        TotalProd=TotalProd+np.sum(da); #total production in column
        F[k]=np.sum(C)/TotalProd; #fraction retained in crust
        #C[0]=0;

    Out[:,0]=Density[:,0];
    Out[:,1]=C; # set up output variable
    return Out;

