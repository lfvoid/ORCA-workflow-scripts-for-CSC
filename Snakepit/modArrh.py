##########################################
# Python script for fitting MCTST rates to a modified Arrhenius function k(T) = Aexp(-Ea/T) * exp(B/T^3)
# where the second exponential function is only used for the tunneling correction.
# By Lauri Franzon
##########################################

import sys
import numpy as np
from numpy import log10,e,around,mean
from pandas import read_csv
from scipy.optimize import curve_fit

# Read k(T) data.
filename = str(sys.argv[1])
Compdata=read_csv(filename,sep='\t')
logkomp = log10(Compdata['Rate'])
lognotunnel = log10(Compdata['Rate']/Compdata['kappa']) #Rate coefficient without tunneling correction. 

# Define linear Arrhenius function
def logkfit(T,logA,EA):
    logk = logA - log10(e)*EA/T
    return logk

# Define function for tunneling correction
def logtunnel(T,B):
    logBT = log10(e)*B/T**3
    return logBT

# Fit k(T) data without tunneling correction to Arrhenius equation
params, cov = curve_fit(logkfit, Compdata['T'], lognotunnel)

parBT, covBT = curve_fit(logtunnel, Compdata['T'],logkomp - params[0] + log10(e)*params[1]/Compdata['T'])

print('Exact fit parameters:')
print('A',10**params[0])
print('Ea/K',params[1])
print('B',parBT[0])
print()

# Round fit parameters for convenience and see how well the fit does. 

scinot="{0:.2E}"
A=params[0]
Ea=around(params[1],0)
B=around(parBT[0],0)

#Calculate R^2 coefficient of rounded fit.

fit = 10**logkfit(Compdata['T'],A,Ea)*10**logtunnel(Compdata['T'],B)
ssr = sum((fit-Compdata['Rate'])**2)
ssd = sum((Compdata['Rate']-mean(Compdata['Rate']))**2)
R2 = 1 -ssr/ssd

print('Rounded fit parameters, k(T)= A exp(-Ea/T) exp(B/T^3) :')
print('A',scinot.format(10**A))
print('Ea/K',Ea)
print('B',scinot.format(B))
print()
print('R-squared of rounded fit')
print(R2)

