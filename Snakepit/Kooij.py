##########################################
# Python script for fitting MCTST rates to a Kooij function.
# By Lauri Franzon
##########################################

import sys
import numpy as np
from pandas import read_csv
from scipy.optimize import curve_fit

# Read k(T) data.
filename = str(sys.argv[1])
Compdata=read_csv(filename,sep='\t')
logkomp = np.log10(Compdata['Rate'])

# Define Kooij function
def logkfit(T,logA,n,EA):
    logk = logA + n*np.log10(T) -np.log10(np.e)*EA/T
    return logk

# Fit k(T) data to function.
params, cov = curve_fit(logkfit, Compdata['T'], logkomp)

print('Exact fit parameters:')
print('A',10**params[0])
print('n',params[1])
print('Ea/K',params[2])
print()

# Round fit parameters for convenience and see how well the fit does. 

scinot="{0:.2E}"
A=params[0]
n=np.around(params[1],2)
Ea=np.around(params[2],0)

#Calculate R^2 coefficient of rounded fit.

fit = 10**logkfit(Compdata['T'],A,n,Ea)
ssr = sum((fit-Compdata['Rate'])**2)
ssd = sum((Compdata['Rate']-np.mean(Compdata['Rate']))**2)
R2 = 1 -ssr/ssd

print('Rounded fit parameters:')
print('A',scinot.format(10**A))
print('n',n)
print('Ea/K',Ea)
print()
print('R-squared of rounded fit')
print(R2)

