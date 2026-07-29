##########################################
# Python script for calculating MC-TST rates with ALL the T-dependent factors considered.
# By Lauri Franzon
##########################################

import sys
from pandas import read_csv

###########
# READ DATA
###########

Trbool = str(sys.argv[1])                             # Boolean for choosing either a T-range or a single Temperature. WORK IN PROGRESS.
Tlow= float(sys.argv[2])                              # Lower Temperature bound.
Thigh= float(sys.argv[3])                             # Upper Temperature bound.
R1CCC = float(sys.argv[4])-float(sys.argv[5])         # Coupled-Cluster energy correction for reactant 1
R2CCC = float(sys.argv[6])-float(sys.argv[7])         # Coupled-Cluster energy correction for reactant 2
TSCCC = float(sys.argv[8])-float(sys.argv[9])         # Coupled-Cluster energy correction for transition state
R1mass = float(sys.argv[10])                          # The mass of reactant 1 in amu
R2mass = float(sys.argv[11])                          # The mass of reactant 2 in amu
PCCC = float(sys.argv[12])-float(sys.argv[13])        # Coupled-Cluster energy correction for the IRC Product.
IRCCC = float(sys.argv[14])-float(sys.argv[15])       # Coupled-Cluster energy correction the IRC Reactant conformer.
Deg = float(sys.argv[16])                             # Reaction path degeneracy to multiply the final rate with.
tunnelswitch = str(sys.argv[17])                      # Boolean to switch off tunneling if so desired.
gswitch = str(sys.argv[18])                           # "G" if DFT calculations were done with Gaussian.

# Read the information on the R, TS & P conformers.
R1c = read_csv('Reactant1_conformers.txt',sep=' ')
R2c = read_csv('Reactant2_conformers.txt',sep=' ')
TSc = read_csv('TS_conformers.txt',sep=' ')
if tunnelswitch=='y':
    Pc = read_csv('IRC_Product_conformer.txt',sep=' ')
    IRc = read_csv('IRC_Reactant_conformer.txt',sep=' ')


# Import relevant math and physical constants
from numpy import empty,pi,sqrt,arange,exp,log,abs,max,sum,round,cosh
from scipy.constants import h,c,k,u,m_e,N_A,e,physical_constants
eh=physical_constants['Hartree energy'][0]
amu=physical_constants["atomic mass constant"][0]

print('Data loaded.')

kha = k/eh                                         # Boltzmann constant in Hartree
coef=100*h*c/k                                     # Conversion factor between the vibrational temperature and wavenumber.
ea0=e*c*physical_constants['Bohr radius'][0]*1e+21 # Conversion factor from Debye to atomic units
mu=(R1mass*R2mass)/(R1mass+R2mass)*u/m_e           # The reduced mass in atomic units
# Conversion factor of LRTST rate coefficient from atomic units to cm^3/s
autocm3=1e+6 * physical_constants['Bohr radius'][0]**3 / physical_constants['atomic unit of time'][0]

if gswitch == "G":
    R1c.A=R1c.A*1e+7/c
    R1c.B=R1c.B*1e+7/c
    R1c.C=R1c.C*1e+7/c
    R2c.A=R2c.A*1e+7/c
    R2c.B=R2c.B*1e+7/c
    R2c.C=R2c.C*1e+7/c
    TSc.A=TSc.A*1e+7/c
    TSc.B=TSc.B*1e+7/c
    TSc.C=TSc.C*1e+7/c

##################################
# DEFINE FUNCTIONS
##################################

# Long-range transition state theory rate coefficient for dipole-dipole interactions.
def LRTST(dip1,dip2,T):
    kau = 5.42*mu**(-1/2) * dip1*dip2**(2/3)*ea0**(-4/3) * (kha*T)**(-1/6)
    return kau*autocm3

# Harmonic vibrational contribution to kT ln Q
def lnQv(vib,T):
    return -kha*T*log(1-exp(-coef*vib/T))

def lnQr(s,Ar,Br,Cr,T):  # symmetry number, rotational constants, Temperature
    return kha*T*log((T/coef)**1.5*sqrt(pi/(Ar*Br*Cr))/s)

# Grimme's Quasi-Harmonic intrapolation kT ln Q. In case it is ever wanted.
def lnQHRO(vib,Ar,Br,Cr,T):
    tau=25
    weight=1/(1+(tau/vib)**4)
    muR = Ar*Br*Cr/(Ar*Br+Ar*Cr+Br*Cr)    # Reduced rotational constant
    return lnQv(vib,T)*weight+kha*T*log(sqrt(T*pi/(coef*(3*muR+vib))))*(1-weight)

# Eckart tunneling probability as a function of energy.
def EckartE(res):

    # Setting range of energies. 'res' stands for energy resolution. Function is rerun with a better one if convergence isn't reaced.
    E = arange(Emin,2*max(V),res*max(V)/50)
    Tp=[0 for x in range(len(E))]
    C=pi**2/(2*ume*L**2)     # calculation of C factor
    for i in range(len(E)):
        # Calculate the tunneling probability as a function of energy
        a=0.5*sqrt(E[i]/C)
        b=0.5*sqrt((E[i]-A)/C)
        d=0.5*sqrt((B-C)/C)
        # Tp for "Tunneling probability". Really just the probability of crossing the barrier, as scattering effects above max(V) are included. 
        Tp[i]=1-((cosh(2*pi*(a-b))+cosh(2*pi*d))/(cosh(2*pi*(a+b))+cosh(2*pi*d)))

    return Tp,E

# Function that calculates T-dependent tunneling coefficients from E-dependent Eckart tunneling probabilities.
def EckartT(Tp,T):

    # Define the integrand: A Boltzmann-weighted tunneling probability.
    Integrand=[0 for x in range(len(Tp))]
    for j in range(len(Tp)):
        Integrand[j]=Tp[j]*exp(-E[j]/(kha*T))
    GI=[0 for x in range(len(Tp))]
    for l in range(len(Tp)-1):
        GI[l]=(0.5*(Integrand[l]+Integrand[l+1])*abs(E[l]-E[l+1])) # Numerical integration by using the area of squares

    # Tunneling coefficient: QM tunneling probability divided by classical tunneling probability.
    # Final term corrects for the fact that our energy scale doesn't extent to infinity.
    tc=sum(GI)*exp(v1/(kha*T))/(kha*T)+exp(v1/(kha*T))*exp(-E[-1]/(kha*T))
    return tc


#################################################################
# DETERMINE GLOBAL MINIMUM CONFORMERS OF R & TS
#################################################################

# Recalculate Gibbs free energy.
# NOTE 1: We are neglecting translational thermal contributions, because they only matter when the net number of particles change -> No difference between any R, TS or P conformer.
# NOTE 2: We are also neglecting the electronic thermal contribution, because that only matters when R and P have different spin multiplets.

# Determine global minimum conformer at 298.0 K, assume it doesn't change, and treat it as a reference

# Correct Gibbs free energy for reactant conformers
for r in range(len(R1c)):
    # Rotational kT ln Q contribution. Assume sigma = 1
    rotlnQ = lnQr(1,R1c.iloc[r]['A'],R1c.iloc[r]['B'],R1c.iloc[r]['C'],298.0)
    # Vibrational kT ln Q contribution. Loop over vibrational frequencies
    viblnQ=0.0
    for v in range(8,R1c.shape[1]):
        viblnQ=viblnQ+lnQv(R1c.iloc[r,v],298.0)
    # Replace Gibbs free energy (round to 8 decimals for the output version)
    R1c.iloc[r,3]=round(R1c.iloc[r]['Energy']+R1c.iloc[r]['ZPE']+kha*298-rotlnQ-viblnQ,8)

for r in range(len(R2c)):
    # Rotational kT ln Q contribution. Assume sigma = 1
    rotlnQ = lnQr(1,R2c.iloc[r]['A'],R2c.iloc[r]['B'],R2c.iloc[r]['C'],298.0)
    # Vibrational kT ln Q contribution. Loop over vibrational frequencies
    viblnQ=0.0
    for v in range(8,R2c.shape[1]):
        viblnQ=viblnQ+lnQv(R2c.iloc[r,v],298.0)
    # Replace Gibbs free energy (round to 8 decimals for the output version)
    R2c.iloc[r,3]=round(R2c.iloc[r]['Energy']+R2c.iloc[r]['ZPE']+kha*298-rotlnQ-viblnQ,8)

# Correct Gibbs free energy for TS conformers. Ignore imaginary frequency.
for t in range(len(TSc)):
    # Rotational kT ln Q contribution. Assume sigma = 1
    rotlnQ = lnQr(1,TSc.iloc[t]['A'],TSc.iloc[t]['B'],TSc.iloc[t]['C'],298.0)
    # Vibrational kT ln Q contribution. Loop over vibrational frequencies
    viblnQ=0.0
    for v in range(9,TSc.shape[1]):
        viblnQ=viblnQ+lnQv(TSc.iloc[t,v],298.0)
    # Replace Gibbs free energy (round to 8 decimals for the output version)
    TSc.iloc[t,3]=round(TSc.iloc[t]['Energy']+TSc.iloc[t]['ZPE']+kha*298-rotlnQ-viblnQ,8)

# Find the global minimum conformers.

R1min=R1c['Gibbs'].argmin()
R1Emin=(R1c['Energy']+R1c['ZPE']).argmin()             # Only for printing, global minimum in terms of E is separated from global minimum in terms of G.
R1minE=R1c['Energy'].loc[R1min]+R1c['ZPE'].loc[R1min]
R1EminE=R1c['Energy'].loc[R1Emin]+R1c['ZPE'].loc[R1Emin]
R1minG=R1c['Gibbs'].min()

R2min=R2c['Gibbs'].argmin()
R2Emin=(R2c['Energy']+R2c['ZPE']).argmin()             # Only for printing, global minimum in terms of E is separated from global minimum in terms of G.
R2minE=R2c['Energy'].loc[R2min]+R2c['ZPE'].loc[R2min]
R2EminE=R2c['Energy'].loc[R2Emin]+R2c['ZPE'].loc[R2Emin]
R2minG=R2c['Gibbs'].min()

TSmin=TSc['Gibbs'].argmin()
TSEmin=(TSc['Energy']+TSc['ZPE']).argmin()             # Only for printing, global minimum in terms of E is separated from global minimum in terms of G.
TSminE=TSc['Energy'].loc[TSmin]+TSc['ZPE'].loc[TSmin]
TSEminE=TSc['Energy'].loc[TSEmin]+TSc['ZPE'].loc[TSEmin]
TSminG=TSc['Gibbs'].min()

# Overwrite conformer files using the correct Gibbs free energies
#R1c.to_csv('Reactant1_conformers.txt', sep=' ', index=False)
#R2c.to_csv('Reactant2_conformers.txt', sep=' ', index=False)
#TSc.to_csv('TS_conformers.txt', sep=' ', index=False)

# Note: The IRC reactant & product entropies are not recalculated and rewritten, because only the electronic & zero-point energies matter for these.

print('Global minimum conformers found.')

##############################################################################################################################
# CALCULATE ECKART TUNNELING COEFFICIENT FOR THE LOWEST TEMPERATURE WHILE VARYING THE ENERGY RESOLUTION OF TUNNELING INTEGRAL.
##############################################################################################################################

# Input for the Eckart tunneling calculation: Coupled-cluster-corrected zero-point energies and imaginary frequency of global minimum TS and the connected reactant and product conformers.
imfreq = TSc['v1'].loc[TSmin]

# T-range.
T=arange(Tlow,Thigh+1,1)
kappa=empty(len(T))

if tunnelswitch=='y': # If flag to neglect tunneling isn't set, calculate tunneling effects.

    # Use ZPE-corrected CCSD(T) electronic energies if available and DFT energies otherwise.
    if IRCCC==0 and PCCC==0:
        RE = IRc['Energy'].loc[0]+IRc['ZPE'].loc[0]
        TSE = TSminE
        PE = Pc['Energy'].loc[0]+Pc['ZPE'].loc[0]
    else:
        RE = IRc['Energy'].loc[0]+IRc['ZPE'].loc[0] + IRCCC
        TSE = TSminE + TSCCC
        PE = Pc['Energy'].loc[0]+Pc['ZPE'].loc[0] + PCCC 

    # T-range.
    kapdiff=1                                      # initial convergence control set to 1
    res=1

    # Set up Potential Energy curve
    # General note: This code is copied from a version that includes the mass as an input parameter, despite the fact that it does not directly impact
    # the tunneling probability. He we are using a reduced mass of mu=1 and converting it to atomic units with m_proton/m_electron whenever masses show up.

    # Forward and backward barriers.
    v1 = TSE-RE
    v2 = TSE-PE

    wau=(imfreq*100)*c*2.418884326509e-17  # Conversion of imaginary frequency to atomic units. 
    ume=u/m_e  # The afforementioned dummy mass.

    #Eckart's parameters
    F=-4*(pi**2)*(wau**2)*ume
    A=v1-v2
    B=(sqrt(v2)+sqrt(v1))**2
    L=-pi*(A-B)*(B+A)/(sqrt(-2*F*B)*B)

    # Generating reaction coordinate
    x = arange(-3, 3, 0.01)

    # Generating potential barrier V(x) using the Eckart parameters
    y=[0 for i in range(len(x))]
    V=[0 for i in range(len(x))]
    for i in range(len(x)):
        y[i]=-exp( (2*pi*x[i])/L )
        V[i]=( (-(y[i]*A)/(1-y[i]) ) - ( (y[i]*B)/((1-y[i])**2)) )

    # Determining the minimum energy at which tunneling can occur.
    
    if RE>PE: 
        # Exothermic reaction: Tunneling possible at any energy. Emin is the reactant ZPE.
        Emin = V[0]
    else:
        # Endothermic reaction: Tunneling is only possible for energies above the product ZPE. Emin set to Product ZPE.
        Emin = V[-1]
    # For thermoneutral reactions, both sides of the if-else should return the same value.


    # For the lowest temperature, rerun the tunneling calculation until you reach good numerical precision
    [Tp,E]=EckartE(res)                            # Calculate E-dependent tunneling probabilities
    kappaOld=EckartT(Tp,T[0])                      # Calculate T-dependent tunneling factor for the lowest temperature
    runs=0
    while kapdiff >= 0.001:                        # Convergence criteria
        res=res/10                                 # Increase resolution
        [Tp,E]=EckartE(res)
        kappaNew=EckartT(Tp,T[0])                  # Recalculate tunneling coefficient 
        kapdiff=abs(kappaNew-kappaOld)/kappaOld    # Calculate relative difference between iterations.
        kappaOld=kappaNew
        runs=runs+1

    # Once good numerical precision is reached, we use kappaNew as our low-T tunneling coefficient.
    kappa[0]=kappaNew
    print('Numerical Eckart tunneling integral converged.')

else:
    kappa.fill(1.0)
    print('Eckart tunneling neglected.')

###########################################################
# LOOP OVER THE TEMPERATURE AND CALCULATE THE MC-TST RATES:
# THE LOWEST TEMPERATURE IS CALCUALTED OUTSIDE THE LOOP SINCE WE ALREADY 
###########################################################

# Translational component of the partition function for bimolecular reactions Q_TS /(Q_R1 * Q_R2) =
# V^(-1) (2pi kT)^(-3/2) h^3 * f(m1,m2) = (2pi)^(-3/2) p (kT)^(-5/2) h^3 f(m1,m2) where f(m1,m2) = (m1+m2)^(3/2)/(m1^(3/2)m2^(3/2))
# When divided by the ideal gas reference concentration p/kT we get (2pikT)^(-3/2) h^3 f(m1,m2).
# And this then gets multiplied by kT/h, resulting in (2pi)^(-3/2) h^2/sqrt(kT) f(m1,m2). Final factor of 1e+6 is unit conversion from m^3 to cm^3
tf=empty(len(T))
tf[:] = (2*pi)**-1.5 / sqrt(T[0]) * (R1mass+R2mass)**1.5 * R1mass**-1.5 * R2mass**-1.5
tf[:] = tf[:] * amu**-1.5 * h**2 * 1e+6/sqrt(k)

kcanTST=empty(len(T))
kMCLRTST=empty(len(T))
kMCLRTST[:] = 0.0    # I could aso use the zeros command, so perhaps this is paranoia.

u = open('MCTSTRates.txt','w+')
u.write('T'+'\t'+'Rate'+'\t'+'MCLRTST'+'\t'+'canonicalTST'+'\t'+'kappa'+'\t'+'QRovib'+'\t'+'Exp'+'\t'+'Conformer'+'\t'+'Constants'+'\n')

for t in range(len(T)):
    # Scale the temperature dependence of the translational partition function
    tf[t] = tf[0]*sqrt(T[0]/T[t])
    # Tunneling factor for temperature t (if tunneling isn't neglected)
    if tunnelswitch=='y':
        kappa[t]=EckartT(Tp,T[t])

    lnq=empty(len(R1c))
    # Correct Gibbs free energy for reactant conformers
    for r in range(len(R1c)):
        # Rotational kT ln Q contribution. Assume sigma = 1
        rotlnQ = lnQr(1,R1c.iloc[r]['A'],R1c.iloc[r]['B'],R1c.iloc[r]['C'],T[t])
        # Vibrational kT ln Q contribution. Loop over vibrational frequencies
        viblnQ=0.0
        for v in range(8,R1c.shape[1]-1):   # -1 because we added the ln Q column earlier
            viblnQ=viblnQ+lnQv(R1c.iloc[r,v],T[t])
        # Replace Gibbs free energy and ln Q
        lnq[r]=rotlnQ+viblnQ      # Save Rovibrational Partition function component
        R1c.iloc[r,3]=R1c.iloc[r]['Energy']+R1c.iloc[r]['ZPE']+kha*T[t]-rotlnQ-viblnQ
    R1c['ln Q'] = lnq

    lnq=empty(len(R2c))
    # Correct Gibbs free energy for reactant conformers
    for r in range(len(R2c)):
        # Rotational kT ln Q contribution. Assume sigma = 1
        rotlnQ = lnQr(1,R2c.iloc[r]['A'],R2c.iloc[r]['B'],R2c.iloc[r]['C'],T[t])
        # Vibrational kT ln Q contribution. Loop over vibrational frequencies
        viblnQ=0.0
        for v in range(8,R2c.shape[1]-1):   # -1 because we added the ln Q column earlier
            viblnQ=viblnQ+lnQv(R2c.iloc[r,v],T[t])
        # Replace Gibbs free energy and ln Q
        lnq[r]=rotlnQ+viblnQ      # Save Rovibrational Partition function component
        R2c.iloc[r,3]=R2c.iloc[r]['Energy']+R2c.iloc[r]['ZPE']+kha*T[t]-rotlnQ-viblnQ
    R2c['ln Q'] = lnq

    lnq=empty(len(TSc))
    # Correct Gibbs free energy for TS conformers. Ignore imaginary frequency.
    for s in range(len(TSc)):
        # Rotational kT ln Q contribution. Assume sigma = 1
        rotlnQ = lnQr(1,TSc.iloc[s]['A'],TSc.iloc[s]['B'],TSc.iloc[s]['C'],T[t])
        # Vibrational kT ln Q contribution. Loop over vibrational frequencies
        viblnQ=0.0
        for v in range(9,TSc.shape[1]-1):   # -1 because we added the ln Q column earlier
            viblnQ=viblnQ+lnQv(TSc.iloc[s,v],T[t])
        # Replace Gibbs free energy and ln Q
        lnq[s]=rotlnQ+viblnQ      # Save Rovibrational Partition function component
        TSc.iloc[s,3]=TSc.iloc[s]['Energy']+TSc.iloc[s]['ZPE']+kha*T[t]-rotlnQ-viblnQ

        # Calculate classical Eyring rate for Conformer s
        rateE[t,s]=exp(-(TSc['Gibbs'].iloc[s]+TSCCC-R1c['Gibbs'].iloc[R1min]-R1CCC-R2c['Gibbs'].iloc[R2min]-R2CCC)/(kha*T[t]))
    TSc['ln Q'] = lnq

    RBfactor=exp(-(R1c['Gibbs']-R1c['Gibbs'].iloc[R1min])/(kha*T[t])).sum()*exp(-(R2c['Gibbs']-R2c['Gibbs'].iloc[R2min])/(kha*T[t])).sum()
    TSBfactor=exp(-(TSc['Gibbs']-TSc['Gibbs'].iloc[TSmin])/(kha*T[t])).sum()
    Qfactor=exp((TSc['ln Q'].iloc[TSmin]-R1c['ln Q'].iloc[R1min]-R2c['ln Q'].iloc[R2min])/(kha*T[t])+1)
    Efactor=exp(-(TSminE+TSCCC-R1minE-R1CCC-R2minE-R2CCC)/(kha*T[t]))

    kcanTST[t]=Deg*tf[t]*sum(rateE[t])*kappa[t]/RBfactor # Canonical TST rate

    # Compute the LRTST rate coefficient for every reactant conformer pair and take the Boltzmann-weighted average.
    for r1 in range(len(R1c)):
        for r2 in range(len(R2c)):
            kMCLRTST[t]=kMCLRTST[t]+LRTST(R1c.iloc[r1]['Dip'],R2c.iloc[r2]['Dip'],T[t])* \
            exp(-(R1c['Gibbs'].iloc[r1]-R1c['Gibbs'].iloc[R1min])/(kha*T[t]))*exp(-(R2c['Gibbs'].iloc[r2]-R2c['Gibbs'].iloc[R2min])/(kha*T[t]))
    kMCLRTST[t]=kMCLRTST[t]/RBfactor # Correct rate coefficient by dividing by the sums of conformer populations.

    # Write data to file.
    u.write(str(T[t])+'\t'+str(kMCLRTST[t]*kcanTST[t]/(kMCLRTST[t]+kcanTST[t]))+'\t'+str(kMCLRTST[t])+'\t'+str(kcanTST[t])+'\t'+str(kappa[t])+'\t'+ \
        str(Qfactor)+'\t'+str(Efactor)+'\t'+str(TSBfactor/RBfactor)+'\t'+str(Deg*tf[t])+'\n')

u.close()

# Print raw data if you want.
print('Rates calculated. Results in MCTSTRates.txt.')

# Calculate entropy contribution to Delta G at 298 K (again) for printing:
# Translational component of -kT lnQ
lnq298=-kha*298*log( 101325 * h**3 * k**-2.5 * amu**-1.5 )
lnq298=lnq298-kha*298*log((2*pi)**-1.5 * 298**-2.5 * (R1mass+R2mass)**1.5 * R1mass**-1.5 * R2mass**-1.5)
# Rotational component of -kT lnQ
lnq298=lnq298-lnQr(1,TSc.iloc[TSmin]['A'],TSc.iloc[TSmin]['B'],TSc.iloc[TSmin]['C'],298.0)+lnQr(1,R1c.iloc[R1min]['A'],R1c.iloc[R1min]['B'],R1c.iloc[R1min]['C'],298.0)+lnQr(1,R2c.iloc[R2min]['A'],R2c.iloc[R2min]['B'],R2c.iloc[R2min]['C'],298.0)
# Vibrational component of -kT lnQ
for v in range(8,R1c.shape[1]-1):
    lnq298=lnq298+lnQv(R1c.iloc[R1min,v],298.0)
for v in range(8,R2c.shape[1]-1):
    lnq298=lnq298+lnQv(R2c.iloc[R2min,v],298.0)
for v in range(9,TSc.shape[1]-1):
    lnq298=lnq298-lnQv(TSc.iloc[TSmin,v],298.0)

print()
print('Some data at 298 K for saving:')
print('Delta E (kJ/mol) '+str((N_A*0.001*eh)*(TSEminE+TSCCC-R1EminE-R1CCC-R2EminE-R2CCC)))
print('Delta G (kJ/mol) '+str((N_A*0.001*eh)*(TSminE+TSCCC-R1minE-R1CCC-R2minE-R2CCC+lnq298-kha*298)))
print('Delta E (DFT) '+str((N_A*0.001*eh)*(TSEminE-R1EminE-R2EminE)))
print('Minimum conformers:')
print('R1,E: '+str(R1Emin)+' R1,G: '+str(R1min)+' R2,E: '+str(R2Emin)+' R2,G: '+str(R2min)+' TS,E: '+str(TSEmin)+' TS,G: '+str(TSmin))
print()
print('Frequency: '+str(imfreq))
print('Thinnest barrier: '+str(TSc['v1'].min()))
if tunnelswitch=='y':
    print('IRC Forward Barrier: '+str((N_A*0.001*eh)*(v1)))
    print('IRC Backward Barrier: '+str((N_A*0.001*eh)*(v2)))
print('Tunneling: '+str(kappa[int(298-Tlow)]))
print()
print('Canonical TST rate: '+str(kcanTST[int(298-Tlow)]))
print('Long-range TST rate: '+str(kMCLRTST[int(298-Tlow)]))
print('Total rate: '+str(kMCLRTST[int(298-Tlow)]*kcanTST[int(298-Tlow)]/(kMCLRTST[int(298-Tlow)]+kcanTST[int(298-Tlow)])))
