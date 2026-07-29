import sys
import numpy as np
import pandas as pd

Ecut=float(sys.argv[1])  # Electronic energy cutoff in kJ/mol for filtering. Has value "0.0" if no cutoff is chosen.
RMSDT=float(sys.argv[2]) # RMSD threshold in Å for duplicate filtering
ET=float(sys.argv[3])     # Energy threshold in Hartree for duplicate filtering
RMSDfile=sys.argv[4]     # File with RMSD values for each conformer pair.
enerfile=sys.argv[5]     # File with energy, dipole, etc. values for each conformer.
T=float(sys.argv[6])     # Temperature at which Boltzmann population is calculated

# Open RMSD and energy files as DataFrames:
ConfRMSD=pd.read_csv(RMSDfile,sep='\t')
ConfE=pd.read_csv(enerfile,sep='\s+')

# Add two columns to ConfRMSD with the energy and dipole moment differences for each conformer pair. 
ediff=np.empty(len(ConfRMSD))
dipdiff=np.empty(len(ConfRMSD))
for p in range(len(ConfRMSD)):
    ediff[p]=np.abs(ConfE[ConfE.c==ConfRMSD.iloc[p].c1].E.iloc[0] - ConfE[ConfE.c==ConfRMSD.iloc[p].c2].E.iloc[0])
ConfRMSD['Energy_Ha']=ediff

# For conversion between kJ/mol and Hartree.
import scipy.constants as sc
kj=sc.physical_constants['Hartree energy'][0]*sc.N_A*0.001

# Apply the energy cutoff:
if Ecut > 0:
    LowE=ConfE[(ConfE.E-ConfE.E.min())<Ecut/kj].index+1
    HighE=ConfE[(ConfE.E-ConfE.E.min())>Ecut/kj].index+1
else:
    LowE=ConfE.index+1

# Make columns with the relative conformer electronic & Gibbs free energies.
ConfE['Erel']=np.around(kj*(ConfE.E-ConfE.E.min()),3)
ConfE['Grel']=np.around(kj*(ConfE.Gibbs-ConfE.Gibbs.min()),3)

# Print out which files to keep and which to delete:
Dupscumul=[]
Keepcumul=[]
for c in LowE:

    # Skip conformers that have already been recognized as duplicates.
    if c in Dupscumul:
        continue
    
    # Make a printout of the close cases (withing +/- 20%) for manual observation.
    Close=ConfRMSD[(ConfRMSD['c1']==c) & (ConfRMSD['RMSD_(Å)']<1.2*RMSDT) & (ConfRMSD['Energy_Ha']<1.2*ET) & ((ConfRMSD['RMSD_(Å)']>0.8*RMSDT) | (ConfRMSD['Energy_Ha']>0.8*ET))]
    if len(Close)>0:
        for c2 in range(len(Close)):
            print("Close call:",Close.Name1.iloc[c2],Close.Name2.iloc[c2],Close['RMSD_(Å)'].iloc[c2],Close.Energy_Ha.iloc[c2])

    # Take out the identical conformers from dataframe.
    Dups=ConfRMSD[(ConfRMSD['c1']==c) & (ConfRMSD['RMSD_(Å)']<RMSDT) & (ConfRMSD['Energy_Ha']<ET)]
    if len(Dups)>0:
        # Make an array of conformer indices, starting from c
        confs=np.insert(Dups.c2.values,0,c)
        # Add duplicates to cumulative duplicate list
        Dupscumul.extend(Dups['c2'].values)
        # Find the minimum energy conformer from the set of identical ones and keep it.
        keep=ConfE.iloc[confs-1].E.idxmin()+1
        Keepcumul.append(keep-1)
    else:
        Keepcumul.append(c-1)

Keepcumul = list(dict.fromkeys(Keepcumul)) 

# Make a new dataset with only the unique conformers.
Unique=ConfE.loc[Keepcumul]

# Calculate a 'global' partition function and a cumulative population sum. Use G values if available.
if Unique.Grel.sum()==0:
    Unique['Boltz']=np.exp(-Unique.Erel*1000/(sc.R*T))
    Q=Unique.Boltz.sum()
    Unique['Population']=Unique.sort_values('Erel').Boltz.cumsum()/Q
else:
    Unique['Boltz']=np.exp(-Unique.Grel*1000/(sc.R*T))
    Q=Unique.Boltz.sum()
    Unique['Population']=Unique.sort_values('Grel').Boltz.cumsum()/Q

print("Partition function {:.4f}".format(Q))

# Format thermal population using three decimals.
Unique['Population']=Unique.Population.map(lambda x: f"{x:.4F}")

# Use the list of indices fromt the previous loop to print out all the unique conformers into a file. 
Unique[['Name','Erel','Grel','Population','Dip','conv','cycles','time']].sort_values('Erel').to_csv('Unique.txt',index=False,sep='\t')
