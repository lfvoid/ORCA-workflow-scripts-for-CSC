import sys
import numpy as np
import pandas as pd
from itertools import permutations

noa=int(sys.argv[1])      # Number of atoms in the molecule
numconf=int(sys.argv[2])  # Number of conformers whose geometries to compare
nMet=int(sys.argv[3])     # Number of methyl groups requiring rotation
if nMet > 0:
    Metlist = np.array(sys.argv[4:4+nMet],dtype=int)
    # Array of H atom indices for methyl rotations
    Metlist = Metlist-1          # Now with python indexing
    intlist = sys.argv[4+nMet:]  # List of filenames for the internal geometries
else:
    intlist = sys.argv[4:]       # List of filenames for the internal geometries

# Open the reference geometry.
Refgeom=pd.read_csv('Refgeom.xyz',sep='\s+')

#######################################################
# STANDARDIZE THE INTERNAL GEOMETRIES OF THE CONFORMERS
#######################################################

# Create the Geoms array, to which the standardized internal coordinates will be added.
Geoms=np.empty((numconf,noa,3), dtype=float)
# First element: Conformer
# Second element: Atom
# Third element: R, A, or D

# Array for xyzts for standardizing the internal geometries.
xyz=np.empty((noa,3), dtype=float)

for c in range(numconf):
    # Open the internal geometry of the conformer as a DataFrame.
    Conf=pd.read_csv(intlist[c],sep='\s+')

    # Make a list of the differences between internal coordinate definitions
    rta=list(Conf[Refgeom['c1']!=Conf['c1']].index)
    ata=list(Conf[Refgeom['c2']!=Conf['c2']].index)
    dta=list(Conf[Refgeom['c3']!=Conf['c3']].index)

    if len(rta) > 0 or len(ata) > 0 or len(dta) > 0:
    # Internal geometry is not defined the same way as reference. We need to rebuild the coordinates using the xyz file.

        # Open the correct xyz file and assign the coordinates to the xyz array:
        with open(intlist[c][:-3]+'xyz', "r") as file: 
            # Skip the first two lines
            file.readline()  # First line
            file.readline()  # Second line
    
            # Assign coordinates to xyz:
            for i, line in enumerate(file):
                # Extract the coordinates and assign them to the correct row of the array
                xyz[i] = list(map(float, line.split()[1:4]))  # Parse x, y, z and assign

        
        # In case some rows include changes in more than one coordinate definition, remove indices to prevent double-counting.
        for ind in dta:
            if ind in rta or ind in ata:
                dta.remove(ind)

        for ind in ata:
            if ind in rta:
                ata.remove(ind)

        # Loop over bond length requiring adjustment.
        for a1 in rta:
            # Index of atom 2 is found by checking what's in Refgeom at that location.
            a2=Refgeom['c1'].loc[a1]-1 # Note python indexing.
                
            # Calculate the distance and assign it.
            r12=np.sqrt((xyz[a1,0]-xyz[a2,0])**2+(xyz[a1,1]-xyz[a2,1])**2+(xyz[a1,2]-xyz[a2,2])**2)
            Conf.iloc[a1,4]=np.around(r12,12)

            # Changes in bond definitions require adjustment of the associated angles and dihedrals:
            a3=Refgeom['c2'].loc[a1]-1 # Index of atom 3.
            
            # Distance between atoms 2 and 3
            r23 = np.sqrt((xyz[a2,0]-xyz[a3,0])**2+(xyz[a2,1]-xyz[a3,1])**2+(xyz[a2,2]-xyz[a3,2])**2)
            # Calculate the angle and assign it.
            theta=np.dot(xyz[a1]-xyz[a2], xyz[a3]-xyz[a2])/(r12*r23)
            Conf.iloc[a1,5] = np.around(np.arccos(np.clip(theta, -1, 1))*180/np.pi,8)

            a4=Refgeom['c3'].loc[a1]-1 # Index of atom 4

            # Define bond vectors
            b1 = xyz[a2] - xyz[a1]
            b2 = xyz[a3] - xyz[a2]
            b3 = xyz[a4] - xyz[a3]
            # Compute normal vectors
            cp12 = np.cross(b1, b2)
            cp23 = np.cross(b2, b3)
            # Sine and cosine for the dihedral angle calculation
            cos_phi = np.dot(cp12, cp23)
            sin_phi = np.dot(b2/np.linalg.norm(b2), np.cross(cp12, cp23))
            #Dihedral angle
            radih = np.arctan2(sin_phi, cos_phi)
            # Print dihedral in (0,360) format instead of (-180,180) format.
            if radih > 0:
                Conf.iloc[a1,6] = np.around(np.arctan2(sin_phi, cos_phi)*180/np.pi,8)
            else:
                Conf.iloc[a1,6] = np.around(360+np.arctan2(sin_phi, cos_phi)*180/np.pi,8)
                
        for a1 in ata:
            # Indices of atoms 2 & 3 are found by checking what's in Refgeom at that location.
            a2=Refgeom['c1'].loc[a1]-1 # Note python indexing.
            a3=Refgeom['c2'].loc[a1]-1 # Note python indexing.
            r12=Conf.iloc[a1,4]

            # Distance between atoms 2 and 3
            r23 = np.sqrt((xyz[a2,0]-xyz[a3,0])**2+(xyz[a2,1]-xyz[a3,1])**2+(xyz[a2,2]-xyz[a3,2])**2)
            # Calculate the angle and assign it.
            theta=np.dot(xyz[a1]-xyz[a2], xyz[a3]-xyz[a2])/(r12*r23)
            Conf.iloc[a1,5] = np.around(np.arccos(np.clip(theta, -1, 1))*180/np.pi,8)
        
            # Changes in angle definitions require adjustemnt in the corresponding dihedral.
            a4=Refgeom['c3'].loc[a1]-1 # Index of atom 4
            # Define bond vectors
            b1 = xyz[a2] - xyz[a1]
            b2 = xyz[a3] - xyz[a2]
            b3 = xyz[a4] - xyz[a3]
            # Compute normal vectors
            cp12 = np.cross(b1, b2)
            cp23 = np.cross(b2, b3)
            # Sine and cosine for the dihedral angle calculation
            cos_phi = np.dot(cp12, cp23)
            sin_phi = np.dot(b2/np.linalg.norm(b2), np.cross(cp12, cp23))
            #Dihedral angle
            radih = np.arctan2(sin_phi, cos_phi)
            # Print dihedral in (0,360) format instead of (-180,180) format.
            if radih > 0:
                Conf.iloc[a1,6] = np.around(np.arctan2(sin_phi, cos_phi)*180/np.pi,8)
            else:
                Conf.iloc[a1,6] = np.around(360+np.arctan2(sin_phi, cos_phi)*180/np.pi,8)

        for a1 in dta:
            # Indices of atoms 2-4 are found by checking what's in Refgeom at that location.
            a2=Refgeom['c1'].loc[a1]-1 # Note python indexing.
            a3=Refgeom['c2'].loc[a1]-1 # Note python indexing.
            a4=Refgeom['c3'].loc[a1]-1 # Note python indexing.

            # Define bond vectors
            b1 = xyz[a2] - xyz[a1]
            b2 = xyz[a3] - xyz[a2]
            b3 = xyz[a4] - xyz[a3]
            # Compute normal vectors
            cp12 = np.cross(b1, b2)
            cp23 = np.cross(b2, b3)
            # Sine and cosine for the dihedral angle calculation
            cos_phi = np.dot(cp12, cp23)
            sin_phi = np.dot(b2/np.linalg.norm(b2), np.cross(cp12, cp23))
            #Dihedral angle
            radih = np.arctan2(sin_phi, cos_phi)
            # Print dihedral in (0,360) format instead of (-180,180) format.
            if radih > 0:
                Conf.iloc[a1,6] = np.around(np.arctan2(sin_phi, cos_phi)*180/np.pi,8)
            else:
                Conf.iloc[a1,6] = np.around(360+np.arctan2(sin_phi, cos_phi)*180/np.pi,8)

        # Finally, add the newly redefined internal coordinates to Geoms array.
        Geoms[c]=Conf.iloc[:,4:]
        
    else:
        # Internal coordinates defined the same way. Excellent! Add coordinates to Geoms array:
        Geoms[c]=Conf.iloc[:,4:]

print(numconf,'internal geometries standardized!')

# A function for performing methyl rotations, since they screw up the RMSD calculation otherwise.
def CH3_rot(set1, set2):
    ref_diff = -3.0
    for p in permutations(range(3)):  # All cyclic permutations of [0,1,2]
        permset = np.array([set2[i] for i in p])  # Reorder test_set
        # Calculate the difference using only the dihedral angle.

        diff = np.sum(np.cos(np.deg2rad(set1[:,2] - permset[:,2]))) 
        # cos(0) = 1 -> diff = 3 means maximal agreement and diff = -3 means maximal disagreement. 

        if diff > ref_diff:
            ref_diff= diff
            best_order = p
            best_perm = permset

    return best_perm


############################################################################
# CALCUALTE ROOT MEAN SQUARE DISTANCE FROM STANDARDIZED INTERNAL COORDINATES
############################################################################

u = open('RMSD.txt','w+')
g = open('Geoms.int','w+')
u.write('c1'+'\t'+'c2'+'\t'+'Name1'+'\t'+'Name2'+'\t'+'RMSD_(Å)'+'\n')
for ci in range(numconf):
    g.write(str(ci+1)+'\t'+'c1'+'\t'+'c2'+'\t'+'c3'+'\t'+'R'+'\t'+'A'+'\t'+'D'+'\n')
    for cj in range(ci+1,numconf):

        # Rotate over all three H atom permutations for each CH3 group.
        if nMet > 0:
            for m in Metlist:   # For each methyl group...
                Geoms[cj,m:m+3]=CH3_rot(Geoms[ci,m:m+3],Geoms[cj,m:m+3]) # ...find the atom permutation that minimizes the difference between the conformer pair.

	# Formula for distance (squared) in spherical coordinates: 
	#|r1-r2|**2 = r1**2 + r2**2 -2*r1*r2*cos(theta1-theta2) - 2*r1*r2*sin(theta1)*sin(theta2)*(cos(phi1-phi2)-1)
        sd=Geoms[ci,:,0]**2+Geoms[cj,:,0]**2-2*Geoms[ci,:,0]*Geoms[cj,:,0]*np.cos(np.deg2rad(Geoms[ci,:,1]-Geoms[cj,:,1]))- \
        2*Geoms[ci,:,0]*Geoms[cj,:,0]*np.sin(np.deg2rad(Geoms[ci,:,1]))*np.sin(np.deg2rad(Geoms[cj,:,1]))*(np.cos(np.deg2rad(Geoms[ci,:,2]-Geoms[cj,:,2]))-1)
        rmsd=np.sqrt(np.average(sd))
        u.write(str(ci+1)+'\t'+str(cj+1)+'\t'+intlist[ci][:-4]+'\t'+intlist[cj][:-4]+'\t'+f"{rmsd:.12f}\n")
    for a in range(noa):
        g.write(str(Refgeom['E'].iloc[a])+'\t'+str(Refgeom['c1'].iloc[a])+'\t'+str(Refgeom['c2'].iloc[a])+'\t'+str(Refgeom['c3'].iloc[a])+'\t'+ \
        str(Geoms[ci,a,0])+'\t'+str(Geoms[ci,a,1])+'\t'+str(Geoms[ci,a,2])+'\n')
u.close()
g.close()

print('RMSD values found in file RMSD.txt')
print('Internal geometries are found in the file Geoms.int')
