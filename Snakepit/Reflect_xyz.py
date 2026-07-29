import sys
import numpy as np

xyz=str(sys.argv[1]) # xyz filename
i1=int(sys.argv[2])-1 # Index of atom 1 (python indexing)
i2=int(sys.argv[3])-1 # Index of atom 1 (python indexing)
i3=int(sys.argv[4])-1 # Index of atom 1 (python indexing)
i4=int(sys.argv[5])-1 # Index of atom 1 (python indexing)

with open(xyz) as f:
    lines = f.readlines()

# Grab the correct lines and turn them into np arrays.
a1 = np.array(lines[i1+2].split()[1:], dtype=float)
a2 = np.array(lines[i2+2].split()[1:], dtype=float)
a3 = np.array(lines[i3+2].split()[1:], dtype=float)
a4 = np.array(lines[i4+2].split()[1:], dtype=float)

# Step 1: Form the plane from atomic positions a1, a2, and a4
a = a2 - a1
b = a4 - a1

# Find the normal vector and normalize
n = np.cross(a, b)
nn = n / np.linalg.norm(n)

# Step 4: Reflect v3 across the plane
a3refl = a3 - 2 * np.dot(a3 - a1, nn) * nn

print(' ',lines[i3+2].split()[0],'        ', a3refl[0],' ',a3refl[1],'  ',a3refl[2])
