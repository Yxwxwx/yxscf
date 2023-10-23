import numpy as np
from pyscf import scf, gto

# Set up molecular geometry and basis
mol = gto.M(
    atom = 'H 0 0 0; F 0 0 1.1',
    basis = 'ccpvqz')

# ==> Set default program options <==
# Maximum SCF iterations
max_iter = 40
# Energy convergence criterion
E_conv = 1.0e-10
D_conv = 1.0e-8
# ==>Get informt <==
# get atom coordinate
A_t = mol.atom_coords()
# get atom charge
Z_A = mol.atom_charges()
# get number of AO
nao = mol.nao_nr()
# get number of electron
ne = sum(mol.nelec)
# get the number of alpha orb
nalpha = mol.nelec[0]
# get the number of beta orb
nbeta = mol.nelec[1]
# get the number of doubly occupied orbitals
ndocc = min(nalpha, nbeta)
# get the number of single occupied orbitals
nsocc = abs(nalpha - nbeta)
# print
print("Number of AO: ", nao)
print("Number of alpha electrons: ", nalpha)
print("Number of beta electrons: ", nbeta)
print("Number of electrons: ", ne)
print("Number of doubly occupied orbitals: ", ndocc)
print("Number of single occupied orbitals: ", nsocc)

# 1e-int
S = mol.intor("int1e_ovlp_sph")  # overlop
T = mol.intor("int1e_kin_sph")  # kintic
V = mol.intor("int1e_nuc_sph")  # nuc_e
H = T + V

# 2e-int
I = mol.intor("int2e_sph", aosym=1)
I = np.reshape(I, [nao, nao, nao, nao])

# start DM as just as pyscf guss
myscf = mol.RHF().run()
myscf.MP2().run()

# SCF & Previous Energy
SCF_E = 0.0
E_old = 0.0

# ==> start MP2 calculation <==
#get the orbital energies
print('Start MP2 calculation')
SCF_E = myscf.e_tot
eigs = myscf.mo_energy
coeffs = myscf.mo_coeff

e_occ = eigs[:ndocc]
e_vir = eigs[ndocc:]
#get the coefficients
c_occ = coeffs[:, :ndocc]
c_vir = coeffs[:, ndocc:]

# Naive Algorithm for ERI Transformation
#O(N^8)
Imo = np.einsum('pi,qa,pqrs,rj,sb->iajb', c_occ, c_vir, I, c_occ, c_vir, optimize=True)

# Compute SS MP2 Correlation
mp2_ss_corr = 0.0
for i in range(ndocc):
    for a in range(nao - ndocc):
        for j in range(ndocc):
            for b in range(nao - ndocc):
                numerator = Imo[i,a,j,b] * (Imo[i, a, j, b] - Imo[i, b, j, a])
                mp2_ss_corr += numerator / (e_occ[i] + e_occ[j] - e_vir[a] - e_vir[b])

# Compute OS MP2 Correlation
mp2_os_corr = 0.0
for i in range(ndocc):
    for a in range(nao - ndocc):
        for j in range(ndocc):
            for b in range(nao - ndocc):
                numerator = Imo[i,a,j,b] * (Imo[i, a, j, b])
                mp2_os_corr += numerator / (e_occ[i] + e_occ[j] - e_vir[a] - e_vir[b])

e_corr = mp2_os_corr + mp2_ss_corr
MP2_E = SCF_E + e_corr
print('MP2 Energy = %4.16f E_correction = %4.16f ' % (MP2_E, e_corr))
