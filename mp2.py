import numpy as np
from pyscf import scf, gto

# Set up molecular geometry and basis
mol = gto.M(
    atom = 'H 0 0 0; F 0 0 1.1',
    basis = 'ccpvdz')

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
# ==>Calculate nuc_e with pyscf <==
# get r_AB martix
r_AB = np.empty((mol.natm, mol.natm))
for A in range(mol.natm):
    for B in range(mol.natm):
        if A != B:
            r_AB[A, B] = np.linalg.norm(A_t[A] - A_t[B])
        else:
            r_AB[A, B] = np.infty

E_nuc = 0.5 * np.einsum("A, B, AB ->", Z_A, Z_A, 1 / r_AB)

# ==>Calculate 1e and 2e int with pyscf <==
# 1e-int
S = mol.intor("int1e_ovlp_sph")  # overlop
T = mol.intor("int1e_kin_sph")  # kintic
V = mol.intor("int1e_nuc_sph")  # nuc_e
H = T + V

# 2e-int
I = mol.intor("int2e_sph", aosym=1)
I = np.reshape(I, [nao, nao, nao, nao])

# Trial & Residual Vector Lists -- one each for alpha & beta
F_list = []
R_list = []

# Calculate the coulomb matrices from density matrix
def get_j(D):
    J = np.zeros((nao, nao))  # Initialize the Coulomb matrix
    J = np.einsum('pqrs,rs->pq', I, D, optimize=True)
    return J


# Calculate the exchange  matrices from density matrix
def get_k(D):
    # K = np.zeros((nao, nao))  # Initialize the K matrix
    K = np.einsum('prqs,rs->pq', I, D, optimize=True)
    return K

# Calculate the density matrix
def make_D(fock, norb):
    eigs, coeffs = scf.hf.eig(fock, S)  # this is a PySCF function to carry out the diagonalization
    c_occ = coeffs[:, :norb]
    D =  np.einsum('pi,qi->pq', c_occ, c_occ, optimize=True)
    return D

# ==> Build DIIS Extrapolation Function <==
def diis_xtrap(F_list, DIIS_RESID):
    # Build B matrix
    n = len(F_list)
    B_dim = n + 1
    B = np.empty((B_dim, B_dim))
    B[-1, :] = -1
    B[:, -1] = -1
    B[-1, -1] = 0
    for i in range(n):
        for j in range(n):
            B[i, j] = np.einsum('ij,ij->', DIIS_RESID[i], DIIS_RESID[j], optimize=True)
    # Build RHS of Pulay equation
    rhs = np.zeros((B_dim))
    rhs[-1] = -1

    # Solve Pulay equation for c_i's with NumPy
    coeff = np.linalg.solve(B, rhs)

    # Build DIIS Fock matrix
    F_DIIS = np.zeros_like(F_list[0])
    for x in range(coeff.shape[0] - 1):
        F_DIIS += coeff[x] * F_list[x]

    return F_DIIS

# start DM as just as pyscf guss
scf_eng = mol.RHF().run()
scf_eng.MP2().run()
D = scf_eng.get_init_guess()

# SCF & Previous Energy
SCF_E = 0.0
E_old = 0.0

# ==> RHF-SCF Iterations <==
for scf_iter in range(1, max_iter + 1):

    # GET Fock martix
    F = H + 2 * get_j(D) - get_k(D)
    #build error vector
    '''error vector = FDS - SDF '''
    diis_r = F.dot(D).dot(S) - S.dot(D).dot(F)
    F_list.append(F)
    R_list.append(diis_r)
    SCF_E = np.einsum('pq,pq->', (H + F), D, optimize=True) + E_nuc
    dE = SCF_E - E_old
    dRMS = 0.5 * np.mean(diis_r ** 2) ** 0.5
    print('SCF Iteration %3d: Energy = %4.16f dE = % 1.5E dRMS = %1.5E' % (scf_iter, SCF_E, dE, dRMS))

    if (abs(dE) < E_conv) and (dRMS < D_conv):
        print("SCF convergence! Congrats")
        break
    E_old = SCF_E
    # DIIS Extrapolation
    if scf_iter >= 2:
        if scf_iter == 2:
            print("DIIS start!")
        F = diis_xtrap(F_list, R_list)
    D = make_D(F, ndocc)

# ==> start MP2 calculation <==
#get the orbital energies
print('Start MP2 calculation')
eigs, coeffs = scf.hf.eig(F, S)  # this is a PySCF function to carry out the diagonalization
e_occ = eigs[:ndocc]
e_vir = eigs[ndocc:]
#get the coefficients
c_occ = coeffs[:, :ndocc]
c_vir = coeffs[:, ndocc:]

# Naive Algorithm for ERI Transformation
#O(N^8)
Imo = np.einsum('pi,qa,pqrs,rj,sb->iajb', c_occ, c_vir, I, c_occ, c_vir, optimize=True)

'''
# ==> Transform I -> Imo @ O(N^5) <==
tmp = np.einsum('pi,pqrs->iqrs', c_occ, I, optimize=True)
tmp = np.einsum('qa,iqrs->iars', c_vir, tmp, optimize=True)
tmp = np.einsum('iars,rj->iajs', tmp, c_occ, optimize=True)
Imo = np.einsum('iajs,sb->iajb', tmp, c_vir, optimize=True)
'''

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
print('MP2 Energy = %4.16f E_correctionw = %4.16f ' % (MP2_E, e_corr))
