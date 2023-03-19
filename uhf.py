import numpy as np
from scipy.linalg import sqrtm, eig
from pyscf import scf, gto

import os, sys
from pyscf.tools import dump_mat
print_matrix = lambda x, title="", stdout=sys.stdout: (print(title, file=stdout), dump_mat.dump_rec(stdout, x))

# Set up molecular geometry and basis
mol = gto.M(atom='O 0 0 0.5; O 0 0 -0.5', basis='ccpvdz', spin=2, charge=0)

# ==> Set default program options <==
# Maximum SCF iterations
max_iter = 200
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

#calculate s=S^{-0.5}
#A = sqrtm(eigh(S, eigvals_only=True)[::-1])
'''
w, v = eig(S)
L = v
D = np.diag(w)
D_neg_sqrt = np.diag(1 / np.sqrt(-w))
A = L.dot(D_neg_sqrt).dot(L.T)
'''

# 2e-int
I = mol.intor("int2e_sph", aosym=1)
I = np.reshape(I, [nao, nao, nao, nao])

# Trial & Residual Vector Lists -- one each for alpha & beta
F_list_a = []
F_list_b = []
R_list_a = []
R_list_b = []


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
    B_dim = len(F_list) + 1
    B = np.empty((B_dim, B_dim))
    B[-1, :] = -1
    B[:, -1] = -1
    B[-1, -1] = 0
    for i in range(len(F_list)):
        for j in range(len(F_list)):
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


# ==> Build alpha & beta CORE guess <==
# Perform a UHF calculation to obtain the alpha and beta density matrices
scf_eng = scf.UHF(mol)
e_uhf_ref = scf_eng.scf()
# Extract the alpha and beta density matrices
Da = scf_eng.get_init_guess()[0]
Db = scf_eng.get_init_guess()[1]
Da_ref, Db_ref = scf_eng.make_rdm1()

# SCF & Previous Energy
SCF_E = 0.0
E_old = 0.0

# ==> UHF-SCF Iterations <==
for scf_iter in range(1, max_iter + 1):

    # GET Fock martix
    Fa = T + V + get_j(Da) + get_j(Db) - get_k(Da)
    # TODO: check if this is correct
    Fb = T + V + get_j(Da) + get_j(Db) - get_k(Da)

    # Check if the Fock matrix is identical to the reference
    Fa_ref, Fb_ref = scf_eng.get_fock(dm=(Da, Db))
    print("Fa difference = %6.4e" % np.linalg.norm(Fa - Fa_ref))
    print("Fb difference = %6.4e" % np.linalg.norm(Fb - Fb_ref))

    # Compute DIIS residual for Fa & Fb
    '''error vector = FDS - SDF '''
    diis_r_a = Fa.dot(Da).dot(S) - S.dot(Da).dot(Fa)
    diis_r_b = Fb.dot(Db).dot(S) - S.dot(Db).dot(Fb)
    
    # Append trial & residual vectors to lists
    F_list_a.append(Fa)
    F_list_b.append(Fb)
    R_list_a.append(diis_r_a)
    R_list_b.append(diis_r_b)

    # Compute UHF Energy
    SCF_E = np.einsum('pq,pq->', (Da + Db), H, optimize=True)
    SCF_E += np.einsum('pq,pq->', Da, Fa, optimize=True)
    SCF_E += np.einsum('pq,pq->', Db, Fb, optimize=True)
    SCF_E *= 0.5
    SCF_E += E_nuc
    dE = SCF_E - E_old
    dRMS = 0.5 * (np.mean(diis_r_a ** 2) ** 0.5 + np.mean(diis_r_b ** 2) ** 0.5)
    print('SCF Iteration %3d: Energy = %4.16f dE = % 1.5E dRMS = %1.5E' % (scf_iter, SCF_E, dE, dRMS))

    if (abs(dE) < E_conv) and (dRMS < D_conv):
        print("SCF convergence! Congrats")
        break

    E_old = SCF_E
    # DIIS Extrapolation
    if scf_iter >= 20:
        if scf_iter == 20:
            print("DIIS start!")
        Fa = diis_xtrap(F_list_a, R_list_a)
        Fb = diis_xtrap(F_list_b, R_list_b)

    Da = make_D(Fa, nalpha)
    Db = make_D(Fb, nbeta)

print("\n")
print("UHF Energy = %12.8f" % SCF_E)
print("Ref Energy = %12.8f" % e_uhf_ref)
print("Difference in total energy         = %6.4e" % (SCF_E - e_uhf_ref))
print("Difference in alpha density matrix = %6.4e" % np.mean(np.abs(Da - Da_ref)))
print("Difference in beta  density matrix = %6.4e" % np.mean(np.abs(Db - Db_ref)))