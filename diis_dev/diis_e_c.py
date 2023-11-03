import time
import numpy as np
import scipy
from pyscf import gto

# Set up molecular geometry and basis
mol = gto.M(atom=
"""
O  0.0  0.0  0.0
O  0.0  0.0  1.5
H  1.0  0.0  0.0
H  0.0  0.7  1.0
"""
, basis='ccpvdz')
# ==> Set default program options <==
# Maximum SCF iterations
max_iter = 100
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
t = time.time()

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

print('\nTotal time taken for integrals: %.3f seconds.' % (time.time() - t))

t = time.time()

# Calculate the density matrix
def make_D(fock, norb):
    eigs, coeffs = scipy.linalg.eigh(fock, S)  # this is a PySCF function to carry out the diagonalization
    c_occ = coeffs[:, :norb]
    D =  np.einsum('pi,qi->pq', c_occ, c_occ, optimize=True)
    return D
# Calculate the coulomb matrices from density matrix
def get_j(D):   
    return np.einsum('pqrs,rs->pq', I, D, optimize=True)
# Calculate the exchange  matrices from density matrix
def get_k(D):
    return np.einsum('prqs,rs->pq', I, D, optimize=True)
# Calculate the fock matrix
def make_F(dm):
    return H + 2 * get_j(dm) - get_k(dm)
# Calculate the SCF energy
def get_scf_energy(dm):
    return np.einsum('pq, pq->', (H + make_F(dm)), dm, optimize=True)

# Define the EDIIS minimization function
def ediis_minimize(energies, fock_matrices, density_matrices):
    # Get the number of elements (nx)
    nx = len(energies)

    # Check that the input arrays have the same length (nx)
    assert nx == len(fock_matrices)
    assert nx == len(density_matrices)

    # Calculate the difference matrix df
    df = np.einsum('inpq, jnqp -> ij', density_matrices, fock_matrices).real
    diag = df.diagonal()
    df = diag[:, None] + diag - df - df.T

    # Define the cost function
    def cost_function(x):
        # Normalize coefficients
        c = x ** 2 / (x ** 2).sum()
        return np.einsum('i,i', c, energies) - np.einsum('i,ij,j', c, df, c)

    # Define the gradient of the cost function
    def gradient(x):
        x2sum = (x ** 2).sum()
        c = x ** 2 / x2sum
        fc = energies - 2 * np.einsum('i,ik -> k', c, df)
        cx = np.diag(x * x2sum) - np.einsum('k, n -> kn', x ** 2, x)
        cx *= 2 / x2sum ** 2
        return np.einsum('k, kn -> n', fc, cx)

    # Minimize the cost function using BFGS optimization
    result = scipy.optimize.minimize(cost_function, np.ones(nx), method='BFGS', jac=gradient, tol=1e-9)

    # Return the minimized cost and the optimized coefficients
    return result.fun, (result.x ** 2) / (result.x ** 2).sum()

def adiis_minimize(fock_matrices, density_matrices):
    # Get the number of elements (nx)
    nx = len(fock_matrices)

    # Check that the input arrays have the same length (nx)
    assert nx == len(density_matrices)

    # Calculate the difference matrix df
    df = np.einsum('inpq, jnqp -> ij', density_matrices, fock_matrices).real
    d_fn = df[:, nx - 1]
    dn_f = df[nx - 1]
    dn_fn = df[nx - 1, nx - 1]
    dd_fn = d_fn - dn_fn
    df = df - d_fn[:,None] - dn_f + dn_fn

    # Define the cost function
    def cost_function(x):
        # Normalize coefficients
        c = x ** 2 / (x ** 2).sum()
        return np.einsum('i,i', c, dd_fn) * 2 + np.einsum('i,ij,j', c, df, c)

    # Define the gradient of the cost function
    def gradient(x):
        x2sum = (x**2).sum()
        c = x**2 / x2sum
        fc = 2*dd_fn
        fc+= np.einsum('j,kj->k', c, df)
        fc+= np.einsum('i,ik->k', c, df)
        cx = np.diag(x*x2sum) - np.einsum('k,n->kn', x**2, x)
        cx *= 2/x2sum**2
        return np.einsum('k,kn->n', fc, cx)

    # Minimize the cost function using BFGS optimization
    result = scipy.optimize.minimize(cost_function, np.ones(nx), method='BFGS', jac=gradient, tol=1e-9)

    # Return the minimized cost and the optimized coefficients
    return result.fun, (result.x ** 2) / (result.x ** 2).sum()

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
# Calculate initial core guess: [Szabo:1996] pp. 145
D = make_D(H, ndocc)
print('\nTotal time taken for setup: %.3f seconds' % (time.time() - t))

print('\nStart SCF iterations:\n')
t = time.time()

SCF_E = 0.0
E_old = 0.0

n_diis = 5
diis_info = []

ediis = False
nediis = 0.1

adiis = True
nadiis = 0.1

cdiis = True
ncdiis = 0.0001

# ==> RHF-SCF Iterations <==
for scf_iter in range(1, max_iter + 1):

    F = make_F(D)
     #build error vector
    '''error vector = FDS - SDF '''
    diis_r = F.dot(D).dot(S) - S.dot(D).dot(F)
    dRMS = np.mean(diis_r**2)**0.5
    SCF_E = get_scf_energy(D)
    dE = SCF_E - E_old
    E_old = SCF_E

    if (abs(dE) < E_conv) and (dRMS < D_conv):
        print('SCF Iteration %3d: Energy = %4.16f dE = % 1.5E dRMS = %1.5E FINAL' % (scf_iter, SCF_E, dE, dRMS))
        print('Final SCF energy: %4.16f hartree' % SCF_E)
        print("SCF convergence in %3d cycles" % scf_iter)
        print('Total time for SCF iterations: %.3f seconds \n' % (time.time() - t))
        break
    if (scf_iter == max_iter):
        raise Exception("Maximum number of SCF cycles exceeded.")
    
    diis_info.append({
        "scf_iter": scf_iter,
        "fock_matrix": F,
        "density_matrix": D,
        "energy": SCF_E,
        "diis_error": diis_r
        })
    if (len(diis_info) > n_diis):
            del(diis_info[0])

    if (dRMS < nediis) and (dRMS > ncdiis) and (ediis == True):

        print('SCF Iteration %3d: Energy = %4.16f dE = % 1.5E dRMS = %1.5E EDIIS' % (scf_iter, SCF_E, dE, dRMS))

        nx = len(diis_info)
        fock_matrices = np.array([item["fock_matrix"] for item in diis_info]).reshape((nx, -1, nao, nao))
        density_matrices = np.array([item["density_matrix"] for item in diis_info]).reshape((nx, -1, nao, nao))
        energies = np.array([item["energy"] for item in diis_info])
        etot, c = ediis_minimize(energies, fock_matrices, density_matrices)
        F = np.einsum('i,i...pq->...pq', c, fock_matrices).reshape((nao,nao))
        D = make_D(F, ndocc)

    elif (dRMS < nadiis) and (dRMS > ncdiis) and (adiis == True):
        
        print('SCF Iteration %3d: Energy = %4.16f dE = % 1.5E dRMS = %1.5E ADIIS' % (scf_iter, SCF_E, dE, dRMS))

        nx = len(diis_info)
        fock_matrices = np.array([item["fock_matrix"] for item in diis_info]).reshape((nx, -1, nao, nao))
        density_matrices = np.array([item["density_matrix"] for item in diis_info]).reshape((nx, -1, nao, nao))
        etot, c = adiis_minimize(fock_matrices, density_matrices, )
        F = np.einsum('i,i...pq->...pq', c, fock_matrices).reshape((nao,nao))
        D = make_D(F, ndocc)

    
    elif ((cdiis == True) and (dRMS <= ncdiis)) or ((ediis == False) and (adiis == False)):

        print('SCF Iteration %3d: Energy = %4.16f dE = % 1.5E dRMS = %1.5E CDIIS' % (scf_iter, SCF_E, dE, dRMS))
        
        nx = len(diis_info)
        fock_matrices = np.array([item["fock_matrix"] for item in diis_info]).reshape((nx, nao, nao))
        diis_error = np.array([item["diis_error"] for item in diis_info]).reshape((nx, nao, nao))
        F = diis_xtrap(fock_matrices, diis_error)
        D = make_D(F, ndocc)
    else:
        print('SCF Iteration %3d: Energy = %4.16f dE = % 1.5E dRMS = %1.5E' % (scf_iter, SCF_E, dE, dRMS))
        D = make_D(F, ndocc)

