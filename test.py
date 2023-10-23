import fci
import numpy as np
import time

assert(fci.compare_excitation({0, 1, 2}, {0, 1, 2}) == (set(), set()))
assert(fci.compare_excitation({0, 1, 2}, {0, 1, 3}) == ({2}, {3}))
assert(fci.compare_excitation({0, 1, 2}, {0, 2, 3}) == ({1}, {3}))

assert (fci.phase_factor(({2}, {3}), [0, 1, 2], [0, 1, 3])) == 1.0
assert (fci.phase_factor(({1}, {3}), [0, 1, 2], [0, 2, 3])) == -1.0

h1e = np.load("doc/h1e.npy")
h2e = np.load("doc/h2e.npy")

np.set_printoptions(threshold=np.inf)
hamiltionian = fci.ci_hamiltonian(h1e, h2e, 6, 0)
start = time.time()
eigvals = np.linalg.eigvals(hamiltionian)
assert (np.abs(np.sort(eigvals)[0] + 7.8399080148963369) < 1e-13)

end = time.time()
print("Direct hamltonian matrix element access algorithm", end - start)

print("===============================================")
#
#ci_diagonal, ci_sparse_matrix = fci.ci_hamiltonian_in_sparse_matrix(h1e, h2e, 6)
# start = time.time()
#
# assert(
#    np.abs(fci.davidson_diagonalization(
#        lambda vec: fci.sparse_matrix_transform(ci_sparse_matrix, vec),
#        ci_diagonal,
#        0,
#        2,
#        400,
#        residue_tol=1e-13
# )[0] + 7.8399080148963369) < 1e-13)
#
# end = time.time()
#
# print("Sparse matrix diagonalization algorithm", end - start)

start = time.time()

assert (
        np.abs(fci.davidson_diagonalization_direct(
            hamiltionian,
            4,
            400,
            residue_tol=1e-13
        ) + 7.8399080148963369) < 1e-13)

end = time.time()
print("Daivison diagonalization algorithm", end - start)
#assert (
#        np.abs(fci.davidson_diagonalization(
#            fci.knowles_handy_full_ci_transformer(h1e, h2e, 6),
#            ci_diagonal,
#            0,
#            2,
#            400,
#            residue_tol=1e-5
#        ) + 7.8399080148963369) < 1e-10)
