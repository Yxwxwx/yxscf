import numpy as np
import itertools
import math
import copy

# Compare the excitation, i.e. the mismatching orbital indices
def compare_excitation(left_indices, right_indices):
    return set(left_indices) - set(right_indices), set(right_indices) - set(left_indices)

# Calculate the phase factor, i.e. the sign of the determinant when the creation/annihilation operators are swapped to
# the tail of the list of operators
def phase_factor(excitation, left_indices, right_indices):
    indices_swap = 0

    # Check that indices are in ascending order
    assert sorted(left_indices) == left_indices
    assert sorted(right_indices) == right_indices

    for indices, orbitals in zip([left_indices, right_indices], excitation):
        for index, orbital_index in enumerate(orbitals):
            indices_swap += indices.index(orbital_index) - index
    return math.pow(-1, indices_swap)

def ci_hamiltonian(one_electron_integrals, two_electron_integrals, n_elecs, n_spin=0):
    n_rows, n_cols = one_electron_integrals.shape
    n_orbs = n_rows

    assert n_cols == n_rows
    assert np.all(np.array(two_electron_integrals.shape) == n_orbs)

    n_alpha = (n_elecs + n_spin) // 2
    n_beta = n_elecs - n_alpha

    # generates all possible configurations for the occupied orbitals, with indices of the orbitals in ascending order
    alpha_combinations = [list(x) for x in itertools.combinations(range(n_orbs), n_alpha)]
    beta_combinations = [list(x) for x in itertools.combinations(range(n_orbs), n_beta)]

    # the dimension of the hamiltonian matrix (dimension of the determinant basis)
    n_dim = len(alpha_combinations) * len(beta_combinations)

    hamiltonian_matrix = np.zeros((n_dim, n_dim))

    for i in range(n_dim):
        i_alpha_combination = alpha_combinations[i % len(beta_combinations)]
        i_beta_combination = beta_combinations[i // len(beta_combinations)]

        for j in range(i, n_dim):
            j_alpha_combination = alpha_combinations[j % len(beta_combinations)]
            j_beta_combination = beta_combinations[j // len(beta_combinations)]

            # compare the excitation

            alpha_excitation = compare_excitation(i_alpha_combination, j_alpha_combination)
            beta_excitation = compare_excitation(i_beta_combination, j_beta_combination)

            n_alpha_excitation = len(alpha_excitation[0])
            n_beta_excitation = len(beta_excitation[0])
            # more than two electrons are excited - the matrix element is zero
            if n_alpha_excitation + n_beta_excitation > 2:
                continue

            # the phase factor\
            alpha_phase = phase_factor(alpha_excitation, i_alpha_combination, j_alpha_combination)
            beta_phase = phase_factor(beta_excitation, i_beta_combination, j_beta_combination)

            total_phase = alpha_phase * beta_phase

            # No excitation - the matrix element is the same on both sides
            if n_alpha_excitation == 0 and n_beta_excitation == 0:
                one_electron_part = \
                    np.einsum("ii->", one_electron_integrals[np.ix_(i_alpha_combination, i_alpha_combination)]) \
                    + np.einsum("ii->", one_electron_integrals[np.ix_(i_beta_combination, i_beta_combination)])
                # <ij | v | ij>, or (ii | jj). Non trivial contribution from configurations having
                # the same spin for i and the same spin for j
                coulomb_part = \
                    np.einsum("iijj->", two_electron_integrals[np.ix_(i_alpha_combination, i_alpha_combination,
                                                                      j_beta_combination, j_beta_combination)]) \
                    + 0.5 * np.einsum("iijj->", two_electron_integrals[np.ix_(i_alpha_combination, i_alpha_combination,
                                                                              j_alpha_combination, j_alpha_combination)]) \
                    + 0.5 * np.einsum("iijj->", two_electron_integrals[np.ix_(i_beta_combination, i_beta_combination,
                                                                              j_beta_combination, j_beta_combination)])
                # <ij | v | ji>, or (ij | ji).
                # i and j must have the same spin, and thus the mixed spin terms are omitted
                exchange_part = \
                    0.5 * np.einsum("ijji->", two_electron_integrals[np.ix_(i_alpha_combination, j_alpha_combination,
                                                                            j_alpha_combination, i_alpha_combination)]) \
                    + 0.5 * np.einsum("ijji->", two_electron_integrals[np.ix_(i_beta_combination, j_beta_combination,
                                                                              j_beta_combination, i_beta_combination)])
                element = one_electron_part + coulomb_part - exchange_part

                hamiltonian_matrix[i, j] += element * total_phase

            if n_alpha_excitation + n_beta_excitation == 1:
                alpha_shared_orbitals = list(set(i_alpha_combination).intersection(set(j_alpha_combination)))
                beta_shared_orbitals = list(set(i_beta_combination).intersection(set(j_beta_combination)))

                concatenated = alpha_shared_orbitals + beta_shared_orbitals

                if n_alpha_excitation == 1:
                    index_a = list(alpha_excitation[0])[0]
                    index_b = list(alpha_excitation[1])[0]
                    # <a i | v | b i>, or (ab | ii)
                    coulomb_submatrix = two_electron_integrals[np.ix_([index_a], [index_b], concatenated, concatenated)]

                    # <a i | v | i b>, or (ai | ib).
                    # a and i must have same spin (a and b are already have the same spin)
                    exchange_submatrix = two_electron_integrals[
                        np.ix_([index_a], alpha_shared_orbitals, alpha_shared_orbitals, [index_b])]

                if n_beta_excitation == 1:
                    index_a = list(beta_excitation[0])[0]
                    index_b = list(beta_excitation[1])[0]
                    coulomb_submatrix = two_electron_integrals[np.ix_([index_a], [index_b], concatenated, concatenated)]
                    exchange_submatrix = two_electron_integrals[
                        np.ix_([index_a], beta_shared_orbitals, beta_shared_orbitals, [index_b])]

                element = \
                    one_electron_integrals[index_a, index_b] \
                    + np.einsum("ijkk->", coulomb_submatrix) - np.einsum("ikkj->", exchange_submatrix)
                hamiltonian_matrix[i, j] += element * total_phase
                hamiltonian_matrix[j, i] += element * total_phase

            if n_alpha_excitation == 2:
                left_excitation, right_excitation = map(list, alpha_excitation)
                # <ab | v | xy> - <ab | v | yx>, or (ax | by) - (ay | bx)
                element = two_electron_integrals[left_excitation[0], right_excitation[0],
                left_excitation[1], right_excitation[1]] \
                          - two_electron_integrals[left_excitation[0], right_excitation[1],
                left_excitation[1], right_excitation[0]]
                hamiltonian_matrix[i, j] += element * total_phase
                hamiltonian_matrix[j, i] += element * total_phase

            if n_beta_excitation == 2:
                left_excitation, right_excitation = map(list, beta_excitation)
                # <ab | v | xy> - <ab | v | yx>, or (ax | by) - (ay | bx)
                element = two_electron_integrals[left_excitation[0], right_excitation[0],
                left_excitation[1], right_excitation[1]] \
                          - two_electron_integrals[left_excitation[0], right_excitation[1],
                left_excitation[1], right_excitation[0]]
                hamiltonian_matrix[i, j] += element * total_phase
                hamiltonian_matrix[j, i] += element * total_phase

            if n_alpha_excitation == 1 and n_beta_excitation == 1:
                a = list(alpha_excitation[0])[0]
                b = list(beta_excitation[0])[0]
                x = list(alpha_excitation[1])[0]
                y = list(beta_excitation[1])[0]

                element = two_electron_integrals[a, x, b, y]

                hamiltonian_matrix[i, j] += element * total_phase
                hamiltonian_matrix[j, i] += element * total_phase

    return hamiltonian_matrix

def ci_direct_diagonalize(one_electron_integrals, two_electron_integrals, n_elecs, n_spin=0):
    return np.linalg.eigvals(ci_hamiltonian(one_electron_integrals, two_electron_integrals, n_elecs, n_spin=n_spin))

#def davidson_diagonalization_direct(hamiltonian_matrix, start_search_dim, n_dim, residue_tol=1e-8,
#                                    max_iter=400):
#    t = np.eye(n_dim, start_search_dim)
#    V = np.zeros((n_dim, n_dim))
#    I = np.eye(n_dim)
#
#    for m in range(start_search_dim, max_iter, start_search_dim):
#        if m <= start_search_dim:
#            for j in range(0, start_search_dim):
#                V[:, j] = t[:, j] / np.linalg.norm(t[:, j])
#            theta_old = 1
#        else:
#            theta_old = theta[:1]
#
#        V[:, :m], R = np.linalg.qr(V[:, :m])
#        T = np.dot(V[:, :m].T, np.dot(hamiltonian_matrix, V[:, :m]))
#        THETA, S = np.linalg.eigh(T)
#        idx = THETA.argsort()
#        theta = THETA[idx]
#        s = S[:, idx]
#        for j in range(0, start_search_dim):
#            w = np.dot((hamiltonian_matrix - theta[j] * I), np.dot(V[:, :m], s[:, j]))
#            q = w / (theta[j] - hamiltonian_matrix[j, j])
#            V[:, (m + j)] = q
#        norm = np.linalg.norm(theta[:1] - theta_old)
#        if norm < residue_tol:
#            return theta[:1]
#            break
#

def davidson_diagonalization_direct(hamiltonian_matrix, start_dim, max_dim, residue_tol=1e-8, max_iter=400):
    n_dim = hamiltonian_matrix.shape[0]
    V = np.zeros((n_dim, max_dim))
    I = np.eye(n_dim)

    for m in range(start_dim, max_iter, start_dim):
        if m <= start_dim:
            for j in range(0, start_dim):
                V[:, j] = I[:, j] / np.linalg.norm(I[:, j])
            prev_theta = 1
        else:
            prev_theta = theta[0]

        V[:, :m], _ = np.linalg.qr(V[:, :m])
        T = np.dot(V[:, :m].T, np.dot(hamiltonian_matrix, V[:, :m]))
        eigenvalues, eigenvectors = np.linalg.eigh(T)
        idx = eigenvalues.argsort()
        theta = eigenvalues[idx]
        s = eigenvectors[:, idx]

        for j in range(0, start_dim):
            w = np.dot((hamiltonian_matrix - theta[j] * I), np.dot(V[:, :m], s[:, j]))
            q = w / (theta[j] - hamiltonian_matrix[j, j])
            V[:, (m + j)] = q

        norm_diff = np.abs(theta[0] - prev_theta)
        if norm_diff < residue_tol:
            return theta[0]

    return None  # If convergence is not achieved within max_iter

