import numpy as np
from scipy import sparse
from project.task3 import AdjacencyMatrixFA
from project.task2 import graph_to_nfa, regex_to_dfa
from networkx import MultiDiGraph


def create_bool_vector(vector_size, true_indexes):
    vector = np.zeros(vector_size, dtype=bool)
    for true_ind in true_indexes:
        vector[true_ind] = True
    return vector


def ms_bfs_based_rpq(
    regex: str, graph: MultiDiGraph, start_nodes: set[int], final_nodes: set[int]
) -> set[tuple[int, int]]:

    regex_dfa = regex_to_dfa(regex)
    dfa_adjacency_matrix_fa = AdjacencyMatrixFA(regex_dfa)


    graph_nfa = graph_to_nfa(graph, start_nodes, final_nodes)
    nfa_adjacency_matrix_fa = AdjacencyMatrixFA(graph_nfa)

    k = dfa_adjacency_matrix_fa.states_count
    n = nfa_adjacency_matrix_fa.states_count

    dfa_start_state_ind = list(dfa_adjacency_matrix_fa.start_states)[0]
    nfa_start_states_ind = nfa_adjacency_matrix_fa.start_states
    nfa_start_states_number = len(nfa_start_states_ind)

    # dfa_index_to_state = {index: state for state, index in dfa_adjacency_matrix_fa.states.items()}
    nfa_index_to_state = {index: state for state, index in nfa_adjacency_matrix_fa.states.items()}


    def create_start_front():
        data = np.ones(nfa_start_states_number, dtype=bool)
        rows = [dfa_start_state_ind + k * j for j in range(nfa_start_states_number)]
        columns = [start_state_ind for start_state_ind in nfa_start_states_ind]

        return sparse.csr_matrix(
            (data, (rows, columns)), shape=(k * nfa_start_states_number, n), dtype=bool
        )


    labels = (
        dfa_adjacency_matrix_fa.matricies.keys()
        & nfa_adjacency_matrix_fa.matricies.keys()
    )


    dfa_decomposed_matrices = dfa_adjacency_matrix_fa.matricies
    dfa_decomposed_transposed_matrices = {
        key: mat.transpose() for key, mat in dfa_decomposed_matrices.items()
    }

    nfa_decomposed_matrices = nfa_adjacency_matrix_fa.matricies


    def update_front(front):
        decomposed_fronts = {}
        for label in labels:
            decomposed_fronts[label] = front @ nfa_decomposed_matrices[label]
            for ind in range(nfa_start_states_number):
                decomposed_fronts[label][ind * k : (ind + 1) * k] = (
                    dfa_decomposed_transposed_matrices[label]
                    @ decomposed_fronts[label][ind * k : (ind + 1) * k]
                )

        new_front = sparse.csr_matrix((k * nfa_start_states_number, n), dtype=bool)

        for decomposed_front in decomposed_fronts.values():
            new_front += decomposed_front

        return new_front


    current_front = create_start_front()
    visited = sparse.csr_matrix((k * nfa_start_states_number, n), dtype=bool)


    while current_front.count_nonzero() > 0:
        visited += current_front
        current_front = update_front(current_front)
        current_front = current_front > visited

    dfa_final_states_ind = dfa_adjacency_matrix_fa.final_states

    res = set()


    nfa_final_states_ind = nfa_adjacency_matrix_fa.final_states
    nfa_final_states_vector = create_bool_vector(
        nfa_adjacency_matrix_fa.states_count, nfa_final_states_ind
    )

    for i, nfa_start_state_ind in enumerate(nfa_start_states_ind, 0):
        for dfa_final_state_ind in dfa_final_states_ind:
            row = visited.getrow(i * k + dfa_final_state_ind)
            row_vector = create_bool_vector(n, row.indices)

            vector = row_vector & nfa_final_states_vector

            reached_nfa_final_states_ind = np.nonzero(vector)[0]

            for reached_nfa_final_state_ind in reached_nfa_final_states_ind:
                res.add(
                    (
                        nfa_index_to_state[nfa_start_state_ind],
                        nfa_index_to_state[reached_nfa_final_state_ind],
                    )
                )

    return res
