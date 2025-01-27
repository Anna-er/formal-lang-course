from copy import copy
from functools import reduce
from typing import Generic, TypeVar
from networkx import MultiDiGraph
from pyformlang.finite_automaton import Symbol
import scipy.sparse as sp

from project.task2 import regex_to_dfa, graph_to_nfa
from project.task3 import AdjacencyMatrixFA

Matrix = TypeVar("Matrix")


class MsBfsRpq(Generic[Matrix]):
    __matrix_type: Matrix
    __adj_dfa: AdjacencyMatrixFA
    __adj_nfa: AdjacencyMatrixFA
    __shift: int
    __united_symbols: set[Symbol]

    def __init__(
        self,
        adj_dfa: AdjacencyMatrixFA,
        adj_nfa: AdjacencyMatrixFA,
        matrix_type=sp.csr_matrix,
    ):
        self.__matrix_type = matrix_type
        self.__adj_dfa = adj_dfa
        self.__adj_nfa = adj_nfa
        self.__start_states_list = list(adj_nfa.start_states)
        self.__shift = self.__adj_dfa.states_number
        self.__united_symbols = set(self.__adj_dfa.adj_matrices.keys()).intersection(
            self.__adj_nfa.adj_matrices.keys()
        )
        self.__permutation_matrices = {
            symbol: sp.block_diag(
                [
                    adj_dfa.adj_matrices[symbol].transpose()
                    for _ in self.__start_states_list
                ]
            )
            for symbol in self.__united_symbols
        }

    def __update_front(self, front_right: Matrix) -> Matrix:
        def front_mul_matrix(cur_front, symbol) -> Matrix:
            mul = cur_front @ self.__adj_nfa.adj_matrices[symbol]
            return self.__permutation_matrices[symbol] @ mul

        updated_front = reduce(
            lambda vector, matrix: vector + front_mul_matrix(front_right, matrix),
            self.__united_symbols,
            self.__matrix_type(front_right.shape, dtype=bool),
        )

        return updated_front

    def __get_init_front(self) -> Matrix:
        vectors = []
        for nfa_state_num in range(len(self.__adj_nfa.start_states)):
            right_vector = self.__matrix_type(
                (self.__shift, self.__adj_nfa.states_number), dtype=bool
            )
            for i in self.__adj_dfa.start_states:
                right_vector[i, self.__start_states_list[nfa_state_num]] = True

            vectors.append(right_vector)

        return sp.vstack(vectors)

    def __visited_to_result(self, visited: Matrix):
        result = set()
        for left, nfa_state in zip(*visited.nonzero()):
            if (
                left % self.__shift in self.__adj_dfa.final_states
                and nfa_state in self.__adj_nfa.final_states
            ):
                result.add(
                    (
                        self.__adj_nfa.num_to_state[
                            self.__start_states_list[left // self.__shift]
                        ],
                        self.__adj_nfa.num_to_state[nfa_state],
                    )
                )
        return result

    def __ms_bfs(self):
        front_right = self.__get_init_front()
        visited = copy(front_right)

        while front_right.count_nonzero():
            front_right = self.__update_front(front_right)
            front_right = front_right > visited
            visited += front_right

        return self.__visited_to_result(visited)

    def __call__(self) -> set[tuple[int, int]]:
        return self.__ms_bfs()


def ms_bfs_based_rpq(
    regex: str,
    graph: MultiDiGraph,
    start_nodes: set[int],
    final_nodes: set[int],
    matrix_type=sp.csr_matrix,
) -> set[tuple[int, int]]:
    regex_dfa = regex_to_dfa(regex)
    adj_dfa = AdjacencyMatrixFA(regex_dfa, matrix_type)
    graph_nfa = graph_to_nfa(graph, start_nodes, final_nodes)
    adj_nfa = AdjacencyMatrixFA(graph_nfa, matrix_type)

    result = MsBfsRpq(adj_dfa, adj_nfa, matrix_type)()

    return result
