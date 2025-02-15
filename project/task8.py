from itertools import product
import networkx as nx
import scipy.sparse as sp
from pyformlang import rsa, cfg as pycfg
from pyformlang.finite_automaton import NondeterministicFiniteAutomaton, State
from project.task3 import AdjacencyMatrixFA, intersect_automata
from project.task2 import graph_to_nfa
from typing import Set, Tuple


def cfg_to_rsm(cfg: pycfg.CFG) -> rsa.RecursiveAutomaton:
    return ebnf_to_rsm(cfg.to_text())


def ebnf_to_rsm(ebnf: str) -> rsa.RecursiveAutomaton:
    return rsa.RecursiveAutomaton.from_text(ebnf)


def bool_decomposed_rsm(rsm: rsa.RecursiveAutomaton) -> AdjacencyMatrixFA:
    nfa = NondeterministicFiniteAutomaton()

    for nonterminal, box in rsm.boxes.items():
        box_dfa = box.dfa

        for s in box_dfa.final_states | box_dfa.start_states:
            if s in box_dfa.start_states:
                nfa.add_start_state(State((nonterminal, s)))
            if s in box_dfa.final_states:
                nfa.add_final_state(State((nonterminal, s)))

        for edge in box_dfa.to_networkx().edges(data="label"):
            nfa.add_transition(
                State((nonterminal, edge[0])), edge[2], State((nonterminal, edge[1]))
            )

    return AdjacencyMatrixFA(nfa)


def __compute_closure(decomposed_rsa, decomposed_graph, rsm):
    last_nonzero_number = 0
    current_nonzero_number = None

    while last_nonzero_number != current_nonzero_number:
        last_nonzero_number = current_nonzero_number
        intersection = intersect_automata(decomposed_rsa, decomposed_graph)

        transitive_closure = intersection.transitive_closure()

        for row_index, column_index in zip(*transitive_closure.nonzero()):
            row_state = intersection.num_to_state[row_index]
            column_state = intersection.num_to_state[column_index]

            row_inside_state, row_graph_state = row_state.value
            row_symbol, row_rsm_state = row_inside_state.value
            column_inside_state, column_graph_state = column_state.value
            column_symbol, column_rsm_state = column_inside_state.value

            if (
                row_symbol == column_symbol
                and row_rsm_state in rsm.boxes[row_symbol].dfa.start_states
                and column_rsm_state in rsm.boxes[row_symbol].dfa.final_states
            ):
                row_graph_index = decomposed_graph.states_to_num[row_graph_state]
                column_graph_index = decomposed_graph.states_to_num[column_graph_state]

                decomposed_graph.adj_matrices[row_symbol][
                    row_graph_index, column_graph_index
                ] = True

        current_nonzero_number = sum(
            decomposed_graph.adj_matrices[nonterminal].count_nonzero()
            for nonterminal in decomposed_graph.adj_matrices
        )


def tensor_based_cfpq(
    rsm: rsa.RecursiveAutomaton,
    graph: nx.DiGraph,
    start_nodes: Set[int] | None = None,
    final_nodes: Set[int] | None = None,
    matrix_type=sp.csr_matrix,
) -> Set[Tuple[int, int]]:
    decomposed_rsa = bool_decomposed_rsm(rsm)
    decomposed_graph = AdjacencyMatrixFA(
        graph_to_nfa(graph, start_nodes, final_nodes), matrix_type=matrix_type
    )

    __compute_closure(decomposed_rsa, decomposed_graph, rsm)

    answer = {
        (decomposed_graph.num_to_state[n], decomposed_graph.num_to_state[m])
        for n, m in product(
            decomposed_graph.start_states, decomposed_graph.final_states
        )
        if decomposed_graph.adj_matrices[rsm.initial_label][n, m]
    }
    return answer
