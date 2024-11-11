from pyformlang.cfg import CFG
from pyformlang.rsa import RecursiveAutomaton
from networkx import DiGraph
from pyformlang.finite_automaton import (
    NondeterministicFiniteAutomaton,
    State,
    Symbol,
)
from project.task3 import AdjacencyMatrixFA, graph_to_nfa, intersect_automata
from scipy.sparse import csr_array, csr_matrix


def cfg_to_rsm(cfg: CFG) -> RecursiveAutomaton:
    return ebnf_to_rsm(cfg.to_text())


def ebnf_to_rsm(ebnf: str) -> RecursiveAutomaton:
    return RecursiveAutomaton.from_text(ebnf)


def rsm_to_nfa(rsm: RecursiveAutomaton) -> NondeterministicFiniteAutomaton:
    transitions = []
    start_states = set()
    final_states = set()

    for head, box in rsm.boxes.items():
        FA = box.dfa if isinstance(box.dfa, NondeterministicFiniteAutomaton) else None
        if FA is None:
            continue

        for start_state in FA.start_states:
            start_states.add(State((head, start_state)))

        for final_state in FA.final_states:
            final_states.add(State((head, final_state)))

        for u, v, label in FA.to_networkx().edges(data="label"):
            if label:
                transitions.append((State((head, u)), Symbol(label), State((head, v))))

    nfa = NondeterministicFiniteAutomaton()
    nfa.add_transitions(transitions)
    for state in start_states:
        nfa.add_start_state(state)
    for state in final_states:
        nfa.add_final_state(state)

    return nfa


def update_matrices(
    old: AdjacencyMatrixFA, matrices: dict[Symbol, csr_array]
) -> AdjacencyMatrixFA:
    new_adj = AdjacencyMatrixFA(None)
    new_adj.states_count = old.states_count
    new_adj.states = old.states
    new_adj.start_states = old.start_states
    new_adj.final_states = old.final_states
    new_adj.matricies = old.matricies.copy()

    for sym, matrix in matrices.items():
        if sym in new_adj.matricies:
            new_adj.matricies[sym] += matrix
        else:
            new_adj.matricies[sym] = matrix

    return new_adj


def tensor_based_cfpq(
    rsm: RecursiveAutomaton,
    graph: DiGraph,
    start_nodes: set[int] = None,
    final_nodes: set[int] = None,
) -> set[tuple[int, int]]:
    rsm_fa = rsm_to_nfa(rsm)
    graph_fa = graph_to_nfa(graph, start_nodes, final_nodes)

    rsm_adj = AdjacencyMatrixFA(rsm_fa)
    graph_adj = AdjacencyMatrixFA(graph_fa)

    while True:
        intersection_adj: AdjacencyMatrixFA = intersect_automata(graph_adj, rsm_adj)
        closure: csr_matrix = intersection_adj.transitive_closure()

        new_matrix: dict[Symbol, csr_matrix] = {}
        row_i, col_i = closure.nonzero()

        for x_i, y_i in zip(row_i, col_i):
            if x_i not in intersection_adj.states or y_i not in intersection_adj.states:
                continue

            common_from_state: State = intersection_adj.states[x_i]
            common_to_state: State = intersection_adj.states[y_i]

            graph_from_state = common_from_state.value[0]
            graph_to_state = common_to_state.value[0]
            graph_from_state_i = graph_adj.states.get(graph_from_state)
            graph_to_state_i = graph_adj.states.get(graph_to_state)

            rsm_from_state = common_from_state.value[1]
            rsm_to_state = common_to_state.value[1]
            rsm_from_state_i = rsm_adj.states.get(rsm_from_state)
            rsm_to_state_i = rsm_adj.states.get(rsm_to_state)

            if (
                rsm_from_state != rsm_to_state
                or rsm_from_state_i not in rsm_adj.start_states
                or rsm_to_state_i not in rsm_adj.final_states
            ):
                continue

            label = rsm_from_state  # Предполагаем, что `rsm_from_state` — это символ перехода
            if (label not in graph_adj.matricies) or (
                not graph_adj.matricies[label][graph_from_state_i, graph_to_state_i]
            ):
                if label not in new_matrix:
                    new_matrix[label] = csr_array(
                        (graph_adj.states_count, graph_adj.states_count), dtype=bool
                    )
                new_matrix[label][graph_from_state_i, graph_to_state_i] = True

        if not new_matrix:
            break

        graph_adj = update_matrices(graph_adj, new_matrix)

    if rsm.initial_label not in graph_adj.matricies:
        return set()

    res = graph_adj.matricies[rsm.initial_label]
    pairs: set[tuple[int, int]] = set()

    for start_state in start_nodes or graph_fa.start_states:
        for final_state in final_nodes or graph_fa.final_states:
            start_state_i = graph_adj.states.get(start_state)
            final_state_i = graph_adj.states.get(final_state)
            if (
                start_state_i is not None
                and final_state_i is not None
                and res[start_state_i, final_state_i]
            ):
                pairs.add((start_state, final_state))

    return pairs
