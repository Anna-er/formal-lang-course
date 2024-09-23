from project.task2 import NondeterministicFiniteAutomaton, regex_to_dfa, graph_to_nfa
import scipy.sparse as sp
from collections.abc import Iterable
from pyformlang.finite_automaton import Symbol
from functools import reduce
from networkx import MultiDiGraph
import itertools


class AdjacencyMatrixFA:
    def __init__(self, automation: NondeterministicFiniteAutomaton = None):
        self.matricies = {}

        if automation is None:
            self.states = {}
            self.alphabet = set()
            self.start_states = set()
            self.final_states = set()
            self.states_count = 0
            return

        self.states = {state: idx for idx, state in enumerate(automation.states)}
        self.states_count = len(self.states)
        self.alphabet = automation.symbols

        graph = automation.to_networkx()

        for symbol in self.alphabet:
            self.matricies[symbol] = sp.csr_matrix(
                (self.states_count, self.states_count), dtype=bool
            )

        for u, v, label in graph.edges(data="label"):
            if not (str(u).startswith("starting_") or str(v).startswith("starting_")):
                self.matricies[label][self.states[u], self.states[v]] = True

        self.start_states = {self.states[key] for key in automation.start_states}
        self.final_states = {self.states[key] for key in automation.final_states}

    def accepts(self, word: Iterable[Symbol]) -> bool:
        """Check if the given word is accepted by the automaton."""

        # Initialize the stack of configurations
        configurations = [(list(word), state) for state in self.start_states]

        while configurations:
            # Get the next configuration
            word_rest, current_state = configurations.pop()

            # If the word is empty and the current state is final, accept
            if not word_rest and current_state in self.final_states:
                return True

            # Iterate over the next possible states
            for next_state in self.states.values():
                # If the transition is possible, add the new configuration to the stack
                if self.matricies[word_rest[0]][current_state, next_state]:
                    configurations.append((word_rest[1:], next_state))

        # If no accepting configuration was found, reject
        return False

    def is_empty(self) -> bool:
        """Check if the automaton is empty (accepts no words)."""
        transitive_closure = self.transitive_closure()

        for start_state in self.start_states:
            for final_state in self.final_states:
                if transitive_closure[start_state, final_state]:
                    return False

        return True

    def transitive_closure(self) -> sp.csr_matrix:
        """Compute the transitive closure of the automaton."""

        # Initialize the reachability matrix
        reachability = sp.csr_matrix((self.states_count, self.states_count), dtype=bool)

        # The diagonal of the reachability matrix is always True
        reachability.setdiag(True)

        # If the automaton is empty, return the diagonal matrix
        if not self.matricies:
            return reachability

        # Compute the reachability matrix
        for label, matrix in self.matricies.items():
            reachability = reachability + matrix

        # Compute the transitive closure using Warshall's algorithm
        for intermediate in range(self.states_count):
            for start in range(self.states_count):
                for end in range(self.states_count):
                    reachability[start, end] = (
                        reachability[start, end]
                        or (reachability[start, intermediate] and reachability[intermediate, end])
                    )

        return reachability



def intersect_automata(
    first_automaton: AdjacencyMatrixFA, second_automaton: AdjacencyMatrixFA
) -> AdjacencyMatrixFA:
    """
    Intersects two automata and returns the result as a new automaton.
    """

    first_matrices = first_automaton.matricies
    second_matrices = second_automaton.matricies

    intersect_automaton = AdjacencyMatrixFA()

    intersect_automaton.states_count = (
        first_automaton.states_count * second_automaton.states_count
    )

    for label in first_matrices.keys():
        if label not in second_matrices:
            continue

        first_matrix = first_matrices[label]
        second_matrix = second_matrices[label]
        intersect_matrix = sp.kron(first_matrix, second_matrix, format="csr")

        intersect_automaton.matricies[label] = intersect_matrix

    intersect_automaton.states = {
        (state1, state2): (
            first_automaton.states[state1] * second_automaton.states_count
            + second_automaton.states[state2]
        )
        for state1, state2 in itertools.product(
            first_automaton.states.keys(), second_automaton.states.keys()
        )
    }

    intersect_automaton.start_states = [
        (start_state1 * second_automaton.states_count + start_state2)
        for start_state1, start_state2 in itertools.product(
            first_automaton.start_states, second_automaton.start_states
        )
    ]

    intersect_automaton.final_states = [
        (final_state1 * second_automaton.states_count + final_state2)
        for final_state1, final_state2 in itertools.product(
            first_automaton.final_states, second_automaton.final_states
        )
    ]

    intersect_automaton.alphabet = first_automaton.alphabet.union(
        second_automaton.alphabet
    )

    return intersect_automaton


def tensor_based_rpq(
    regex_str: str, graph: MultiDiGraph, start_nodes: set[int], final_nodes: set[int]
) -> set[tuple[int, int]]:
    regex_automaton = AdjacencyMatrixFA(regex_to_dfa(regex_str))
    graph_automaton = AdjacencyMatrixFA(graph_to_nfa(graph, start_nodes, final_nodes))

    intersect_automaton = intersect_automata(regex_automaton, graph_automaton)

    transitive_closure = intersect_automaton.transitive_closure()

    regex_start_nodes = [
        state
        for state in regex_automaton.states
        if regex_automaton.states[state] in regex_automaton.start_states
    ]
    regex_final_nodes = [
        state
        for state in regex_automaton.states
        if regex_automaton.states[state] in regex_automaton.final_states
    ]

    return {
        (start_node, final_node)
        for start_node in start_nodes
        for final_node in final_nodes
        for regex_start_node in regex_start_nodes
        for regex_final_node in regex_final_nodes
        if transitive_closure[
            intersect_automaton.states[(regex_start_node, start_node)],
            intersect_automaton.states[(regex_final_node, final_node)],
        ]
    }
