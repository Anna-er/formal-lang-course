from pyformlang.finite_automaton import (
    DeterministicFiniteAutomaton, 
    NondeterministicFiniteAutomaton, 
    State, Symbol
)
from pyformlang.regular_expression import Regex
from typing import Set
from networkx import MultiDiGraph


def regex_to_dfa(regex: str) -> DeterministicFiniteAutomaton:
    regex_obj = Regex(regex)
    nfa = regex_obj.to_epsilon_nfa()
    dfa = nfa.minimize()
    return dfa


def graph_to_nfa(
    graph: MultiDiGraph, 
    start_states: Set[int] = None, 
    final_states: Set[int] = None
) -> NondeterministicFiniteAutomaton:
    nfa = NondeterministicFiniteAutomaton()

    for u, v, data in graph.edges(data=True):
        symbol = Symbol(data.get("label", ""))
        nfa.add_transition(State(u), symbol, State(v))

    if not start_states:
        start_states = set(graph.nodes)
    for state in start_states:
        nfa.add_start_state(State(state))

    if not final_states:
        final_states = set(graph.nodes)
    for state in final_states:
        nfa.add_final_state(State(state))

    return nfa
