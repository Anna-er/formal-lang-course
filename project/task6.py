import networkx as nx
import pyformlang
from pyformlang.cfg import Production, Variable, Epsilon, CFG, Terminal


def cfg_to_weak_normal_form(cfg: pyformlang.cfg.CFG) -> pyformlang.cfg.CFG:
    updated_productions = set(cfg.to_normal_form().productions)
    for nullable_var in cfg.get_nullable_symbols():
        updated_productions.add(Production(Variable(nullable_var.value), [Epsilon()]))

    return CFG(
        start_symbol=cfg.start_symbol, productions=updated_productions
    ).remove_useless_symbols()


def hellings_based_cfpq(
    cfg: pyformlang.cfg.CFG,
    graph: nx.DiGraph,
    start_nodes: set[int] = None,
    final_nodes: set[int] = None,
) -> set[tuple[int, int]]:
    weak_cnf_cfg = cfg_to_weak_normal_form(cfg)

    term_to_vars = {}
    pair_to_vars = {}

    for production in weak_cnf_cfg.productions:
        if len(production.body) == 1 and isinstance(production.body[0], Terminal):
            term_to_vars.setdefault(production.body[0], set()).add(production.head)
        elif len(production.body) == 2:
            pair_to_vars.setdefault(tuple(production.body), set()).add(production.head)

    graph_edges = {
        (v1, Terminal(label), v2) for v1, v2, label in graph.edges.data("label")
    }

    new_edges = {
        (v1, var, v2)
        for v1, term, v2 in graph_edges
        if term in term_to_vars
        for var in term_to_vars[term]
    } | {(v, var, v) for v in graph.nodes for var in weak_cnf_cfg.get_nullable_symbols()}

    queue = list(new_edges)

    def update_queue(e1, e2, buff):
        s1, a, f1 = e1
        s2, b, f2 = e2
        if f1 == s2:
            pair = (a, b)
            if pair in pair_to_vars:
                for var in pair_to_vars[pair]:
                    new_edge = (s1, var, f2)
                    if new_edge not in new_edges:
                        buff.add(new_edge)
                        queue.append(new_edge)

    while queue:
        edge = queue.pop(0)
        buffer = set()
        for e in new_edges:
            update_queue(edge, e, buffer)
            update_queue(e, edge, buffer)
        new_edges |= buffer

    start_var = weak_cnf_cfg.start_symbol
    return {
        (v1, v2)
        for v1, var, v2 in new_edges
        if v1 in start_nodes and var == start_var and v2 in final_nodes
    }
