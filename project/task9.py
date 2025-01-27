from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Set, Tuple, Self, TypeVar
from pyformlang.finite_automaton import Symbol
import networkx as nx
from pyformlang import rsa


@dataclass(frozen=True)
class RsmState:
    var: Symbol
    sub_state: str


GSSNodeT = TypeVar("GSSNodeT", bound="GSSNode")


@dataclass(frozen=True)
class SPPFNode:
    gssn: GSSNodeT
    state: RsmState
    node: int


class GSSNode:
    state: RsmState
    node: int
    edges: Dict[RsmState, Set[Self]]
    pop_set: Set[int]

    def __init__(self, state: RsmState, node: int):
        self.state = state
        self.node = node
        self.edges = {}
        self.pop_set = set()

    def pop(self, cur_node: int) -> Set[SPPFNode]:
        res_set = set()

        if cur_node not in self.pop_set:
            res_set |= {
                SPPFNode(gss_node, new_st, cur_node)
                for new_st, value in self.edges.items()
                for gss_node in value
            }
            self.pop_set.add(cur_node)
        return res_set

    def add_edge(self, ret_st: RsmState, ptr: Self) -> Set[SPPFNode]:
        res_set = set()

        st_edges = self.edges.get(ret_st, set())
        if ptr not in st_edges:
            st_edges.add(ptr)
            res_set |= {SPPFNode(ptr, ret_st, cur_node) for cur_node in self.pop_set}

        self.edges[ret_st] = st_edges

        return res_set


class GSStack:
    body: Dict[Tuple[RsmState, int], GSSNode]

    def __init__(self):
        self.body = {}

    def get_node(self, rsm_st: RsmState, node: int):
        return self.body.setdefault((rsm_st, node), GSSNode(rsm_st, node))


@dataclass
class RsmStateData:
    term_edges: Dict[Symbol, RsmState]
    var_edges: Dict[Symbol, Tuple[RsmState, RsmState]]
    is_final: bool


class GllCFPQSolver:
    nodes2edges: Dict[int, Dict[Symbol, Set[int]]]
    rsmstate2data: Dict[Symbol, Dict[str, RsmStateData]]
    start_rstate: RsmState
    rsm: rsa.RecursiveAutomaton
    graph: nx.DiGraph
    gss: GSStack
    accept_gssnode: GSSNode
    unprocessed: Set[SPPFNode]
    added: Set[SPPFNode]

    @staticmethod
    def _init_graph_data(graph: nx.DiGraph):
        nodes2edges = defaultdict(lambda: defaultdict(set))

        for from_n, to_n, symb in graph.edges(data="label"):
            if symb is not None:
                nodes2edges[from_n][symb].add(to_n)
        return nodes2edges

    @staticmethod
    def _init_rsm_data(rsm: rsa.RecursiveAutomaton):
        rsm_state2data = defaultdict(dict)

        for var in rsm.boxes:
            box = rsm.boxes[var]
            gbox = box.dfa.to_networkx()

            for sub_state in gbox.nodes:
                rsm_state2data[var][sub_state] = RsmStateData(
                    {}, {}, sub_state in box.dfa.final_states
                )

            edges = gbox.edges(data="label")
            for from_st, to_st, symb in edges:
                if symb is not None:
                    if Symbol(symb) not in rsm.boxes:
                        rsm_state2data[var][from_st].term_edges[symb] = RsmState(
                            var, to_st
                        )
                    else:
                        bfa = rsm.boxes[Symbol(symb)].dfa
                        rsm_state2data[var][from_st].var_edges[symb] = (
                            RsmState(Symbol(symb), bfa.start_state.value),
                            RsmState(var, to_st),
                        )
        return rsm_state2data

    def __init__(
        self,
        rsm: rsa.RecursiveAutomaton,
        graph: nx.DiGraph,
    ):
        self.nodes2edges = self._init_graph_data(graph)
        self.rsmstate2data = self._init_rsm_data(rsm)

        start_symb = rsm.initial_label
        start_fa = rsm.boxes[start_symb].dfa
        self.start_rstate = RsmState(start_symb, start_fa.start_state.value)

        self.rsm = rsm
        self.graph = graph

        self.gss = GSStack()
        self.accept_gssnode = self.gss.get_node(RsmState(Symbol("$"), "fin"), -1)

        self.unprocessed: Set[SPPFNode] = set()
        self.added: Set[SPPFNode] = set()

    def add_sppf_nodes(self, snodes: Set[SPPFNode]):
        snodes.difference_update(self.added)

        self.added |= snodes
        self.unprocessed |= snodes

    def filter_poped_nodes(
        self, snodes: Set[SPPFNode], prev_snode: SPPFNode
    ) -> Tuple[Set[SPPFNode], Set[Tuple[int, int]]]:
        node_res_set = set()
        start_fin_res_set = set()

        for sn in snodes:
            if sn.gssn == self.accept_gssnode:
                start_node = prev_snode.gssn.node
                fin_node = sn.node
                start_fin_res_set.add((start_node, fin_node))
            else:
                node_res_set.add(sn)

        return node_res_set, start_fin_res_set

    def step(self, sppfnode: SPPFNode) -> Set[Tuple[int, int]]:
        rsm_dat = self.rsmstate2data[sppfnode.state.var][sppfnode.state.sub_state]

        def term_step():
            rsm_terms = rsm_dat.term_edges
            graph_terms = self.nodes2edges[sppfnode.node]
            for term in rsm_terms:
                if term in graph_terms:
                    new_sppf_nodes = set()
                    rsm_new_st = rsm_terms[term]
                    graph_new_nodes = graph_terms[term]
                    new_sppf_nodes |= {
                        SPPFNode(sppfnode.gssn, rsm_new_st, gn)
                        for gn in graph_new_nodes
                    }

                    self.add_sppf_nodes(new_sppf_nodes)

        def var_step() -> Set[Tuple[int, int]]:
            start_fin_set = set()
            for edge in rsm_dat.var_edges.values():
                var_start_rsm_st, ret_rsm_st = edge

                inner_gss_node = self.gss.get_node(var_start_rsm_st, sppfnode.node)
                post_pop_sppf_nodes = inner_gss_node.add_edge(ret_rsm_st, sppfnode.gssn)

                post_pop_sppf_nodes, sub_start_fin_set = self.filter_poped_nodes(
                    post_pop_sppf_nodes, sppfnode
                )

                self.add_sppf_nodes(post_pop_sppf_nodes)
                self.add_sppf_nodes(
                    {SPPFNode(inner_gss_node, var_start_rsm_st, sppfnode.node)}
                )

                start_fin_set |= sub_start_fin_set

            return start_fin_set

        def pop_step() -> Set[Tuple[int, int]]:
            new_sppf_nodes = sppfnode.gssn.pop(sppfnode.node)
            new_sppf_nodes, start_fin_set = self.filter_poped_nodes(
                new_sppf_nodes, sppfnode
            )
            self.add_sppf_nodes(new_sppf_nodes)
            return start_fin_set

        term_step()
        res_set = var_step()

        if rsm_dat.is_final:
            res_set |= pop_step()

        return res_set

    def __call__(
        self,
        from_nodes: Set[int],
        to_nodes: Set[int],
    ) -> Set[Tuple[int, int]]:
        reach_set = set()
        for snode in from_nodes:
            gssn = self.gss.get_node(self.start_rstate, snode)
            gssn.add_edge(RsmState(Symbol("$"), "fin"), self.accept_gssnode)

            self.add_sppf_nodes({SPPFNode(gssn, self.start_rstate, snode)})

        while self.unprocessed:
            reach_set |= self.step(self.unprocessed.pop())

        filtered_set = {st_fin for st_fin in reach_set if st_fin[1] in to_nodes}
        return filtered_set


def gll_based_cfpq(
    rsm: rsa.RecursiveAutomaton,
    graph: nx.DiGraph,
    start_nodes: Set[int] | None = None,
    final_nodes: Set[int] | None = None,
) -> Set[Tuple[int, int]]:
    solver = GllCFPQSolver(rsm, graph)
    result = solver(
        start_nodes if start_nodes else graph.nodes(),
        final_nodes if final_nodes else graph.nodes(),
    )
    return result
