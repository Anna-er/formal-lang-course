from typing import Any, Set, Tuple
import networkx as nx
from pyformlang.cfg import CFG, Terminal, Variable
from scipy.sparse import csr_matrix
from project.task6 import cfg_to_weak_normal_form


def matrix_based_cfpq(
    cfg: CFG,
    graph: nx.DiGraph,
    start_nodes: Set[int] = None,
    final_nodes: Set[int] = None,
) -> Set[Tuple[int, int]]:
    wnf: CFG = cfg_to_weak_normal_form(cfg)
    nodes_amount: int = graph.number_of_nodes()
    node_to_index: dict[Any, int] = {
        node: idx for idx, node in enumerate(graph.nodes())
    }
    index_to_node: dict[int, Any] = {idx: node for node, idx in node_to_index.items()}

    var_mats: dict[Variable, csr_matrix] = {}
    for u, v, data in graph.edges(data=True):
        label = data.get("label")
        if label:
            for p in wnf.productions:
                if len(p.body) == 1 and isinstance(p.body[0], Terminal):
                    terminal = p.body[0].value
                    if terminal == label:
                        head = p.head
                        if head not in var_mats:
                            var_mats[head] = csr_matrix(
                                (nodes_amount, nodes_amount), dtype=bool
                            )
                        var_mats[head][node_to_index[u], node_to_index[v]] = True

    # Учитываем nullable символы для диагональных элементов
    nullable = wnf.get_nullable_symbols()
    for node in graph.nodes:
        for var in nullable:
            var = Variable(var.value)
            if var not in var_mats:
                var_mats[var] = csr_matrix((nodes_amount, nodes_amount), dtype=bool)
            var_mats[var][node_to_index[node], node_to_index[node]] = True

    # Обработка правил вида A -> B C с добавлением новых достижимых пар
    added = True
    while added:
        added = False
        for p in wnf.productions:
            if len(p.body) == 2:
                B = Variable(p.body[0].value)
                C = Variable(p.body[1].value)
                head = p.head
                if B in var_mats and C in var_mats:
                    if head not in var_mats:
                        var_mats[head] = csr_matrix(
                            (nodes_amount, nodes_amount), dtype=bool
                        )

                    new_mat = var_mats[B] @ var_mats[C]
                    new_mat_coo = new_mat.tocoo()

                    for u, v, value in zip(
                        new_mat_coo.row, new_mat_coo.col, new_mat_coo.data
                    ):
                        if value and not var_mats[head][u, v]:
                            var_mats[head][u, v] = True
                            added = True

    # Формируем множество достижимых пар для символа стартовой грамматики
    pairs = set()
    start_symbol = wnf.start_symbol
    if start_symbol in var_mats:
        final_mat = var_mats[start_symbol].tocoo()
        for u_idx, v_idx in zip(final_mat.row, final_mat.col):
            u = index_to_node[u_idx]
            v = index_to_node[v_idx]
            if (not start_nodes or u in start_nodes) and (
                not final_nodes or v in final_nodes
            ):
                pairs.add((u, v))

    return pairs
