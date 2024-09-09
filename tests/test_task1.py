from pathlib import Path
from project import task1


class TestGraphUtils:
    def test_gzip(self):
        info = task1.graph_data("gzip")
        gzip = (2687, 2293, ["d", "a"])
        assert info == gzip
    def test_bzip(self):
        info = task1.graph_data("bzip")
        bzip = (632, 556, ["d", "a"])
        assert info == bzip

    def test_writing_to_file(self):
        file_name = "graph.dot"
        graph_info = (12, 65, ("a", "b"))
        task1.create_labeled_two_cycles_graph(
            graph_info[0], graph_info[1], graph_info[2], file_name
        )

        file_path = Path(file_name)
        assert file_path.is_file()

        file_path.unlink()
