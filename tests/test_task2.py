from project.task2 import regex_to_dfa


def test_regex_to_dfa():
    regex = "a|b|c*"

    dfa = regex_to_dfa(regex)

    assert dfa.accepts("a")
    assert dfa.accepts("b")
    assert dfa.accepts("ccc")
    assert dfa.accepts("")

    assert not dfa.accepts("aa")
    assert not dfa.accepts("d")
    assert not dfa.accepts("ab")

