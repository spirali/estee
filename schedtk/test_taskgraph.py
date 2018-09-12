
from .taskgraph import TaskGraph


def test_is_descendant():
    graph = TaskGraph()

    n1, n2, n3, n4, n5 = [graph.new_task() for i in range(5)]

    n1.add_input(n2)
    n1.add_input(n4)
    n2.add_input(n3)
    n2.add_input(n4)

    assert not n1.is_predecessor_of(n4)
    assert n4.is_predecessor_of(n1)
    assert not n2.is_predecessor_of(n4)
    assert n4.is_predecessor_of(n2)
    assert n2.is_predecessor_of(n1)
    assert not n1.is_predecessor_of(n5)
    assert not n5.is_predecessor_of(n1)
    assert not n1.is_predecessor_of(n1)