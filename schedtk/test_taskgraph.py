
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


def test_task_graph_copy(plan1):
    task_graph = plan1.copy()

    for t1, t2 in zip(task_graph.tasks, plan1.tasks):
        assert id(t1) != id(t2)
        assert t1.id == t2.id
        assert len(t1.inputs) == len(t2.inputs)
        for i1, i2 in zip(t1.inputs, t2.inputs):
            assert id(i1) != id(i2)
            assert i1.id == i2.id

        for i1, i2 in zip(sorted(t1.consumers, key=lambda t: t.id),
                          sorted(t2.consumers, key=lambda t: t.id)):
            assert id(i1) != id(i2)
            assert i1.id == i2.id


def test_task_graph_merge(plan1):

    task_graph = TaskGraph.merge([plan1, plan1, plan1, plan1])
    assert task_graph.task_count == 4 * plan1.task_count

    for i, t in enumerate(task_graph.tasks):
        print(i)
        assert t.id == i
        assert len(t.inputs) == len(plan1.tasks[i % plan1.task_count].inputs)
        assert t.duration == plan1.tasks[i % plan1.task_count].duration
        assert id(t) != id(plan1.tasks[i % plan1.task_count])
