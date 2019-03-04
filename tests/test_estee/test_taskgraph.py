
from estee.common import Task, TaskGraph


def test_is_descendant():
    graph = TaskGraph()

    n1, n2, n3, n4, n5 = [graph.new_task(output_size=1 if i != 5 else None) for i in range(5)]

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

    task_graph.validate()

    assert set(task_graph.tasks) == set(plan1.tasks)

    for task_id in task_graph.tasks:
        t1 = task_graph.tasks[task_id]
        t2 = plan1.tasks[task_id]
        assert id(t1) != id(t2)
        assert t1.id == t2.id
        assert len(t1.inputs) == len(t2.inputs)
        assert len(t1.outputs) == len(t2.outputs)

        for i1, i2 in zip(t1.inputs, t2.inputs):
            assert id(i1) != id(i2)
            assert i1.id == i2.id

        for o1, o2 in zip(t1.outputs, t2.outputs):
            assert id(o1) != id(o2)
            assert o1.id == o2.id

            for i1, i2 in zip(sorted(o1.consumers, key=lambda t: t.id),
                              sorted(o2.consumers, key=lambda t: t.id)):
                assert id(i1) != id(i2)
                assert i1.id == i2.id


def test_task_graph_merge(plan1):

    task_graph = TaskGraph.merge([plan1, plan1, plan1, plan1])
    assert task_graph.task_count == 4 * plan1.task_count

    task_graph.validate()

    for i, t in task_graph.tasks.items():
        assert t.id == i
        assert len(t.inputs) == len(plan1.tasks[i % plan1.task_count].inputs)
        assert t.duration == plan1.tasks[i % plan1.task_count].duration
        assert id(t) != id(plan1.tasks[i % plan1.task_count])


def test_task_graph_export_dot(plan1, tmpdir):
    name = str(tmpdir.join("test.dot"))
    plan1.write_dot(name)
    with open(name) as f:
        assert f.read().count("\n") == 35


def test_task_copy():
    task = Task(123, cpus=2, duration=5, expected_duration=10, outputs=[2, 3])
    for o in task.outputs:
        o.expected_size = o.size + 1

    copy = task.simple_copy()
    assert copy.id == 123
    assert copy.cpus == task.cpus
    assert copy.duration == task.duration
    assert copy.expected_duration == task.expected_duration
    assert len(copy.outputs) == len(task.outputs)

    for (orig, copied) in zip(task.outputs, copy.outputs):
        assert orig.size == copied.size
        assert orig.expected_size == copied.expected_size
