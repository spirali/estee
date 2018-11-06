from .dax import load
from .taskgraph import TaskGraph


def test_load_plan1():
    tg = load("plan1.dax")
    assert isinstance(tg, TaskGraph)
    assert len(tg.tasks) == 8
    tasks = {}
    for t in tg.tasks:
        ns = t.name.split("_")
        assert t.duration == float(ns[1])
        assert t.size == int(ns[2])
        tasks[t.name] = t
    assert len(tasks["a1_2_1"].inputs) == 0
    assert len(tasks["a1_2_1"].consumers) == 1
    assert len(tasks["a2_3_3"].inputs) == 0
    assert len(tasks["a2_3_3"].consumers) == 1
    assert len(tasks["a3_2_1"].inputs) == 1
    assert len(tasks["a3_2_1"].consumers) == 2
    assert len(tasks["a4_1_6"].inputs) == 0
    assert len(tasks["a4_1_6"].consumers) == 2
    assert len(tasks["a5_1_1"].inputs) == 3
    assert len(tasks["a5_1_1"].consumers) == 1
    assert len(tasks["a6_6_1"].inputs) == 1
    assert len(tasks["a6_6_1"].consumers) == 1
    assert len(tasks["a7_1_2"].inputs) == 0
    assert len(tasks["a7_1_2"].consumers) == 1
    assert len(tasks["a8_1_1"].inputs) == 4
    assert len(tasks["a8_1_1"].consumers) == 0
