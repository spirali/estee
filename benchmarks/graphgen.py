
from schedsim.common.taskgraph import TaskGraph

from clusters import clusters

import itertools
import pandas
import numpy as np
import uuid

def normal(loc, scale):
    return max(0.00000001, np.random.normal(loc, scale))


def exponential(scale):
    return max(0.00000001, np.random.exponential(scale))


def gridcat(count):
    g = TaskGraph()
    opens = [g.new_task("input{}".format(i), duration=normal(0.01, 0.001),
                        output_size=normal(300, 25)) for i in range(count)]
    hashes = []
    for i in range(count):
        o1 = opens[i]
        for j in range(i + 1, count):
            o2 = opens[j]
            sz = o1.output.size + o2.output.size
            d = normal(0.2, 0.01) + sz / 1000.0
            cat = g.new_task("cat", duration=d, output_size=sz)
            cat.add_input(o1)
            cat.add_input(o2)
            d = normal(0.2, 0.01) + sz / 500.0
            makehash = g.new_task("hash", duration=d, output_size=16 / 1024 / 1024)
            makehash.add_input(cat)
            hashes.append(makehash.output)
    m = g.new_task("merge", duration=0.1, output_size=16 / 1024 / 1024)
    m.add_inputs(hashes)
    return g


def plain1n(count):
    g = TaskGraph()
    for i in range(count):
        t = np.random.choice([10, 20, 60, 180])
        g.new_task("t{}".format(i), duration=normal(t, t / 10), expected_duration=t, cpus=1)
    return g


def plain1e(count):
    g = TaskGraph()
    for i in range(count):
        g.new_task("t{}".format(i), duration=exponential(60), expected_duration=60, cpus=1)
    return g


def plain1cpus(count):
    g = TaskGraph()
    for i in range(count):
        t = np.random.choice([10, 20, 60, 180])
        g.new_task("t{}".format(i), duration=normal(t, t / 10), expected_duration=t,
                   cpus=np.random.randint(1, 4))
    return g

"""
def plain2(count):
    g = TaskGraph()
    for i in range(count):
        t1 = g.new_task("a{}".format(i), duration=normal(5, 2), output_size=40)
        t2 = g.new_task("b{}".format(i), duration=normal(120, 20), output_size=120, cpus=4)
        t2.add_input(t1)
        t3 = g.new_task("c{}".format(i), duration=normal(30, 1))
        t3.add_input(t2)
    return g


def merges1(count):
    g = TaskGraph()

    tasks1 = g.new_task("a{}".format(i), duration=normal(15, 3), output_size=normal(10, 3))
    for i in range(count):
        t = g.new_task("b{}".format(i), duration=normal(15, 2), output_size=normal(20, 3))
        t.add_input(tasks1[i])
        t.add_input(tasks1[(i + 1) % count])

def merges2(count):
    g = TaskGraph()

    tasks1 = g.new_task("a{}".format(i), duration=normal(15, 3), output_size=normal(10, 3))
    for i in range(count):
        t = g.new_task("b{}".format(i), duration=normal(15, 2), output_size=normal(20, 3))
        t.add_input(tasks1[i])
        t.add_input(tasks1[(i + 1) % count])
"""


def gen_graphs(graph_defs, output):
    result = []
    for graph_def in graph_defs:
        fn = graph_def[0]
        args = graph_def[1:]
        result.append([fn.__name__, str(uuid.uuid4()), fn(*args)])
    f = pandas.DataFrame(result, columns=["graph_name", "graph_id", "graph"])
    f.to_pickle(output)


elementary_graphs = [
    (plain1n, 380),
    (plain1e, 380),
    (plain1cpus, 380),

]

gen_graphs(elementary_graphs, "elementary.xz")