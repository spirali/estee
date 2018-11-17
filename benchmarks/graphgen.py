
from schedsim.common import TaskGraph, TaskOutput

from clusters import clusters

import itertools
import pandas
import numpy as np
import uuid
import sys

sys.setrecursionlimit(4500)

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


def triplets(count):
    g = TaskGraph()
    for i in range(count):
        t1 = g.new_task("a{}".format(i), duration=normal(5, 1.5), expected_duration=5, output_size=40)
        t2 = g.new_task("b{}".format(i), duration=normal(118, 20), expected_duration=120, output_size=120, cpus=4)
        t2.add_input(t1)
        t3 = g.new_task("c{}".format(i), duration=normal(32, 3), expected_duration=30)
        t3.add_input(t2)
    return g


def merge_neighbours(count):
    g = TaskGraph()

    tasks1 = [g.new_task("a{}".format(i), duration=normal(15, 3),
                                         expected_duration=15,
                                         outputs=[TaskOutput(normal(99, 2.5), 100)])
              for i in range(count)]
    for i in range(count):
        t = g.new_task("b{}".format(i), duration=normal(15, 2), expected_duration=15)
        t.add_input(tasks1[i])
        t.add_input(tasks1[(i + 1) % count])
    return g

def merge_triplets(count):
    g = TaskGraph()

    tasks1 = [g.new_task("a{}".format(i), duration=normal(15, 3),
                                         expected_duration=15,
                                         outputs=[TaskOutput(normal(99, 2.5), 100)])
              for i in range(count)]
    for i in range(0, count, 3):
        t = g.new_task("b{}".format(i), duration=normal(15, 2), expected_duration=15)
        t.add_input(tasks1[i])
        t.add_input(tasks1[i + 1])
        t.add_input(tasks1[i + 2])
    return g

def merge_small_big(count):
    g = TaskGraph()
    tasks1 = [g.new_task("a{}".format(i), duration=normal(11, 3),
                                         expected_duration=11,
                                         output_size=0.5)
              for i in range(count)]

    tasks2 = [g.new_task("b{}".format(i), duration=normal(15, 3),
                                          expected_duration=15,
                                          outputs=[TaskOutput(normal(99, 2.5), 100)])
              for i in range(count)]

    for i, (t1, t2) in enumerate(zip(tasks1, tasks2)):
        t = g.new_task("b{}".format(i), duration=normal(10, 1), expected_duration=10)
        t.add_input(t1)
        t.add_input(t2)
    return g

def fork1(count):
    g = TaskGraph()
    tasks1 = [g.new_task("a{}".format(i), duration=normal(17, 3),
                                         expected_duration=17,
                                         output_size=100)
              for i in range(count)]

    for i in range(count):
        t = g.new_task("b{}".format(i), duration=normal(15, 2), expected_duration=15)
        t.add_input(tasks1[i])
        t = g.new_task("c{}".format(i), duration=normal(15, 2), expected_duration=15)
        t.add_input(tasks1[i])
    return g

def fork2(count):
    g = TaskGraph()
    tasks1 = [g.new_task("a{}".format(i), duration=normal(17, 3),
                                         expected_duration=17,
                                         outputs=[100, 100])
              for i in range(count)]

    for i in range(count):
        t = g.new_task("b{}".format(i), duration=normal(15, 2), expected_duration=15)
        t.add_input(tasks1[i].outputs[0])
        t = g.new_task("c{}".format(i), duration=normal(15, 2), expected_duration=15)
        t.add_input(tasks1[i].outputs[1])
    return g

def bigmerge(count):
    g = TaskGraph()
    tasks1 = [g.new_task("a{}".format(i), duration=normal(17, 3),
                                         expected_duration=17,
                                         outputs=[100])
              for i in range(count)]

    t = g.new_task("m1", duration=40, expected_duration=40)
    t.add_inputs(tasks1)

    g.write_dot("/tmp/g.dot")

    return g


def duration_stairs(count):
    g = TaskGraph()
    for i in range(count):
        g.new_task("a{}".format(i), duration=i,
                                    expected_duration=i)
        g.new_task("b{}".format(i), duration=i,
                                    expected_duration=i)
    return g


def size_stairs(count):
    g = TaskGraph()
    tasks = []

    t = g.new_task("a", duration=0.1,
                        expected_duration=0.1,
                        outputs=list(range(count)))

    for i, o in enumerate(t.outputs):
        t = g.new_task("b{}".format(i), duration=20,
                                        expected_duration=20)
        t.add_input(o)
    g.write_dot("/tmp/g.dot")
    return g


def splitters(depth):
    g = TaskGraph()
    tasks = [g.new_task("root", duration=1, expected_duration=1, output_size=512)]
    for i in range(depth):
        new = [g.new_task("a{}-{}".format(i, j), duration=normal(20, 1),
               expected_duration=20,
               output_size=128)
               for j in range(len(tasks) * 2)]
        for j, t in enumerate(new):
            t.add_input(tasks[j // 2])
        tasks = new
    return g


def conflux(depth):
    g = TaskGraph()
    tasks = [g.new_task("top{}".format(j), duration=normal(20, 1.5),
             expected_duration=20,
             output_size=128)
             for j in range(2 ** depth)]
    for i in range(depth):
        new = [g.new_task("a{}-{}".format(i, j), duration=normal(20, 1),
               expected_duration=20,
               output_size=128)
               for j in range(len(tasks) // 2)]
        for j, t in enumerate(new):
            t.add_input(tasks[j * 2])
            t.add_input(tasks[j * 2 + 1])
        tasks = new
    return g


def grid(size):
    g = TaskGraph()
    tasks = [g.new_task("a".format(j), duration=normal(20, 1),
             expected_duration=20,
             output_size=128)
             for j in range(size)]
    prev = tasks[0]
    for t in tasks[1:]:
        t.add_input(prev)
        prev = t


    for i in range(size - 1):
        new = [g.new_task("a{}-{}".format(i, j), duration=normal(20, 1),
               expected_duration=20,
               output_size=128)
               for j in range(size)]
        for t1, t2 in zip(tasks, new):
            t2.add_input(t1)
        prev = new[0]
        for t in new[1:]:
            t.add_input(prev)
            prev = t
        tasks = new
    return g


def gen_graphs(graph_defs, output):
    result = []
    for graph_def in graph_defs:
        fn = graph_def[0]
        args = graph_def[1:]
        g = fn(*args)
        assert isinstance(g, TaskGraph)
        assert len(g.tasks) < 800  # safety check
        result.append([fn.__name__, str(uuid.uuid4()), g])
    f = pandas.DataFrame(result, columns=["graph_name", "graph_id", "graph"])
    f.to_pickle(output)


elementary_graphs = [
    (plain1n, 380),
    (plain1e, 380),
    (plain1cpus, 380),
    (triplets, 110),
    (merge_neighbours, 107),
    (merge_triplets, 111),
    (merge_small_big, 80),
    (fork1, 100),
    (fork2, 100),
    (bigmerge, 320),
    (duration_stairs, 190),
    (size_stairs, 190),
    (splitters, 7),
    (conflux, 7),
    (grid, 19),
]

gen_graphs(elementary_graphs, "elementary.xz")