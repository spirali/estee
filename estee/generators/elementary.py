import numpy as np

from estee.common import DataObject
from .utils import exponential, normal
from ..common import TaskGraph


def plain1n(count):
    g = TaskGraph()
    for i in range(count):
        t = int(np.random.choice([10, 20, 60, 180]))
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
        t = int(np.random.choice([10, 20, 60, 180]))
        g.new_task("t{}".format(i), duration=normal(t, t / 10), expected_duration=t,
                   cpus=np.random.randint(1, 4))
    return g


def triplets(count, cpus):
    g = TaskGraph()
    for i in range(count):
        t1 = g.new_task("a{}".format(i), duration=normal(5, 1.5), expected_duration=5,
                        output_size=40)
        t2 = g.new_task("b{}".format(i), duration=normal(120, 20), expected_duration=120,
                        output_size=120, cpus=cpus)
        t2.add_input(t1)
        t3 = g.new_task("c{}".format(i), duration=normal(32, 3), expected_duration=32)
        t3.add_input(t2)
    return g


def merge_neighbours(count):
    g = TaskGraph()

    tasks1 = [g.new_task("a{}".format(i), duration=normal(15, 3),
                         expected_duration=15,
                         outputs=[DataObject(normal(99, 2.5), 100)])
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
                         outputs=[DataObject(normal(99, 2.5), 100)])
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
                         outputs=[DataObject(normal(99, 2.5), 100)])
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
    t = g.new_task("a", duration=0.1,
                   expected_duration=0.1,
                   outputs=list(range(count)))

    for i, o in enumerate(t.outputs):
        t = g.new_task("b{}".format(i), duration=20,
                       expected_duration=20)
        t.add_input(o)
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


def fern(size):
    g = TaskGraph()
    prev = g.new_task("init",
                      output_size=5,
                      expected_duration=20, duration=normal(20, 1))
    for i in range(size):
        a = g.new_task("a{}".format(i),
                       outputs=[DataObject(normal(5 + i / 10, i / 100), 5 + i / 10)],
                       expected_duration=17 + i / 10, duration=normal(17 + i / 10, 3))
        b = g.new_task("b{}".format(i),
                       outputs=[DataObject(normal(42, 2), 42)],
                       expected_duration=35, duration=normal(35, 3))
        a.add_input(prev)
        a.add_input(b)
        prev = a
    return g
