import argparse
import sys
import uuid

import numpy as np
import pandas

from schedsim.common import TaskGraph, TaskOutput
from schedsim.generators.pegasus import cybershake, epigenomics, ligo, montage, sipht

sys.setrecursionlimit(4500)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename")
    parser.add_argument("type", choices=["elementary", "irw", "pegasus"])
    return parser.parse_args()


def normal(loc, scale):
    return max(0.0000001, np.random.normal(loc, scale))


def exponential(scale):
    return max(0.0000001, np.random.exponential(scale))


# Elementary generators

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
        t1 = g.new_task("a{}".format(i), duration=normal(5, 1.5), expected_duration=5,
                        output_size=40)
        t2 = g.new_task("b{}".format(i), duration=normal(118, 20), expected_duration=120,
                        output_size=120, cpus=4)
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


# IRW (Inspired by Real World)

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


def crossv(inner_count, factor=1.0):
    g = TaskGraph()

    CHUNK_SIZE = 320
    CHUNK_COUNT = 5

    generator = g.new_task("generator", duration=normal(5, 0.5), expected_duration=5,
                           outputs=[CHUNK_SIZE for _ in range(CHUNK_COUNT)])
    chunks = generator.outputs

    merges = []
    for i in range(CHUNK_COUNT):
        merge = g.new_task("merge{}".format(i), duration=normal(1.1, 0.02), expected_duration=1,
                           output_size=CHUNK_SIZE * (CHUNK_COUNT - 1))
        merge.add_inputs([c for j, c in enumerate(chunks) if i != j])
        merges.append(merge)

    for i in range(inner_count):
        results = []
        for i in range(CHUNK_COUNT):
            train = g.new_task("train{}".format(i), duration=exponential(680 * factor),
                               expected_duration=660 * factor, output_size=18, cpus=4)
            train.add_input(merges[i])
            evaluate = g.new_task("eval{}".format(i), duration=normal(34 * factor, 3),
                                  expected_duration=30 * factor, output_size=0.0001, cpus=4)
            evaluate.add_input(train)
            evaluate.add_input(chunks[i])
            results.append(evaluate.output)

        t = g.new_task("final", duration=0.2, expected_duration=0.2)
        t.add_inputs(results)
    return g


def crossv4(inner_count):
    graphs = [crossv(inner_count) for _ in range(4)]
    return TaskGraph.merge(graphs)


def fastcrossv(inner_count):
    return crossv(inner_count, 0.02)


def mapreduce(count):
    g = TaskGraph()
    splitter = g.new_task("splitter", duration=10, expected_duration=10,
                          outputs=[2.5 * 1024 for _ in range(count)])
    maps = [g.new_task("map{}".format(i),
                       duration=normal(49, 10),
                       expected_duration=60,
                       outputs=[TaskOutput(size=normal(250 / count, 20 / count),
                                           expected_size=250 / count)
                                for _ in range(count)])
            for i in range(count)]
    for t, o in zip(maps, splitter.outputs):
        t.add_input(o)

    for i in range(count):
        outputs = [m.outputs[i] for m in maps]
        t = g.new_task("reduce{}".format(i), duration=normal(sum(o.size / 25 for o in outputs), 5),
                       expected_duration=10)
        t.add_inputs(outputs)

    return g


# Utils

def gen_graphs(graph_defs, output):
    result = []
    for graph_def in graph_defs:
        fn = graph_def[0]
        name = fn.__name__
        args = graph_def[1:]
        g = fn(*args)
        g.validate()
        print(name, g.task_count)
        assert g.task_count < 800  # safety check
        result.append([name, str(uuid.uuid4()), g])
    f = pandas.DataFrame(result, columns=["graph_name", "graph_id", "graph"])
    f.to_pickle(output)


elementary_generators = [
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

irw_generators = [
    (gridcat, 20),
    (crossv, 8),
    (crossv4, 4),
    (fastcrossv, 8),
    (mapreduce, 160),
]

pegasus_generators = [
    (montage, 50),
    (cybershake, 50),
    (epigenomics, 50),
    (ligo, (20, 10, 15)),
    (sipht, 2)
]


def main():
    args = parse_args()

    if args.type == "elementary":
        generators = elementary_generators
    elif args.type == "irw":
        generators = irw_generators
    elif args.type == "pegasus":
        generators = pegasus_generators
    else:
        assert 0

    gen_graphs(generators, args.filename)


if __name__ == "__main__":
    main()
