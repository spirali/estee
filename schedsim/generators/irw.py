from schedsim.common import TaskOutput
from .utils import exponential, normal
from ..common import TaskGraph


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
