from estee.common import TaskOutput
from .utils import exponential, normal
from ..common import TaskGraph


def gridcat(count):
    g = TaskGraph()
    opens = [g.new_task("input{}".format(i),
                        duration=normal(0.01, 0.001),
                        expected_duration=0.01,
                        output_size=normal(300, 25)) for i in range(count)]
    hashes = []
    for i in range(count):
        o1 = opens[i]
        for j in range(i + 1, count):
            o2 = opens[j]
            sz = o1.output.size + o2.output.size
            d = normal(0.2, 0.01) + sz / 1000.0
            cat = g.new_task("cat", duration=d,
                             expected_duration=0.2 + sz / 1000.0, output_size=sz)
            cat.add_input(o1)
            cat.add_input(o2)
            d = normal(0.2, 0.01) + sz / 500.0
            makehash = g.new_task("hash", duration=d,
                                  expected_duration=0.2 + sz / 500.0, output_size=16 / 1024 / 1024)
            makehash.add_input(cat)
            hashes.append(makehash.output)
    m = g.new_task("merge", duration=0.1, expected_duration=0.1, output_size=16 / 1024 / 1024)
    m.add_inputs(hashes)
    return g


def crossv(inner_count, train_cpus=4, eval_cpus=4, factor=1.0):
    g = TaskGraph()

    CHUNK_SIZE = 320
    CHUNK_COUNT = 5

    generator = g.new_task("generator", duration=normal(5, 0.5), expected_duration=5,
                           outputs=[CHUNK_SIZE for _ in range(CHUNK_COUNT)])
    chunks = generator.outputs

    merges = []
    for i in range(CHUNK_COUNT):
        merge = g.new_task("merge{}".format(i), duration=normal(1.1, 0.02), expected_duration=1.1,
                           output_size=CHUNK_SIZE * (CHUNK_COUNT - 1))
        merge.add_inputs([c for j, c in enumerate(chunks) if i != j])
        merges.append(merge)

    for i in range(inner_count):
        results = []
        for i in range(CHUNK_COUNT):
            train = g.new_task("train{}".format(i), duration=exponential(680 * factor),
                               expected_duration=680 * factor, output_size=18, cpus=train_cpus)
            train.add_input(merges[i])
            evaluate = g.new_task("eval{}".format(i), duration=normal(34 * factor, 3),
                                  expected_duration=34 * factor, output_size=0.0001,
                                  cpus=eval_cpus)
            evaluate.add_input(train)
            evaluate.add_input(chunks[i])
            results.append(evaluate.output)

        t = g.new_task("final", duration=0.2, expected_duration=0.2)
        t.add_inputs(results)
    return g


def crossvx(inner_count, outer_count, train_cpus=4, eval_cpus=4):
    graphs = [crossv(inner_count, train_cpus=train_cpus, eval_cpus=eval_cpus)
              for _ in range(outer_count)]
    return TaskGraph.merge(graphs)


def fastcrossv(inner_count):
    return crossv(inner_count, 1, 1, factor=0.02)


def nestedcrossv(parameter_count, factor=1.0, train_cpus=4, eval_cpus=4):
    g = TaskGraph()

    FOLD_SIZE = 320
    FOLD_COUNT = 5
    INNER_FOLD_COUNT = FOLD_COUNT - 1

    assert FOLD_COUNT >= 3

    generator = g.new_task("generator", duration=normal(5, 0.5), expected_duration=5,
                           outputs=[FOLD_SIZE for _ in range(FOLD_COUNT)])
    folds = generator.outputs

    for leave_out_idx in range(FOLD_COUNT):
        inner_folds = folds[:leave_out_idx] + folds[leave_out_idx+1:]
        merges = []
        for i in range(INNER_FOLD_COUNT):
            merge = g.new_task("merge{}".format(i), duration=normal(1.1, 0.02),
                               expected_duration=1.1, output_size=FOLD_SIZE * (INNER_FOLD_COUNT - 1))
            merge.add_inputs([c for j, c in enumerate(inner_folds) if i != j])
            merges.append(merge)

        avg_scores = []
        for p in range(1, parameter_count+1):
            results = []
            for i in range(INNER_FOLD_COUNT):
                train = g.new_task("train{}".format(i), duration=exponential(680 * factor * p),
                                   expected_duration=680 * factor * p, output_size=18,
                                   cpus=train_cpus)
                train.add_input(merges[i])
                evaluate = g.new_task("eval{}".format(i), duration=normal(35 * factor, 3),
                                      expected_duration=35 * factor, output_size=0.0001,
                                      cpus=eval_cpus)
                evaluate.add_input(train)
                evaluate.add_input(inner_folds[i])
                results.append(evaluate.output)

            t = g.new_task("avg_score", duration=0.2, expected_duration=0.2, output_size=0.0001)
            t.add_inputs(results)
            avg_scores.append(t)

        t = g.new_task("best_param", duration=0.2, expected_duration=0.2, output_size=0.0001)
        t.add_inputs(avg_scores)

        merge = g.new_task("merge{}".format(i), duration=normal(1, 0.02),
                           expected_duration=1, output_size=FOLD_SIZE * INNER_FOLD_COUNT)
        merge.add_inputs(inner_folds)

        train = g.new_task("train{}".format(i), duration=exponential(680 * factor),
                           expected_duration=680 * factor, output_size=18, cpus=train_cpus)
        train.add_input(merge)
        train.add_input(t)

        evaluate = g.new_task("eval{}".format(i), duration=normal(38 * factor, 3),
                              expected_duration=38 * factor, output_size=0.0001, cpus=eval_cpus)
        evaluate.add_input(train)
        evaluate.add_input(folds[leave_out_idx])

    return g


def mapreduce(count):
    g = TaskGraph()
    splitter = g.new_task("splitter", duration=10, expected_duration=10,
                          outputs=[2.5 * 1024 for _ in range(count)])
    maps = [g.new_task("map{}".format(i),
                       duration=normal(50, 10),
                       expected_duration=50,
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
