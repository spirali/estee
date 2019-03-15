import argparse
import sys
import uuid

import pandas

from estee.generators.elementary import bigmerge, conflux, duration_stairs, fork1, fork2, \
    grid, merge_neighbours, merge_small_big, merge_triplets, plain1cpus, plain1e, plain1n, \
    size_stairs, splitters, triplets, fern
from estee.generators.irw import crossv, crossvx, fastcrossv, gridcat, mapreduce, nestedcrossv
from estee.generators.pegasus import cybershake, epigenomics, ligo, montage, sipht
from estee.generators.randomized import generate_randomized_graph, SGen, MGen
from estee.serialization.dask_json import json_serialize

sys.setrecursionlimit(80000)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename")
    parser.add_argument("type", choices=[
        "elementary", "elementary2",
        "irw", "irw2",
        "pegasus",
        "rg", "m/rg"])
    return parser.parse_args()


def gen_graphs(graph_defs, output, prefix):
    result = []
    for graph_def in graph_defs:
        first = graph_def[0]
        if callable(first):
            fn = first
            name = fn.__name__
            args = graph_def[1:]
        else:
            name = first
            fn = graph_def[1]
            args = graph_def[2:]
        name = prefix + name
        g = fn(*args)
        g.normalize()
        g.validate()

        for t in g.tasks.values():
            assert t.expected_duration is not None
        for o in g.objects.values():
            if o.expected_size is None:
                o.expected_size = o.size

        print("{} #t={} #o={}".format(name, g.task_count, len(g.objects)))
        assert g.task_count < 80000  # safety check
        result.append([name, str(uuid.uuid4()), g])
    print("Saving to ...", output)
    f = pandas.DataFrame(result, columns=["graph_name", "graph_id", "graph"])
    f["graph"] = f["graph"].apply(lambda g: json_serialize(g))
    f.to_json(output)


elementary_generators = [
    (plain1n, 380),
    (plain1e, 380),
    (plain1cpus, 380),
    (triplets, 110, 4),
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
    (fern, 200),
]

elementary2_generators = [
    (plain1n, 4100),
    (plain1e, 4100),
    (plain1cpus, 3800),
    (triplets, 1510, 4),
    (merge_neighbours, 2040),
    (merge_triplets, 3111),
    (merge_small_big, 1412),
    (fork1, 1470),
    (fork2, 1470),
    (bigmerge, 4300),
    (duration_stairs, 2190),
    (size_stairs, 4290),
    (splitters, 11),
    (conflux, 11),
    (grid, 66),
    (fern, 2200),
]


irw_generators = [
    (gridcat, 20),
    (crossv, 8),
    (crossvx, 4, 4),
    (fastcrossv, 8),
    (mapreduce, 160),
    (nestedcrossv, 5),
]

irw2_generators = [
    ("gridcat", gridcat, 67),
    ("crossv", crossv, 410, 4, 4),
    ("crossvx", crossvx, 16, 23, 4, 4),
    ("fastcrossv", fastcrossv, 410),
    ("mapreduce", mapreduce, 145),
    ("netstercrossv", nestedcrossv, 100, 1.0, 4, 4),
]

pegasus_generators = [
    (montage, 50),
    (cybershake, 50),
    (epigenomics, 50),
    (ligo, (20, 10, 15)),
    (sipht, 2)
]


def generate_randomized_m_graph():
    while True:
        g = generate_randomized_graph(MGen(), 27)
        if g.task_count > 3600:
            return g


def main():
    args = parse_args()

    if args.type == "elementary":
        generators = elementary_generators
    elif args.type == "elementary2":
        generators = elementary2_generators
    elif args.type == "irw":
        generators = irw_generators
    elif args.type == "irw2":
        generators = irw2_generators
    elif args.type == "pegasus":
        generators = pegasus_generators
    elif args.type == "rg":
        generators = [("rg", generate_randomized_graph, SGen(), 12) for _ in range(60)]
    elif args.type == "m/rg":
        generators = [("m/rg", generate_randomized_m_graph) for _ in range(3)]
    else:
        assert 0

    gen_graphs(generators, args.filename, "{}-".format(args.type))


if __name__ == "__main__":
    main()
