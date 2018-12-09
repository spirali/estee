import argparse
import sys
import uuid

import pandas

from schedsim.generators.elementary import bigmerge, conflux, duration_stairs, fork1, fork2, \
    grid, merge_neighbours, merge_small_big, merge_triplets, plain1cpus, plain1e, plain1n, \
    size_stairs, splitters, triplets, fern
from schedsim.generators.irw import crossv, crossvx, fastcrossv, gridcat, mapreduce, nestedcrossv
from schedsim.generators.pegasus import cybershake, epigenomics, ligo, montage, sipht
from schedsim.generators.randomized import generate_randomized_graph, SGen, MGen
from schedsim.serialization.dask_json import json_serialize

sys.setrecursionlimit(80000)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename")
    parser.add_argument("type", choices=[
        "elementary", "m/elementary",
        "irw", "m/irw",
        "pegasus",
        "rg", "m/rg"])
    return parser.parse_args()


def gen_graphs(graph_defs, output):
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
        g = fn(*args)
        g.normalize()
        g.validate()
        print("{} #t={} #o={}".format(name, g.task_count, len(g.outputs)))
        assert g.task_count < 80000  # safety check
        result.append([name, str(uuid.uuid4()), g])
    print("Saving to ...", output)
    f = pandas.DataFrame(result, columns=["graph_name", "graph_id", "graph"])
    f["graph"] = f["graph"].apply(lambda g: json_serialize(g))
    f.to_pickle(output)


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

m_elementary_generators = [
    (plain1n, 7800),
    (plain1e, 7800),
    (plain1cpus, 7800),
    (triplets, 2510, 16),
    (merge_neighbours, 4040),
    (merge_triplets, 6111),
    (merge_small_big, 2600),
    (fork1, 2700),
    (fork2, 2700),
    (bigmerge, 7200),
    (duration_stairs, 3490),
    (size_stairs, 7990),
    (splitters, 12),
    (conflux, 12),
    (grid, 86),
    (fern, 4000),
]


irw_generators = [
    (gridcat, 20),
    (crossv, 8),
    (crossvx, 4, 4),
    (fastcrossv, 8),
    (mapreduce, 160),
    (nestedcrossv, 5),
]

m_irw_generators = [
    ("m/gridcat", gridcat, 90),
    ("m/crossv", crossv, 800, 16, 16),
    ("m/crossvx", crossvx, 26, 32, 16, 16),
    ("m/fastcrossv", fastcrossv, 800),
    ("m/mapreduce", mapreduce, 260),
    ("m/netstercrossv", nestedcrossv, 200, 1.0, 16, 16),
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
    elif args.type == "m/elementary":
        generators = m_elementary_generators
    elif args.type == "irw":
        generators = irw_generators
    elif args.type == "m/irw":
        generators = m_irw_generators
    elif args.type == "pegasus":
        generators = pegasus_generators
    elif args.type == "rg":
        generators = [("rg", generate_randomized_graph, SGen(), 12) for _ in range(10)]
    elif args.type == "m/rg":
        generators = [("m/rg", generate_randomized_m_graph) for _ in range(3)]
    else:
        assert 0

    gen_graphs(generators, args.filename)


if __name__ == "__main__":
    main()
