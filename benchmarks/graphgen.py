import argparse
import sys
import uuid

import pandas

from schedsim.generators.elementary import bigmerge, conflux, duration_stairs, fork1, fork2, \
    grid, merge_neighbours, merge_small_big, merge_triplets, plain1cpus, plain1e, plain1n, \
    size_stairs, splitters, triplets
from schedsim.generators.irw import crossv, crossv4, fastcrossv, gridcat, mapreduce, nestedcrossv
from schedsim.generators.pegasus import cybershake, epigenomics, ligo, montage, sipht

sys.setrecursionlimit(4500)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename")
    parser.add_argument("type", choices=["elementary", "irw", "pegasus"])
    return parser.parse_args()


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
    (nestedcrossv,),
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
