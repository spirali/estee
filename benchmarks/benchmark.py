import argparse
import collections
import itertools
import multiprocessing
import os
import sys

import pandas as pd
from tqdm import tqdm

from schedsim.common.imode import ExactImode, BlindImode
from schedsim.communication import MaxMinFlowNetModel
from schedsim.schedulers.basic import AllOnOneScheduler, RandomAssignScheduler
from schedsim.schedulers.camp import Camp2Scheduler
from schedsim.schedulers.queue import BlevelGtScheduler, RandomGtScheduler, TlevelGtScheduler
from schedsim.simulator import Simulator
from schedsim.worker import Worker

sys.setrecursionlimit(5500)

SCHEDULERS = {
    "single": AllOnOneScheduler,
    "blevel": BlevelGtScheduler,
    "tlevel": TlevelGtScheduler,
    "random-s": RandomAssignScheduler,
    "random-gt": RandomGtScheduler,
    #"dls": DLSScheduler,
    #"etf": ETFScheduler,
    #"last": LASTScheduler,
    #"mcp": MCPScheduler,
    "camp2": lambda: Camp2Scheduler(5000)
}


CLUSTERS = {
    "2x8": [{"cpus": 8}] * 2,
    "4x4": [{"cpus": 4}] * 4,
    "16x4": [{"cpus": 4}] * 16,
    "stairs16": [{"cpus": i} for i in range(1, 6)] + [{"cpus": 1}],
    "32x16": [{"cpus": 16}] * 32,
    "64x16": [{"cpus": 16}] * 64,
    "128x16": [{"cpus": 16}] * 128,
    "256x16": [{"cpus": 16}] * 256,
}


BANDWIDTHS = {
    "8G":   8192,
    "2G":   2048,
    "512M": 512,
    "128M": 128,
    "32M":  32
}


IMODES = {
    "exact": lambda g: ExactImode().process(g),
    "blind": lambda g: BlindImode().process(g)
}


SCHED_TIMINGS = [
    # min_sched_interval, sched_time
    (0.1, 0.05)
]


Instance = collections.namedtuple("Instance",
    ("graph_name", "graph_id", "graph",
     "cluster_name", "bandwidth",
     "scheduler_name", "imode", "min_sched_interval", "sched_time"))


def run_single_instance(instance):
    workers = [Worker(**wargs) for wargs in CLUSTERS[instance.cluster_name]]
    netmodel = MaxMinFlowNetModel(BANDWIDTHS[instance.bandwidth])
    scheduler = SCHEDULERS[instance.scheduler_name]()
    simulator = Simulator(instance.graph, workers, scheduler, netmodel)
    return simulator.run()


def benchmark_scheduler(instance, count):
    return [run_single_instance(instance)
            for _ in range(count)]


def process(conf):
    return benchmark_scheduler(*conf)


def instance_iter(graphs, cluster_names, bandwidths, scheduler_names, imodes, sched_timings):
    graph_cache = {}

    def calculate_imodes(graph):
        if graph not in graph_cache:
            graph_cache[graph] = {}
            for imode in imodes:
                g = graph.copy() if len(imodes) > 1 else graph
                graph_cache[graph][imode] = g
                IMODES[imode](g)

    for graph_def, cluster_name, bandwidth, scheduler_name, imode, (min_sched_interval, sched_time) \
            in itertools.product(graphs, cluster_names, bandwidths, scheduler_names, imodes, sched_timings):
        g = graph_def[1]
        calculate_imodes(g["graph"])
        graph = graph_cache[g["graph"]][imode]
        instance = Instance(
            g["graph_name"], g["graph_id"], graph,
            cluster_name, bandwidth,
            scheduler_name,
            imode,
            min_sched_interval, sched_time)
        yield instance


def parse_args():
    def generate_help(keys):
        return "all,{}".format(",".join(keys))

    parser = argparse.ArgumentParser()
    parser.add_argument("graphset")
    parser.add_argument("resultfile")
    parser.add_argument("--scheduler", help=generate_help(list(SCHEDULERS)), default="all")
    parser.add_argument("--cluster", help=generate_help(list(CLUSTERS)), default="all")
    parser.add_argument("--bandwidth", help=generate_help(list(BANDWIDTHS)), default="all")
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--imode", help=generate_help(list(IMODES)), default="all")
    parser.add_argument("--no-append", action="store_true")
    parser.add_argument("--graphs")
    return parser.parse_args()


def main():
    COLUMNS = ["graph_name",
               "graph_id",
               "cluster_name",
               "bandwidth",
               "scheduler_name",
               "imode",
               "min_sched_interval",
               "sched_time",
               "time"]

    args = parse_args()
    graphset = pd.read_pickle(args.graphset)

    if args.graphs:
        graphset = graphset[graphset["graph_name"].isin(args.graphs.split(","))].reset_index()

    if len(graphset) == 0:
        print("No graphs selected")
        return

    appending = False
    if os.path.isfile(args.resultfile):
        if args.no_append:
            print("Result file '{}' already exists\n"
                  "Remove --no-append to append results to it".format(args.resultfile),
                  file=sys.stderr)
            exit(1)

        appending = True
        print("Appending to result file '{}'".format(args.resultfile))

        oldframe = pd.read_pickle(args.resultfile)
        assert list(oldframe.columns) == COLUMNS
    else:
        print("Creating result file '{}'".format(args.resultfile))
        oldframe = pd.DataFrame([], columns=COLUMNS)

    def select_option(value, keys):
        if value == "all":
            return list(keys)
        value = [v.strip() for v in value.split(",")]
        assert all(v in keys for v in value)

        return value

    schedulers = select_option(args.scheduler, SCHEDULERS)
    clusters = select_option(args.cluster, CLUSTERS)
    bandwidths = select_option(args.bandwidth, BANDWIDTHS)
    imodes = select_option(args.imode, IMODES)

    instances = list(instance_iter(
        graphset.iterrows(),
        clusters,
        bandwidths,
        schedulers,
        imodes,
        SCHED_TIMINGS))

    print("Testing scheduler: {}".format(args.scheduler))

    pool = multiprocessing.Pool()
    iterator = pool.imap(process, ((i, args.repeat) for i in instances))

    rows = []
    counter = 0
    try:
        for instance, result in tqdm(zip(instances, iterator), total=len(instances)):
            counter += 1
            for r in result:
                rows.append((
                    instance.graph_name,
                    instance.graph_id,
                    instance.cluster_name,
                    instance.bandwidth,
                    instance.scheduler_name,
                    instance.imode,
                    instance.min_sched_interval,
                    instance.sched_time,
                    r
                ))
    except KeyboardInterrupt:
        print("Benchmark interrupted, iterated {} instances. Writing intermediate results"
              .format(counter))

    frame = pd.DataFrame(rows, columns=COLUMNS)
    print(frame.groupby(["graph_name", "graph_id", "cluster_name",
                         "bandwidth", "scheduler_name"]).mean())

    if appending:
        base, ext = os.path.splitext(args.resultfile)
        path = "{}.backup{}".format(base, ext)
        print("Creating backup of old results to '{}'".format(path))
        oldframe.to_pickle(path)

    # Remove old results
    oldframe = oldframe[~oldframe.scheduler_name.isin(schedulers)]

    newframe = pd.concat([oldframe, frame], ignore_index=True)
    newframe.to_pickle(args.resultfile)
    print("{} entries in new '{}'".format(newframe["time"].count(), args.resultfile))


if __name__ == "__main__":
    main()
