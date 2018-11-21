import argparse
import multiprocessing
import sys
import os

import numpy as np
import pandas as pd

from schedsim.simulator import Simulator
from schedsim.worker import Worker
from schedsim.communication import MaxMinFlowNetModel
from schedsim.schedulers.basic import AllOnOneScheduler, RandomAssignScheduler
from schedsim.schedulers.queue import BlevelGtScheduler, RandomGtScheduler
from schedsim.schedulers.camp import Camp2Scheduler

from schedsim.schedulers.others import \
    DLSScheduler, K1hScheduler

import collections
import itertools
from tqdm import tqdm

sys.setrecursionlimit(5500)

SCHEDULERS = {
    "single": AllOnOneScheduler,
    "camp2": lambda: Camp2Scheduler(5000),
    "random-s": RandomAssignScheduler,
    "random-gt": RandomGtScheduler,
    "blevel": BlevelGtScheduler,
    #"k1h": K1hScheduler,
    #"dls": DLSScheduler
}


CLUSTERS = {
    "2x8": [{"cpus": 8}] * 2,
    "4x4": [{"cpus": 4}] * 4,
    "stairs16": [{"cpus": i} for i in range(1, 6)] + [{"cpus": 1}]
}

BANDWIDTHS = [
    10240.0,  # 10GB/s
    1024.0,   # 1GB/s
    102.4,  # 0.1GB/s
    10.24,  # 0.01GB/s
]

IMODES = [
    "exact"
]

SCHED_TIMINGS = [
    # min_sched_interval, sched_time
    (0.1, 0.05)
]


Instance = collections.namedtuple("Instance",
    ("graph_name", "graph_id", "graph",
     "cluster_name", "bandwidth",
     "scheduler_name", "min_sched_interval", "sched_time"))


def run_single_instance(instance):
    workers = [Worker(**wargs) for wargs in CLUSTERS[instance.cluster_name]]
    netmodel = MaxMinFlowNetModel(instance.bandwidth)
    scheduler = SCHEDULERS[instance.scheduler_name]()
    simulator = Simulator(instance.graph, workers, scheduler, netmodel)
    return simulator.run()


def benchmark_scheduler(instance, count):
    return [run_single_instance(instance)
            for _ in range(count)]


def process(conf):
    return benchmark_scheduler(*conf)


def instance_iter(graphs, cluster_names, bandwidths, scheduler_names, sched_timings):
    for graph_def, cluster_name, bandwidth, scheduler_name, (min_sched_interval, sched_time) \
            in itertools.product(graphs, cluster_names, bandwidths, scheduler_names, sched_timings):
        g = graph_def[1]
        instance = Instance(
            g["graph_name"], g["graph_id"], g["graph"],
            cluster_name, bandwidth,
            scheduler_name,
            min_sched_interval, sched_time)
        yield instance


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("graphset")
    parser.add_argument("resultfile")
    parser.add_argument("scheduler", choices=["all"] + list(SCHEDULERS.keys()))
    parser.add_argument("repeat", type=int)
    parser.add_argument("imode", choices=IMODES)
    parser.add_argument("--new", action="store_true")
    parser.add_argument("--graph", action="append")
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

    if args.graph:
        graphset = graphset[graphset["graph_name"].isin(args.graph)].reset_index()

    for g in graphset["graph"]:
        g.set_imode_exact()

    if not args.new:
        if not os.path.isfile(args.resultfile):
            print("Result file '{}' does not exists or it is not a file\n"
                "Use --new for create a new one\n".format(args.resultfile),
                file=sys.stderr)
            return
        oldframe = pd.read_pickle(args.resultfile)
        assert list(oldframe.columns) == COLUMNS
    else:
        if os.path.isfile(args.resultfile):
            print("Result file '{}' already exists\n".format(args.resultfile), file=sys.stderr)
            return
        oldframe = pd.DataFrame([], columns=COLUMNS)

    rows = []

    if args.scheduler == "all":
        schedulers = list(SCHEDULERS)
    else:
        schedulers = (args.scheduler,)

    instances = list(instance_iter(
            graphset.iterrows(),
            list(CLUSTERS.keys()),
            BANDWIDTHS,
            schedulers,
            SCHED_TIMINGS))

    print("Testing scheduler: {}".format(args.scheduler))

    pool = multiprocessing.Pool()

    iterator = pool.imap(process, ((i, args.repeat) for i in instances))
    #iterator = (process((x, args.repeat)) for x in instances)

    for instance, result in tqdm(zip(instances, iterator), total=len(instances)):
        for r in result:
            rows.append((
                instance.graph_name,
                instance.graph_id,
                instance.cluster_name,
                instance.bandwidth,
                instance.scheduler_name,
                args.imode,
                instance.min_sched_interval,
                instance.sched_time,
                r
            ))

    frame = pd.DataFrame(rows, columns=COLUMNS)
    print(frame.groupby(["graph_name", "graph_id", "cluster_name", "bandwidth", "scheduler_name"]).mean())
    if not args.new:
        oldframe.to_pickle(args.resultfile + ".backup")

    # Remove old results
    oldframe = oldframe[~oldframe.scheduler_name.isin(schedulers)]

    newframe = pd.concat([oldframe, frame], ignore_index=True)
    newframe.to_pickle(args.resultfile)
    print("{} entries in new {}".format(newframe["time"].count(), args.resultfile))

if __name__ == "__main__":
    main()
