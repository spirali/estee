import argparse
import collections
import itertools
import multiprocessing
import os
import re
import sys
import time

import pandas as pd
from tqdm import tqdm

from schedsim.common import imode
from schedsim.communication import MinMaxFlowNetModel, SimpleNetModel
from schedsim.schedulers.basic import AllOnOneScheduler, RandomAssignScheduler
from schedsim.schedulers.camp import Camp2Scheduler
from schedsim.schedulers.genetic import GeneticScheduler
from schedsim.schedulers.others import DLSScheduler, ETFScheduler, MCPScheduler
from schedsim.schedulers.queue import BlevelGtScheduler, RandomGtScheduler, TlevelGtScheduler
from schedsim.serialization.utils import set_recursion_limit
from schedsim.simulator import Simulator
from schedsim.worker import Worker

set_recursion_limit()

SCHEDULERS = {
    "single": AllOnOneScheduler,
    "blevel": BlevelGtScheduler,
    "tlevel": TlevelGtScheduler,
    "random-s": RandomAssignScheduler,
    "random-gt": RandomGtScheduler,
    "dls": DLSScheduler,
    "etf": ETFScheduler,
    "mcp": MCPScheduler,
    "genetic": GeneticScheduler,
    #"last": LASTScheduler,
    "camp2": lambda: Camp2Scheduler(5000)
}


NETMODELS = {
    "simple": SimpleNetModel,
    "minmax": MinMaxFlowNetModel
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
    "exact": imode.process_imode_exact,
    "blind": imode.process_imode_blind,
    "mean": imode.process_imode_mean,
    "user": imode.process_imode_user,
}


SCHED_TIMINGS = {
    # min_sched_interval, sched_time
    "0.1/0.05": (0.1, 0.05),
    "0.4/0.05": (0.4, 0.05),
    "1.6/0.05": (1.6, 0.05),
    "6.4/0.05": (6.4, 0.05)
}


Instance = collections.namedtuple("Instance",
                                  ("graph_name", "graph_id", "graph",
                                   "cluster_name", "bandwidth", "netmodel",
                                   "scheduler_name", "imode", "min_sched_interval", "sched_time",
                                   "count"))


def run_single_instance(instance):
    begin_time = time.monotonic()
    workers = [Worker(**wargs) for wargs in CLUSTERS[instance.cluster_name]]
    netmodel = NETMODELS[instance.netmodel](instance.bandwidth)
    scheduler = SCHEDULERS[instance.scheduler_name]()
    simulator = Simulator(instance.graph, workers, scheduler, netmodel)
    return simulator.run(), time.monotonic() - begin_time


def benchmark_scheduler(instance):
    return [run_single_instance(instance)
            for _ in range(instance.count)]


def instance_iter(graphs, cluster_names, bandwidths, netmodels, scheduler_names, imodes,
                  sched_timings, count):
    graph_cache = {}

    def calculate_imodes(graph):
        if graph not in graph_cache:
            graph_cache[graph] = {}
            for imode in imodes:
                g = graph.copy() if len(imodes) > 1 else graph
                graph_cache[graph][imode] = g
                IMODES[imode](g)

    for graph_def, cluster_name, bandwidth, netmodel, scheduler_name, imode, sched_timing\
        in itertools.product(graphs, cluster_names, bandwidths, netmodels, scheduler_names, imodes,
                             sched_timings):
        g = graph_def[1]
        calculate_imodes(g["graph"])
        graph = graph_cache[g["graph"]][imode]

        (min_sched_interval, sched_time) = SCHED_TIMINGS[sched_timing]
        instance = Instance(
            g["graph_name"], g["graph_id"], graph,
            cluster_name, BANDWIDTHS[bandwidth], netmodel,
            scheduler_name,
            imode,
            min_sched_interval, sched_time,
            count)
        yield instance


def run_multiprocessing(pool, instances):
    return pool.imap(benchmark_scheduler, instances)


def dask_serialize(data):
    return data


def dask_deserialize(data):
    return data


def process_dask(conf):
    (graph, instance) = conf
    instance = instance._replace(graph=dask_deserialize(graph))
    return benchmark_scheduler(instance)


def run_dask(instances, cluster):
    from dask.distributed import Client

    client = Client(cluster)
    client.run(set_recursion_limit)

    graphs = {}
    instance_to_graph = {}
    instances = list(instances)
    for (i, instance) in enumerate(instances):
        if instance.graph not in graphs:
            graphs[instance.graph] = client.scatter([dask_serialize(instance.graph)])[0]
        inst = instance._replace(graph=None)
        instance_to_graph[inst] = graphs[instance.graph]
        instances[i] = inst

    results = client.map(process_dask, ((instance_to_graph[i], i) for i in instances))
    return client.gather(results)


def parse_args():
    def generate_help(keys):
        return "all,{}".format(",".join(keys))

    parser = argparse.ArgumentParser()
    parser.add_argument("graphset")
    parser.add_argument("resultfile")
    parser.add_argument("--scheduler", help=generate_help(list(SCHEDULERS)), default="all")
    parser.add_argument("--cluster", help=generate_help(list(CLUSTERS)), default="all")
    parser.add_argument("--bandwidth", help=generate_help(list(BANDWIDTHS)), default="all")
    parser.add_argument("--netmodel", help=generate_help(list(NETMODELS)), default="minmax")
    parser.add_argument("--imode", help=generate_help(list(IMODES)), default="user")
    parser.add_argument("--sched-timing", help=generate_help(list(SCHED_TIMINGS)), default="all")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--no-append", action="store_true",
                        help="Exit if the resultfile already exists.")
    parser.add_argument("--skip-completed", action="store_true",
                        help="Skip already computed instances found in the resultfile.")
    parser.add_argument("--graphs", help="Comma separated list of graphs to be used from the "
                                         "input graphset")
    parser.add_argument("--timeout", help="Timeout for the computation. Format hh:mm:ss.")
    parser.add_argument("--dask-cluster", help="Address of Dask scheduler")
    return parser.parse_args()


def parse_timeout(timeout):
    if not timeout:
        return 0
    match = re.match("^(\d{2}):(\d{2}):(\d{2})$", timeout)
    if not match:
        print("Wrong timeout format. Enter timeout as hh:mm:ss.")
        exit(1)
    return int(match.group(1)) * 3600 + int(match.group(2)) * 60 + int(match.group(3))


def skip_completed(instances, frame, args):
    result = []
    columns = ["graph_name",
               "graph_id",
               "cluster_name",
               "bandwidth",
               "netmodel",
               "scheduler_name",
               "imode",
               "min_sched_interval",
               "sched_time"]

    skipped = 0
    for instance in instances:
        completed = frame
        for col in columns:
            completed = completed[completed[col] == getattr(instance, col)]
        if len(completed) < args.repeat:
            result.append(instance._replace(count=args.repeat - len(completed)))
        skipped += len(completed)

    if skipped:
        print("Skipping {} instances".format(skipped))
    return result


def main():
    COLUMNS = ["graph_name",
               "graph_id",
               "cluster_name",
               "bandwidth",
               "netmodel",
               "scheduler_name",
               "imode",
               "min_sched_interval",
               "sched_time",
               "time",
               "execution_time"]

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
    netmodels = select_option(args.netmodel, NETMODELS)
    imodes = select_option(args.imode, IMODES)
    sched_timings = select_option(args.sched_timing, SCHED_TIMINGS)

    instances = list(instance_iter(
        graphset.iterrows(),
        clusters,
        bandwidths,
        netmodels,
        schedulers,
        imodes,
        sched_timings,
        args.repeat))

    if appending and args.skip_completed:
        instances = skip_completed(instances, oldframe, args)
        if not instances:
            print("All instances were already computed")
            return

    print("============ Config ========================")
    print("scheduler : {}".format(", ".join(schedulers)))
    print("cluster   : {}".format(", ".join(clusters)))
    print("netmodel  : {}".format(", ".join(netmodels)))
    print("bandwidths: {}".format(", ".join(bandwidths)))
    print("imode     : {}".format(", ".join(imodes)))
    print("timings   : {}".format(", ".join(sched_timings)))
    print("REPEAT    : {}".format(args.repeat))
    print("============================================")

    if args.dask_cluster:
        iterator = run_dask(instances, args.dask_cluster)
    else:
        pool = multiprocessing.Pool()
        iterator = run_multiprocessing(pool, instances)

    rows = []
    counter = 0
    timeout = parse_timeout(args.timeout)
    start = time.time()

    if timeout:
        print("Timeout set to {} seconds".format(timeout))

    try:
        for instance, result in tqdm(zip(instances, iterator), total=len(instances)):
            counter += 1
            for r_time, r_runtime in result:
                rows.append((
                    instance.graph_name,
                    instance.graph_id,
                    instance.cluster_name,
                    instance.bandwidth,
                    instance.netmodel,
                    instance.scheduler_name,
                    instance.imode,
                    instance.min_sched_interval,
                    instance.sched_time,
                    r_time,
                    r_runtime,
                ))
            if timeout and time.time() - start > timeout:
                print("Timeout reached, iterated {} instances. Writing intermediate results"
                      .format(counter))
                break

    except KeyboardInterrupt:
        print("Benchmark interrupted, iterated {} instances. Writing intermediate results"
              .format(counter))

    frame = pd.DataFrame(rows, columns=COLUMNS)
    print(frame.groupby(["graph_name", "graph_id", "cluster_name",
                         "bandwidth", "netmodel", "imode", "min_sched_interval", "sched_time",
                         "scheduler_name"]).mean())

    if appending:
        base, ext = os.path.splitext(args.resultfile)
        path = "{}.backup{}".format(base, ext)
        print("Creating backup of old results to '{}'".format(path))
        oldframe.to_pickle(path)

    # Remove old results
    if not args.skip_completed:
        oldframe = oldframe[~oldframe.scheduler_name.isin(schedulers)]

    newframe = pd.concat([oldframe, frame], ignore_index=True)
    newframe.to_pickle(args.resultfile)
    print("{} entries in new '{}'".format(newframe["time"].count(), args.resultfile))


if __name__ == "__main__":
    main()
