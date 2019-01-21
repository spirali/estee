import argparse
import collections
import itertools
import multiprocessing
import os
import random
import re
import sys
import threading
import time

import numpy
import pandas as pd
from tqdm import tqdm

from schedsim.common import imode
from schedsim.communication import MinMaxFlowNetModel, SimpleNetModel
from schedsim.schedulers.basic import AllOnOneScheduler, RandomAssignScheduler
from schedsim.schedulers.camp import Camp2Scheduler
from schedsim.schedulers.clustering import LcScheduler
from schedsim.schedulers.genetic import GeneticScheduler
from schedsim.schedulers.others import BlevelScheduler, DLSScheduler, ETFScheduler, MCPScheduler, \
    TlevelScheduler
from schedsim.schedulers.queue import BlevelGtScheduler, RandomGtScheduler, TlevelGtScheduler
from schedsim.serialization.dask_json import json_deserialize, json_serialize
from schedsim.simulator import Simulator
from schedsim.worker import Worker


def generate_seed():
    seed = os.getpid() * time.time()
    for b in os.urandom(4):
        seed *= b
    seed = int(seed) % 2**32
    random.seed(seed)
    numpy.random.seed(seed)


generate_seed()


SCHEDULERS = {
    "single": AllOnOneScheduler,
    "blevel": BlevelGtScheduler,
    "blevel-simple": BlevelScheduler,
    "tlevel": TlevelGtScheduler,
    "tlevel-simple": TlevelScheduler,
    "random-s": RandomAssignScheduler,
    "random-gt": RandomGtScheduler,
    "dls": DLSScheduler,
    "etf": ETFScheduler,
    "mcp": MCPScheduler,
    "genetic": GeneticScheduler,
    #"last": LASTScheduler,
    "camp2": lambda: Camp2Scheduler(5000),
    "lc": LcScheduler
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
    "8G": 8192,
    "2G": 2048,
    "512M": 512,
    "128M": 128,
    "32M": 32
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
                                  ("graph_set", "graph_name", "graph_id", "graph",
                                   "cluster_name", "bandwidth", "netmodel",
                                   "scheduler_name", "imode", "min_sched_interval", "sched_time",
                                   "count"))


def run_single_instance(instance):
    time.sleep(1)
    inf = 2**32

    def create_worker(wargs):
        if instance.netmodel == "simple":
            return Worker(**wargs, max_downloads=inf, max_downloads_per_worker=inf)
        return Worker(**wargs)

    begin_time = time.monotonic()
    workers = [create_worker(wargs) for wargs in CLUSTERS[instance.cluster_name]]
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

    def calculate_imodes(graph, graph_id):
        if graph_id not in graph_cache:
            graph_cache[graph_id] = {}
            for mode in imodes:
                g = json_deserialize(graph)
                IMODES[mode](g)
                graph_cache[graph_id][mode] = json_serialize(g)

    for graph_def, cluster_name, bandwidth, netmodel, scheduler_name, mode, sched_timing \
            in itertools.product(graphs, cluster_names, bandwidths, netmodels, scheduler_names,
                                 imodes,
                                 sched_timings):
        g = graph_def[1]
        calculate_imodes(g["graph"], g["graph_id"])
        graph = graph_cache[g["graph_id"]][mode]

        (min_sched_interval, sched_time) = SCHED_TIMINGS[sched_timing]
        instance = Instance(
            g["graph_set"], g["graph_name"], g["graph_id"], graph,
            cluster_name, BANDWIDTHS[bandwidth], netmodel,
            scheduler_name,
            mode,
            min_sched_interval, sched_time,
            count)
        yield instance


def process_multiprocessing(instance):
    instance = instance._replace(graph=json_deserialize(instance.graph))
    return benchmark_scheduler(instance)


def run_multiprocessing(pool, instances):
    return pool.imap(process_multiprocessing, instances)


def process_dask(conf):
    (graph, instance) = conf
    instance = instance._replace(graph=json_deserialize(graph))
    return benchmark_scheduler(instance)


def run_dask(instances, cluster):
    from dask.distributed import Client

    client = Client(cluster)

    graphs = {}
    instance_to_graph = {}
    instances = list(instances)
    for (i, instance) in enumerate(instances):
        if instance.graph not in graphs:
            graphs[instance.graph] = client.scatter([instance.graph], broadcast=True)[0]
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
    parser.add_argument("--interval", help="From:to indices to compute.")
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


def skip_completed_instances(instances, frame, repeat):
    columns = ["graph_id",
               "cluster_name",
               "bandwidth",
               "netmodel",
               "scheduler_name",
               "imode",
               "min_sched_interval",
               "sched_time"]

    skipped = 0
    counts = frame.groupby(columns).size()

    result = []
    for instance in instances:
        hashed = tuple(getattr(instance, col) for col in columns)
        if hashed in counts:
            count = counts.loc[hashed]
            if count < repeat:
                result.append(instance._replace(count=repeat - count))
            skipped += count
        else:
            result.append(instance)

    if skipped:
        print("Skipping {} instances".format(skipped))
    return result


def load_graphs(graphset):
    graphs = graphset.split(",")
    frame = pd.DataFrame()
    for path in graphs:
        graph = pd.read_pickle(path)
        graph.insert(loc=0, column='graph_set', value=os.path.splitext(path)[0])
        frame = pd.concat([frame, graph], ignore_index=True)
    return frame


def parse_option(value, keys):
    if value == "all":
        return list(keys)
    value = [v.strip() for v in value.split(",")]
    assert all(v in keys for v in value)
    return value


def load_instances(graphset, graphs, scheduler, cluster, bandwidth, netmodel, imode, sched_timing,
                   repeat):
    graphset = load_graphs(graphset)

    if graphs:
        graphset = graphset[graphset["graph_name"].isin(graphs.split(","))].reset_index()

    schedulers = parse_option(scheduler, SCHEDULERS)
    clusters = parse_option(cluster, CLUSTERS)
    bandwidths = parse_option(bandwidth, BANDWIDTHS)
    netmodels = parse_option(netmodel, NETMODELS)
    imodes = parse_option(imode, IMODES)
    sched_timings = parse_option(sched_timing, SCHED_TIMINGS)

    return (
        list(instance_iter(
            graphset.iterrows(),
            clusters,
            bandwidths,
            netmodels,
            schedulers,
            imodes,
            sched_timings,
            repeat)
        ),
        graphset, schedulers, clusters, bandwidths, netmodels, imodes, sched_timings
    )


def compute(graphset, resultfile, scheduler, cluster, bandwidth,
            netmodel, imode, sched_timing, repeat=1,
            no_append=False, graphs=None, timeout=0, interval=None, skip_completed=True,
            dask_cluster=None):
    COLUMNS = ["graph_set",
               "graph_name",
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

    appending = False
    if os.path.isfile(resultfile):
        if no_append:
            print("Result file '{}' already exists\n"
                  "Remove --no-append to append results to it".format(resultfile),
                  file=sys.stderr)
            exit(1)

        appending = True
        print("Appending to result file '{}'".format(resultfile))

        oldframe = pd.read_pickle(resultfile)
        assert list(oldframe.columns) == COLUMNS
    else:
        print("Creating result file '{}'".format(resultfile))
        oldframe = pd.DataFrame([], columns=COLUMNS)

    (instances, graphset, schedulers, clusters, bandwidths, netmodels, imodes, sched_timings) = \
        load_instances(graphset, graphs, scheduler, cluster,
                       bandwidth, netmodel, imode, sched_timing, repeat)
    if len(graphset) == 0:
        print("No graphs selected")
        return

    if interval:
        interval = [min(max(0, int(i)), len(instances)) for i in interval.split(":")]
        instances = instances[interval[0]:interval[1]]

    if appending and skip_completed:
        instances = skip_completed_instances(instances, oldframe, repeat)
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
    print("REPEAT    : {}".format(repeat))
    print("============================================")

    if dask_cluster:
        iterator = run_dask(instances, dask_cluster)
    else:
        pool = multiprocessing.Pool()
        iterator = run_multiprocessing(pool, instances)

    rows = []
    timeout = parse_timeout(timeout)

    if timeout:
        print("Timeout set to {} seconds".format(timeout))

    def run():
        counter = 0
        try:
            for instance, result in tqdm(zip(instances, iterator), total=len(instances)):
                counter += 1
                for r_time, r_runtime in result:
                    rows.append((
                        instance.graph_set,
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

        except KeyboardInterrupt:
            print("Benchmark interrupted, iterated {} instances. Writing intermediate results"
                  .format(counter))

    if timeout:
        thread = threading.Thread(target=run)
        thread.daemon = True
        thread.start()
        thread.join(timeout)
        if thread.is_alive():
            print("Timeout reached")
    else:
        run()

    if not rows:
        print("No results were computed")
        return

    frame = pd.DataFrame(rows, columns=COLUMNS)
    print(frame.groupby(["graph_name", "graph_id", "cluster_name",
                         "bandwidth", "netmodel", "imode", "min_sched_interval", "sched_time",
                         "scheduler_name"]).mean())

    if appending:
        base, ext = os.path.splitext(resultfile)
        path = "{}.backup{}".format(base, ext)
        print("Creating backup of old results to '{}'".format(path))
        oldframe.to_pickle(path)

    # Remove old results
    if not skip_completed:
        oldframe = oldframe[~oldframe.scheduler_name.isin(schedulers)]

    newframe = pd.concat([oldframe, frame], ignore_index=True)
    newframe.to_pickle(resultfile)
    print("{} entries in new '{}'".format(newframe["time"].count(), resultfile))


if __name__ == "__main__":
    args = parse_args()
    compute(**vars(args))
