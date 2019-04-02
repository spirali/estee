import collections
import itertools
import multiprocessing
import os
import random
import re
import sys
import threading
import time
import traceback

import click
import numpy
import pandas as pd
from tqdm import tqdm

from estee.common import imode
from estee.schedulers import WorkStealingScheduler
from estee.schedulers.basic import AllOnOneScheduler, RandomAssignScheduler
from estee.schedulers.camp import Camp2Scheduler
from estee.schedulers.clustering import LcScheduler
from estee.schedulers.genetic import GeneticScheduler
from estee.schedulers.others import BlevelScheduler, DLSScheduler, ETFScheduler, MCPGTScheduler, \
    MCPScheduler, TlevelScheduler
from estee.schedulers.queue import BlevelGtScheduler, RandomGtScheduler, TlevelGtScheduler
from estee.serialization.dask_json import json_deserialize, json_serialize
from estee.simulator import MaxMinFlowNetModel, SimpleNetModel
from estee.simulator import Simulator, Worker
from estee.simulator.trace import FetchEndTraceEvent


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
    "blevel": BlevelScheduler,
    "blevel-gt": BlevelGtScheduler,
    "tlevel": TlevelScheduler,
    "tlevel-gt": TlevelGtScheduler,
    "random": RandomAssignScheduler,
    "random-gt": RandomGtScheduler,
    "dls": DLSScheduler,
    "etf": ETFScheduler,
    "mcp": MCPScheduler,
    "mcp-gt": MCPGTScheduler,
    "genetic": GeneticScheduler,
    "camp2": lambda: Camp2Scheduler(5000),
    "lc": LcScheduler,
    "ws": WorkStealingScheduler
}

NETMODELS = {
    "simple": SimpleNetModel,
    "maxmin": MaxMinFlowNetModel
}

CLUSTERS = {
    "2x8": [{"cpus": 8}] * 2,
    "4x4": [{"cpus": 4}] * 4,
    "8x4": [{"cpus": 4}] * 8,
    "16x4": [{"cpus": 4}] * 16,
    "32x4": [{"cpus": 4}] * 32,
    "8x8": [{"cpus": 8}] * 8,
    "16x8": [{"cpus": 8}] * 16,
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
    "0/0": (0, 0),
    "0.1/0.05": (0.1, 0.05),
    "0.4/0.05": (0.4, 0.05),
    "1.6/0.05": (1.6, 0.05),
    "6.4/0.05": (6.4, 0.05)
}

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
           "execution_time",
           "total_transfer"]

Instance = collections.namedtuple("Instance",
                                  ("graph_set", "graph_name", "graph_id", "graph",
                                   "cluster_name", "bandwidth", "netmodel",
                                   "scheduler_name", "imode", "min_sched_interval", "sched_time",
                                   "count"))


class BenchmarkConfig:
    graph_cache = {}

    def __init__(self, graph_frame, schedulers, clusters, netmodels, bandwidths,
                 imodes, sched_timings, count):
        self.graph_frame = graph_frame
        self.schedulers = schedulers
        self.clusters = clusters
        self.netmodels = netmodels
        self.bandwidths = bandwidths
        self.imodes = imodes
        self.sched_timings = sched_timings
        self.count = count

    def generate_instances(self):
        def calculate_imodes(graph, graph_id):
            if graph_id not in BenchmarkConfig.graph_cache:
                BenchmarkConfig.graph_cache[graph_id] = {}
                for mode in IMODES:
                    g = json_deserialize(graph)
                    IMODES[mode](g)
                    BenchmarkConfig.graph_cache[graph_id][mode] = json_serialize(g)

        for graph_def, cluster_name, bandwidth, netmodel, scheduler_name, mode, sched_timing \
                in itertools.product(self.graph_frame.iterrows(), self.clusters, self.bandwidths,
                                     self.netmodels, self.schedulers, self.imodes,
                                     self.sched_timings):
            g = graph_def[1]
            calculate_imodes(g["graph"], g["graph_id"])
            graph = BenchmarkConfig.graph_cache[g["graph_id"]][mode]

            (min_sched_interval, sched_time) = SCHED_TIMINGS[sched_timing]
            instance = Instance(
                g["graph_set"], g["graph_name"], g["graph_id"], graph,
                cluster_name, BANDWIDTHS[bandwidth], netmodel,
                scheduler_name,
                mode,
                min_sched_interval, sched_time,
                self.count)
            yield instance

    def __repr__(self):
        return """
============ Config ========================
scheduler : {schedulers}
cluster   : {clusters}
netmodel  : {netmodels}
bandwidths: {bandwidths}
imode     : {imodes}
timings   : {timings}
REPEAT    : {repeat}
============================================
        """.format(
            schedulers=", ".join(self.schedulers),
            clusters=", ".join(self.clusters),
            netmodels=", ".join(self.netmodels),
            bandwidths=", ".join(self.bandwidths),
            imodes=", ".join(self.imodes),
            timings=", ".join(self.sched_timings),
            repeat=self.count
        )


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
    simulator = Simulator(instance.graph, workers, scheduler, netmodel, trace=True)
    try:
        sim_time = simulator.run()
        runtime = time.monotonic() - begin_time
        transfer = 0
        for e in simulator.trace_events:
            if isinstance(e, FetchEndTraceEvent):
                transfer += e.output.size
        return sim_time, runtime, transfer
    except Exception:
        traceback.print_exc()
        print("ERROR INSTANCE: {}".format(instance), file=sys.stderr)
        return None, None, None


def benchmark_scheduler(instance):
    return [run_single_instance(instance)
            for _ in range(instance.count)]


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


def compute(instances, timeout=0, dask_cluster=None):
    rows = []

    if not instances:
        return rows

    if dask_cluster:
        iterator = run_dask(instances, dask_cluster)
    else:
        pool = multiprocessing.Pool()
        iterator = run_multiprocessing(pool, instances)

    if timeout:
        print("Timeout set to {} seconds".format(timeout))

    def run():
        counter = 0
        try:
            for instance, result in tqdm(zip(instances, iterator), total=len(instances)):
                counter += 1
                for r_time, r_runtime, r_transfer in result:
                    if r_time is not None:
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
                            r_transfer
                        ))
        except:
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

    return rows


def run_benchmark(configs, oldframe, resultfile, skip_completed, timeout=0, dask_cluster=None):
    for config in configs:
        print(config)

    instances = create_instances(configs, oldframe, skip_completed, 5)
    rows = compute(instances, timeout, dask_cluster)
    if not rows:
        print("No results were computed")
        return

    frame = pd.DataFrame(rows, columns=COLUMNS)
    print(frame.groupby(["graph_name", "graph_id", "cluster_name",
                         "bandwidth", "netmodel", "imode", "min_sched_interval", "sched_time",
                         "scheduler_name"]).mean())

    if len(frame) > 0:
        base, ext = os.path.splitext(resultfile)
        path = "{}.backup{}".format(base, ext)
        print("Creating backup of old results to '{}'".format(path))
        write_resultfile(oldframe, path)

    newframe = pd.concat([oldframe, frame], ignore_index=True)
    write_resultfile(newframe, resultfile)
    print("{} entries in new '{}'".format(newframe["time"].count(), resultfile))


def skip_completed_instances(instances, frame, repeat, columns, batch):
    skipped_output = 0
    skipped_batch = 0
    counts = frame.groupby(columns).size()

    result = []
    for instance in instances:
        hashed = tuple(getattr(instance, col) for col in columns)
        existing_count = 0
        if hashed in counts:
            count = counts.loc[hashed]
            skipped_output += count
            existing_count += count
        if hashed in batch:
            count = batch[hashed]
            skipped_batch += count
            existing_count += count
        if existing_count == 0:
            result.append(instance)
        elif existing_count < repeat:
            result.append(instance._replace(count=repeat - existing_count))

    if skipped_output or skipped_batch:
        print("Skipping {} instances from output, {} from batch, {} left".format(skipped_output,
                                                                                 skipped_batch,
                                                                                 len(result)))
    return result


def limit_max_count(instances, max_count):
    result = []
    for instance in instances:
        if instance.count > max_count:
            remaining = instance.count
            while remaining > 0:
                count = min(max_count, remaining)
                remaining -= count
                result.append(instance._replace(count=count))
        else:
            result.append(instance)

    return result


def create_instances(configs, frame, skip_completed, max_count):
    total_instances = []
    columns = ["graph_id",
               "cluster_name",
               "bandwidth",
               "netmodel",
               "scheduler_name",
               "imode",
               "min_sched_interval",
               "sched_time"]
    batch = {}

    for config in configs:
        instances = list(config.generate_instances())
        if skip_completed:
            instances = skip_completed_instances(instances, frame, config.count, columns, batch)
        instances = limit_max_count(instances, max_count)
        for instance in instances:
            hashed = tuple(getattr(instance, col) for col in columns)
            batch[hashed] = instance.count + batch.get(hashed, 0)
        total_instances += instances

    return total_instances


def load_resultfile(resultfile, append):
    if os.path.isfile(resultfile):
        if not append:
            print("Result file '{}' already exists\n"
                  "Remove --no-append to append results to it".format(resultfile),
                  file=sys.stderr)
            exit(1)

        print("Appending to result file '{}'".format(resultfile))

        oldframe = pd.read_csv(resultfile)
        assert list(oldframe.columns) == COLUMNS
    else:
        print("Creating result file '{}'".format(resultfile))
        oldframe = pd.DataFrame([], columns=COLUMNS)
    return oldframe


def write_resultfile(frame, resultfile):
    frame.to_csv(resultfile, compression='zip', index=False)


def load_graphs(graphsets, graph_names=None):
    frame = pd.DataFrame()
    for path in graphsets:
        graph = pd.read_json(path)
        graph.insert(loc=0, column='graph_set', value=os.path.splitext(path)[0])
        frame = pd.concat([frame, graph], ignore_index=True)

    if graph_names:
        frame = frame[frame["graph_name"].isin(graph_names)].reset_index()
    return frame


def generate_help(keys):
    return "all,{}".format(",".join(keys))


def parse_timeout(timeout):
    if not timeout:
        return 0
    match = re.match(r"^(\d{2}):(\d{2}):(\d{2})$", timeout)
    if not match:
        print("Wrong timeout format. Enter timeout as hh:mm:ss.")
        exit(1)
    return int(match.group(1)) * 3600 + int(match.group(2)) * 60 + int(match.group(3))


@click.command()
@click.argument("graphset")
@click.argument("resultfile")
@click.option("--scheduler", default="all", help=generate_help(SCHEDULERS))
@click.option("--cluster", default="all", help=generate_help(CLUSTERS))
@click.option("--bandwidth", default="all", help=generate_help(BANDWIDTHS))
@click.option("--netmodel", default="all", help=generate_help(NETMODELS))
@click.option("--imode", default="all", help=generate_help(IMODES))
@click.option("--sched-timing", default="all", help=generate_help(SCHED_TIMINGS))
@click.option("--repeat", default=1)
@click.option("--append/--no-append", default=True, help="Exit if the resultfile already exists.")
@click.option("--skip-completed/--no-skip_completed", default=True,
              help="Skip already computed instances found in the resultfile.")
@click.option("--graphs", help="Comma separated list of graphs to be used from the input graphset")
@click.option("--timeout", help="Timeout for the computation. Format hh:mm:ss.")
@click.option("--dask-cluster", help="Address of Dask scheduler.")
def compute_cmd(graphset, resultfile, scheduler, cluster, bandwidth,
                netmodel, imode, sched_timing, repeat, append, skip_completed,
                graphs, timeout, dask_cluster):
    def parse_option(value, keys):
        if value == "all":
            return list(keys)
        value = [v.strip() for v in value.split(",")]
        assert all(v in keys for v in value)
        return value

    graphsets = graphset.split(",")
    schedulers = parse_option(scheduler, SCHEDULERS)
    clusters = parse_option(cluster, CLUSTERS)
    bandwidths = parse_option(bandwidth, BANDWIDTHS)
    netmodels = parse_option(netmodel, NETMODELS)
    imodes = parse_option(imode, IMODES)
    sched_timings = parse_option(sched_timing, SCHED_TIMINGS)
    timeout = parse_timeout(timeout)

    graph_frame = load_graphs(graphsets, None if graphs is None else graphs.split(","))
    if len(graph_frame) == 0:
        print("No graphs selected")
        exit()

    config = BenchmarkConfig(graph_frame, schedulers, clusters, netmodels, bandwidths, imodes,
                             sched_timings, repeat)
    frame = load_resultfile(resultfile, append)

    run_benchmark([config], frame, resultfile, skip_completed, timeout, dask_cluster)


if __name__ == "__main__":
    compute_cmd()
