import argparse
import multiprocessing
import sys

import numpy as np
import pandas as pd

from schedtk import Simulator, Worker
from schedtk.connectors import SimpleConnector
from schedtk.schedulers import AllOnOneScheduler, BlevelGtScheduler, \
    Camp2Scheduler, DLSScheduler, K1hScheduler, RandomAssignScheduler, \
    RandomGtScheduler

sys.setrecursionlimit(2500)

SCHEDULERS = {
    "single": AllOnOneScheduler,
    "camp2": lambda: Camp2Scheduler(5000),
    "random-s": RandomAssignScheduler,
    "random-gt": RandomGtScheduler,
    "blevel": BlevelGtScheduler,
    "k1h": K1hScheduler,
    "dls": DLSScheduler
}


def run_single_instance(task_graph, workers, scheduler, bandwidth):
    workers = [Worker(**wargs) for wargs in workers]
    connector = SimpleConnector(bandwidth)
    simulator = Simulator(task_graph, workers, scheduler, connector)
    return simulator.run()


def benchmark_scheduler(task_graph, scheduler_class, workers, bandwidth, count):
    return np.array(
        [run_single_instance(task_graph, workers, scheduler_class(), bandwidth)
         for _ in range(count)])


def process(instance):
    scheduler_name, count, row = instance
    scheduler = SCHEDULERS[scheduler_name]
    workers = row.cluster_def
    #row.task_graph.write_dot("/tmp/xx.dot")
    times = benchmark_scheduler(row.task_graph, scheduler, workers, row.bandwidth, count)
    row.task_graph.cleanup()
    return np.average(times), np.std(times), times.min()


def data_iter(scheduler_name, count, data):
    for _, row in data.iterrows():
        yield scheduler_name, count, row


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("scheduler", choices=tuple(SCHEDULERS.keys()))
    parser.add_argument("repeat", type=int)
    parser.add_argument("--output")
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    data = pd.read_pickle(args.dataset)

    if args.limit:
        data = data[:args.limit]

    pool = multiprocessing.Pool()
    results = []

    # Single therad
    #for i, r in enumerate(data_iter(args.scheduler, args.repeat, data)):
    #    results.append(process(r))

    for r in pool.imap(process, data_iter(args.scheduler, args.repeat, data)):
        results.append(r)
        if len(results) % 50 == 0:
            print(len(results))

    print("Testing scheduler: {}".format(args.scheduler))

    frame = pd.DataFrame(results, columns=[args.scheduler + "_avg", args.scheduler + "_std", args.scheduler + "_min"])
    mc = [c for c in data.columns if c.endswith("_min")]
    if mc:
        old_min = data[mc].min(axis=1)

    for c in frame.columns:
        if c in data.columns:
            del data[c]
    result_data = pd.concat([data, frame], axis=1)

    if args.output:
        result_data.to_pickle(args.output)

    avg_name = args.scheduler + "_avg"
    if mc:
        result_data["rr"] = result_data[avg_name] / old_min
        print(result_data.groupby(["bandwidth", "cluster_name"])["rr"].mean())

if __name__ == "__main__":
    main()
