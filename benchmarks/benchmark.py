
from schedsim.common import TaskGraph
from schedsim import Worker, Simulator
from schedsim.communication import MaxMinFlowNetModel
from schedsim.generators import random_levels
from schedsim.schedulers import RandomAssignScheduler, BlevelGtScheduler, RandomGtScheduler, AllOnOneScheduler

import random
import numpy as np
import pandas as pd
import itertools

import multiprocessing


def run_single_instance(task_graph, n_workers, scheduler, bandwidth):
    workers = [Worker() for _ in range(n_workers)]
    connector = MaxMinNetModel(bandwidth)
    simulator = Simulator(task_graph, workers, scheduler, connector)
    return simulator.run()


def benchmark_scheduler(task_graph, scheduler_class, n_workers, bandwidth, count):
    data = np.array(
        [run_single_instance(task_graph, n_workers, scheduler_class(), bandwidth)
            for _ in range(count)])
    average = np.average(data)
    std = data.std()
    minimum = data.min()
    return (minimum, average, std)


n_workers = 3

schedulers = [
    ("srandom", RandomAssignScheduler, 1000),
    ("qrandom", RandomGtScheduler, 1000),
    ("blevel1", lambda: BlevelGtScheduler(False), 1),
    ("blevel2", lambda: BlevelGtScheduler(True), 1),
    ("single", AllOnOneScheduler, 1),
]
bandwidths = [0.01, 0.1, 1.0, 10.0, 100.0]


def process_graph(graph):
    results = []
    for bandwidth in bandwidths:
        data = [bandwidth]
        mins = []
        for name, scheduler, count, in schedulers:
            minimum, average, std = benchmark_scheduler(graph, scheduler, n_workers, bandwidth, count)
            mins.append(minimum)
            data.append(average)
            data.append(std)
        data.append(min(mins))
        results.append(data)
    return results


def main():
    graphs = [make_graph() for _ in range(200)]

    columns = ["task_graph", "workers", "bandwidth"]
    for name, _, _ in schedulers:
        columns.append(name + "_avg")
        columns.append(name + "_std")
    columns.append("min")

    pool = multiprocessing.Pool()

    results = []
    for i, (graph, r) in enumerate(zip(graphs,
                                       pool.imap(process_graph, graphs, 1))):
        print("{}/{}".format(i, len(graphs)))
        graph.cleanup()
        g = [graph]
        for data in r:
            results.append([g, n_workers] + data)

    #for graph in graphs:
    #    results += process_graph(graph)

    data = pd.DataFrame(results, columns=columns)

    #avg_names = [name + "_avg" for name, _, _ in schedulers]
    #for n in avg_names:
    #    for bandwidth in bandwidths:
    #        d = data[data.bandwidth == bandwidth]
    #        r = d[n] / d["min"]
    #        print(n, bandwidth, r.mean())

    data.to_pickle("results.xz")


if __name__ == "__main__":
    main()