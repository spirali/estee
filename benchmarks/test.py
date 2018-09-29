import pandas as pd
import matplotlib.pyplot as plt
import seaborn
import numpy as np

from schedtk import Worker, Simulator
from schedtk.connectors import SimpleConnector
from schedtk.schedulers import AllOnOneScheduler


def run_single_instance(task_graph, n_workers, scheduler, bandwidth):
    workers = [Worker() for _ in range(n_workers)]
    connector = SimpleConnector(bandwidth)
    simulator = Simulator(task_graph, workers, scheduler, connector)
    return simulator.run()


def benchmark_scheduler(task_graph, scheduler_class, n_workers, bandwidth, count):
    data = np.array(
        [run_single_instance(task_graph, n_workers, scheduler_class(), bandwidth)
            for _ in range(count)])
    return np.average(data)


def main():
    data = pd.read_pickle("dataset1.xz")
    count = 1
    scheduler = AllOnOneScheduler

    results = []
    for i, row in data.iterrows():
        graph = row.task_graph[0]
        r = [graph, row.workers, row.bandwidth]
        r.append(benchmark_scheduler(graph, scheduler, row.workers, row.bandwidth, count))
        results.append(r)

    frame = pd.DataFrame(results, columns=["task_graph", "workers", "bandwidth", "avg"])
    frame["avg"] /= data["min"]

    df = frame.groupby("bandwidth")["avg"].mean()
    print(df)


main()
