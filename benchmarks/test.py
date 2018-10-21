import pandas as pd
import matplotlib.pyplot as plt
import seaborn
import numpy as np
import argparse

from schedtk import Worker, Simulator
from schedtk.connectors import SimpleConnector
from schedtk.schedulers import CampScheduler, AllOnOneScheduler
import multiprocessing

SCHEDULERS = {
    "single": AllOnOneScheduler,
    "camp": CampScheduler,
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
    name, workers = row.cluster
    row.task_graph.write_dot("/tmp/xx.dot")
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
    return parser.parse_args()


def main():
    args = parse_args()
    data = pd.read_pickle(args.dataset)[0:150]

    pool = multiprocessing.Pool()
    results = []
    #print(data)
    #for r in pool.imap(process, data_iter(args.scheduler, args.repeat, data)):
    #    results.append(r)
    #    if len(results) % 50 == 0:
    #        print("Step {}/{}".format(len(results), len(data)))


    for i, r in enumerate(data_iter(args.scheduler, args.repeat, data)):
        results.append(process(r))

    frame = pd.DataFrame(results, columns=[args.scheduler + "_avg", args.scheduler + "_std", args.scheduler + "_min"])
    result_data = data #pd.concat([data, frame], axis=1)

    print(result_data)
    print(frame)

    if args.output:
        result_data.to_pickle(args.output)

    mc = [c for c in data.columns if c.endswith("_min")]
    if mc:
        m = data[mc].min(axis=1)
        #print(frame[args.scheduler + "_avg"] / m)
        #print(m)
        #print(frame[args.scheduler + "_avg"])





    #frame["avg"] /= data["min"]

    #df = frame.groupby("bandwidth")["avg"].mean()

    #print(df)

    #seaborn.violinplot(y="avg", x="bandwidth", data=frame, palette="Set3")
    #plt.show()

main()
