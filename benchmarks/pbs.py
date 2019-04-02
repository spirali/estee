import json
import os
import socket
import subprocess
import sys
import time

import click
import pandas as pd

from benchmark import BenchmarkConfig, load_graphs, SCHEDULERS, CLUSTERS, NETMODELS, BANDWIDTHS, \
    IMODES, SCHED_TIMINGS, create_instances, run_benchmark, parse_timeout, load_resultfile

BENCHMARK_DIR = os.path.dirname(os.path.abspath(__file__))
DASK_PORT = 8786


def dirpath(path):
    return os.path.join(BENCHMARK_DIR, path)


def filename(path):
    return os.path.splitext(os.path.basename(path))[0]


def get_workdir(jobid, input_file, output):
    return os.path.abspath("runs/{}-{}-{}".format(jobid, filename(input_file), filename(output)))


def parse_configs(definition, graph_frame):
    groups = definition["groups"]
    experiments = definition["experiments"]
    group_cache = {}
    configs = []
    keys = {
        "scheduler": SCHEDULERS,
        "cluster": CLUSTERS,
        "netmodel": NETMODELS,
        "bandwidth": BANDWIDTHS,
        "imode": IMODES,
        "sched-timing": SCHED_TIMINGS
    }
    in_progress = set()

    def get_group(name):
        if name in in_progress:
            print("Recursive definition of {}".format(name))
            exit(1)
        in_progress.add(name)

        data = groups[name]
        result = {}
        if isinstance(data, dict):
            key = data["type"]
            value = data["values"]
            if key != "repeat":
                if value == "all":
                    value = set(keys[key].keys())
                else:
                    value = set(value.split(","))
            result[key] = value
        elif isinstance(data, list):
            result = {}
            for group in data:
                group_data = get_group(group)
                for key in group_data:
                    if key in result:
                        if key == "repeat":
                            result[key] = max(group_data[key], result[key])
                        else:
                            result[key].update(group_data[key])
                    else:
                        result[key] = group_data[key]
        else:
            assert False

        in_progress.remove(name)
        return result

    def get_value(data, key):
        if key == "repeat":
            return data.get(key, 1)

        return list(data.get(key, keys[key].keys()))

    for experiment in experiments:
        data = group_cache.setdefault(experiment, get_group(experiment))
        configs.append(BenchmarkConfig(
            graph_frame,
            get_value(data, "scheduler"),
            get_value(data, "cluster"),
            get_value(data, "netmodel"),
            get_value(data, "bandwidth"),
            get_value(data, "imode"),
            get_value(data, "sched-timing"),
            get_value(data, "repeat")
        ))
    return configs


def run_computation(index, input_file, definition):
    from dask_cluster import start_cluster

    input = definition["inputs"][int(index)]
    output = definition["outputs"][int(index)]

    workdir = get_workdir(os.environ.get("PBS_JOBID", "local-{}".format(index)), input_file, output)

    if not os.path.exists(workdir):
        os.makedirs(workdir)
    with open(os.path.join(workdir, os.path.basename(input_file)), "w") as dst:
        definition["index"] = index
        json.dump(definition, dst, indent=4)

    dask_cluster = None
    if definition.get("dask"):
        start_cluster(port=DASK_PORT, path=BENCHMARK_DIR)
        dask_cluster = "{}:{}".format(socket.gethostname(), DASK_PORT)

    graph_frame = load_graphs([input])

    with open(os.path.join(workdir, "output"), "w") as out:
        with open(os.path.join(workdir, "error"), "w") as err:
            sys.stdout = out
            sys.stderr = err
            frame = load_resultfile(output, True)
            run_benchmark(parse_configs(definition, graph_frame), frame, output, True,
                          parse_timeout(definition.get("timeout")), dask_cluster)


def run_pbs(input_file, definition):
    nodes = 1
    if definition.get("dask"):
        nodes = 8

    print("Starting jobs from file {}".format(input_file))
    for i, input in enumerate(definition["inputs"]):
        input = definition["inputs"][i]
        output = definition["outputs"][i]

        graph_frame = load_graphs([input])
        configs = parse_configs(definition, graph_frame)

        if os.path.isfile(output):
            oldframe = pd.read_csv(output)
            instances = create_instances(configs, oldframe, True, 5)
            if not instances:
                print("All instances were completed for {}".format(input))
                continue

        name = "estee-{}-{}".format(filename(input_file), filename(output))
        qsub_args = {
            "benchmark_dir": BENCHMARK_DIR,
            "name": name,
            "input": os.path.abspath(input_file),
            "index": i,
            "nodes": nodes,
            "working_directory": os.getcwd()
        }
        qsub_input = """
#!/bin/bash
#PBS -q qexp
#PBS -N {name}
#PBS -lselect={nodes}:ncpus=24

source ~/.bashrc
workon estee
cd {working_directory}
python {benchmark_dir}/pbs.py compute {input} --index {index}
""".format(**qsub_args)

        pbs_script = "/tmp/{}-pbs-{}.sh".format(name, int(time.time()))
        with open(pbs_script, "w") as f:
            f.write(qsub_input)

        print("Starting job {}-{} ({})".format(filename(input_file), filename(output), pbs_script))
        result = subprocess.run(["qsub", pbs_script],
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)
        if result.returncode != 0:
            raise Exception("Error during PBS submit: {}\n{}".format(result.stdout.decode(),
                                                                     result.stderr.decode()))
        print("Job id: {}".format(result.stdout.decode().strip()))


@click.command()
@click.argument("input_file")
@click.option("--index")
def compute(input_file, index):
    with open(input_file) as f:
        definition = json.load(f)

    if index is not None:
        run_computation(index, input_file, definition)
    else:
        for index in range(len(definition["inputs"])):
            run_computation(index, input_file, definition)


@click.command()
@click.argument("input_files", nargs=-1)
def submit(input_files):
    for input_file in input_files:
        with open(input_file) as f:
            definition = json.load(f)
        run_pbs(input_file, definition)


@click.group()
def cli():
    pass


if __name__ == "__main__":
    cli.add_command(submit)
    cli.add_command(compute)
    cli()
