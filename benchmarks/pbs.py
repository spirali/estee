import json
import os
import socket
import subprocess
import sys

from dask.cluster import start_cluster


DASK_PORT = 8786


def run_computation(options):
    args = []
    if options.get("dask"):
        start_cluster(DASK_PORT)
        args += ["--dask-cluster", "{}:{}".format(socket.gethostname(), DASK_PORT)]
    args = [
        "--scheduler", options.get("scheduler", "all"),
        "--cluster", options.get("cluster", "all"),
        "--netmodel", options.get("netmodel", "all"),
        "--imode", options.get("imode", "all"),
        "--timings", options.get("timings", "all"),
        "--repeat", options["repeat"],
        "--timeout", options["timeout"],
        "--skip-completed"
    ]

    path = os.path.join(os.path.dirname(__file__), "benchmark.py")
    args = ["python", path, options["input"], options["output"]] + args
    subprocess.run(args)


def run_pbs(options):
    args = ["qsub", "-q", "qexp", "-N", "estee", "-l"]
    nodes = "select={}:ncpus=24"
    if options.get("dask"):
        nodes = nodes.format(8)
    else:
        nodes = nodes.format(1)
    args.append(nodes)


if __name__ == "__main__":
    with open(sys.argv[1]) as f:
        options = json.load(f)

    if os.environ.get("PBS_NODEFILE"):
        run_computation(options)
    else:
        run_pbs(options)
