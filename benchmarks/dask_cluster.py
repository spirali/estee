import argparse
import os
import socket
import subprocess
import tempfile
import time

from distributed import Client

TMP_DIR = tempfile.gettempdir()
HOSTNAME = socket.gethostname()


def get_nodes():
    with open(os.environ['PBS_NODEFILE']) as f:
        return [line.strip() for line in f.readlines()]


def is_local(hostname):
    return hostname.split(".")[0] == HOSTNAME or hostname == HOSTNAME or hostname == "localhost"


def run_cmd(host, cmds, path=""):
    args = ["env", "OMP_NUM_THREADS=1"]
    if path:
        args += ["PYTHONPATH=${{PYTHONPATH}}:{}".format(path)]
    args += cmds

    if not is_local(host):
        args = ["ssh", host, "--", "workon", "estee", "&&" "ulimit", "-u", "32768", "&&"] + args

    args = ["nohup"] + args
    subprocess.Popen(args)


def spawn_workers(host, count, scheduler, path):
    print("Spawning {} workers on {}".format(count, host))
    cookie = time.time()
    return run_cmd(host, ["dask-worker", "--nthreads", "1", "--nprocs", str(count),
                          "--local-directory", os.path.join(TMP_DIR,
                                                            "estee-{}".format(cookie)),
                          scheduler], path)


def kill_workers(host):
    return run_cmd(host, ["pkill", "-f", "-9", "dask-worker"])


def kill_scheduler(host):
    return run_cmd(host, ["pkill", "-f", "-9", "dask-scheduler"])


def spawn_scheduler(host, port, path):
    return run_cmd(host, ["dask-scheduler", "--port", str(port)], path)


def start_cluster(port, path, procs=24):
    print("Starting cluster...")
    spawn_scheduler(HOSTNAME, port, path)
    time.sleep(5)
    scheduler = "{}:{}".format(HOSTNAME, port)
    spawn_workers(HOSTNAME, procs - 1, scheduler, path)

    nodes = get_nodes()
    for node in nodes:
        if not is_local(node):
            spawn_workers(node, procs, scheduler, path)

    client = Client(scheduler)

    print("Waiting for workers to connect...")
    target_workers = procs * len(nodes) - 1
    while True:
        worker_count = len(client.scheduler_info()['workers'])
        if worker_count >= target_workers:
            break
        print("Worker count: {}".format(worker_count))
        time.sleep(2)

    print("Cluster is running at {}".format(scheduler))


def stop_cluster():
    kill_scheduler(HOSTNAME)
    kill_workers(HOSTNAME)


def get_info(port):
    scheduler = "{}:{}".format(HOSTNAME, port)
    client = Client(scheduler, timeout=5)
    workers = client.scheduler_info()['workers']
    print(workers)
    print("Worker count: {}".format(len(workers)))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["start", "stop", "info"])
    parser.add_argument("--port", default=8786)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    port = int(args.port)

    if args.command == "start":
        start_cluster(port=port, path=os.getcwd())
    elif args.command == "stop":
        stop_cluster()
    elif args.command == "info":
        get_info(port)
