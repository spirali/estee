
import logging

import numpy as np
from simpy import Event

from ..simulator.trace import BandwidthChangeEvent

logger = logging.getLogger(__name__)


class NetModel:
    """
        bandwidth - maximal bandwidth between two nodes
                    (this is what the network announces publicly,
                    not necessary how it really behaves)
    """

    def __init__(self, bandwidth=1.0):
        self.bandwidth = float(bandwidth)
        self.worker_bandwidth = {}
        self.event_listener = None

    def init(self, env, workers):
        self.env = env
        self.workers = workers
        for worker in workers:
            assert worker.id is not None
        self.worker_bandwidth = {worker: {
            True: 0,    # out
            False: 0    # in
        } for worker in workers}

    def set_event_listener(self, listener):
        self.event_listener = listener

    def update_bandwidth(self, worker, out, bandwidth):
        if not self.event_listener:
            return

        if self.worker_bandwidth[worker][out] != bandwidth:
            self.worker_bandwidth[worker][out] = bandwidth
            self.event_listener(BandwidthChangeEvent(self.env.now, out, worker, bandwidth))

    def get_bandwidth(self, worker, out):
        return self.worker_bandwidth[worker][out]


class InstantNetModel(NetModel):

    def __init__(self):
        super().__init__(float("inf"))

    def download(self, source, target, size, value=None):
        assert source != target
        event = Event(self.env)
        event.succeed(value)
        return event


class SimpleNetModel(NetModel):

    def download(self, source, target, size, value=None):
        assert source != target

        e = self.env.timeout(size / self.bandwidth, value)

        if self.event_listener:
            self.add_bandwidth(source, target, self.bandwidth)
            e.callbacks.append(lambda _: self.add_bandwidth(source, target, -self.bandwidth))
        return e

    def add_bandwidth(self, source, target, amount):
        self.update_bandwidth(source, True, self.get_bandwidth(source, True) + amount)
        self.update_bandwidth(target, False, self.get_bandwidth(target, False) + amount)


"""
class Download:

    def __init__(self, size, event):
        self.source = source
        self.target = target
        self.size = size
"""


class RunningDownload:

    __slots__ = ("size", "speed", "event", "value")

    def __init__(self, size, event, value):
        self.size = size
        self.event = event
        self.speed = None
        self.value = value

    def __repr__(self):
        return "<RD {} {} {}>".format(id(self), self.size, self.speed)


class MaxMinFlowNetModel(NetModel):

    def init(self, env, workers):
        super().init(env, workers)
        self.downloads = {}
        self.recompute_event = Event(env)
        self.flows = np.zeros((len(workers), len(workers)))

        self.recompute_flows = False

        def network_process():
            while True:
                if self.recompute_flows:
                    self.recompute_flows = False
                    self._recompute_flows()
                    logger.info("Flows reconfigured:\n%s", self.flows)

                timeout = self._update_speeds()
                logger.info("Earliest download finished in: %s", timeout)
                if timeout is not None:
                    start_time = self.env.now
                    r = yield self.recompute_event | self.env.timeout(timeout)
                    self._update_sizes(self.env.now - start_time)
                    if self.recompute_event in r:
                        self.recompute_event = Event(env)
                else:
                    logger.info("No active downloads")
                    yield self.recompute_event
                    self.recompute_event = Event(env)
        env.process(network_process())

    def download(self, source, target, size, value=None):
        assert source != target
        event = Event(self.env)
        rd = RunningDownload(size, event, value)
        logger.info("New download %s; %s-%s size=%s", rd, source, target, size)
        key = (source, target)
        lst = self.downloads.get(key)
        if lst is None:
            lst = []
            self.downloads[key] = lst
        if not lst:
            logger.info("Link %s-%s opened, need recompute flows", source, target)
            self.recompute_flows = True
        lst.append(rd)
        if not self.recompute_event.triggered:
            self.recompute_event.succeed()
        return event

    def _update_speeds(self):
        timeout = None
        for (source, target), lst in self.downloads.items():
            if not lst:
                continue
            speed = self.flows[source.id, target.id] / len(lst)
            for download in lst:
                download.speed = speed
                t = download.size / speed
                timeout = min(t, timeout) if timeout is not None else t
        return timeout

    def _update_sizes(self, time):
        for key, lst in self.downloads.items():
            if not lst:
                continue
            for download in lst[:]:
                if download.speed is None:  # Freshly scheduled
                    continue
                size = download.size - time * download.speed
                if size < 0.00000001:
                    logger.info("Download finished %s", download)
                    lst.remove(download)
                    download.event.succeed(download.value)
                download.size = size
            if not lst:
                source, target = key
                logger.info("Link %s-%s opened, need recompute flows", source, target)
                self.recompute_flows = True

    def _recompute_flows(self):
        connections = np.zeros_like(self.flows, dtype=np.int32)
        for (source, target), lst in self.downloads.items():
            if lst:
                connections[source.id, target.id] = 1
        send_capacities = np.full(len(self.workers), self.bandwidth)
        recv_capacities = send_capacities.copy()
        self.flows = compute_maxmin_flow(send_capacities, recv_capacities, connections)
        self._trace_flows()

    def _trace_flows(self):
        if self.event_listener:
            for w in range(len(self.workers)):
                bw_out = np.sum(self.flows[w, :])
                bw_in = np.sum(self.flows[:, w])
                self.update_bandwidth(self.workers[w], True, bw_out)
                self.update_bandwidth(self.workers[w], False, bw_in)


def compute_maxmin_flow(send_capacities, recv_capacities, connections):
    result = np.zeros_like(connections, dtype=np.float)
    with np.errstate(divide='ignore', invalid='ignore'):
        count = connections.sum()
        while count:
            sends = send_capacities / connections.sum(axis=1)
            recvs = recv_capacities / connections.sum(axis=0)
            sa = np.nanargmin(sends)
            sm = sends[sa]
            ra = np.nanargmin(recvs)
            rm = recvs[ra]
            if sm <= rm:
                d = connections[sa]
                count -= d.sum()
                flow = d * sm
                recv_capacities -= flow
                connections[sa] = 0
                result[sa] += flow
            else:
                d = connections[:, ra]
                count -= d.sum()
                flow = d * rm
                send_capacities -= flow
                connections[:, ra] = 0
                result[:, ra] += flow
    return result
