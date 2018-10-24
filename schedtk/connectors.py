
from simpy import Event


class Connector:
    """
        bandwidth - maximal bandwidth between two nodes
                    (this is what network annoucess publicaly, not necessery how it really behaves)
    """

    def __init__(self, bandwidth=1.0):
        self.bandwidth = bandwidth

    def init(self, env, workers):
        self.workers = workers
        self.env = env


class InstantConnector(Connector):

    def __init__(self):
        super().__init__(float("inf"))

    def download(self, source, target, size):
        event = Event(self.env)
        event.succeed()
        return event


class SimpleConnector(Connector):

    def download(self, source, target, size):
        return self.env.timeout(size / self.bandwidth)
