
from simpy import Event


class Connector:

    def __init__(self):
        pass

    def init(self, env, workers):
        self.workers = workers
        self.env = env


class InstantConnector(Connector):

    def download(self, source, target, size):
        event = Event(self.env)
        event.succeed()
        return event
