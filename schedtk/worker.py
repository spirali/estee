from simpy import Store
from .trace import TaskStartTraceEvent, TaskEndTraceEvent, FetchStartTraceEvent, FetchEndTraceEvent


class Worker:

    def __init__(self):
        self.assigned_tasks = []
        self.ready_tasks = None

        self.data = set()
        self.downloads = {}

    def _download(self, worker, task):
        def _helper():
            yield self.connector.download(worker, self, task.size)
            self.simulator.add_trace_event(FetchEndTraceEvent(self.env.now, self, worker, task))
            del self.downloads[task]
            self.data.add(task)

        assert worker != self
        self.simulator.add_trace_event(FetchStartTraceEvent(self.env.now, self, worker, task))
        p = self.env.process(_helper())
        self.downloads[task] = p
        return p

    def assign_task(self, task):
        self.assigned_tasks.append(task)
        self._init_downloads(task)

    def update_task(self, task):
        assert task in self.assigned_tasks
        self._init_downloads(task)

    def _init_downloads(self, task):
        deps = []
        not_complete = False
        for input in task.inputs:
            if input in self.data:
                continue

            d = self.downloads.get(input)
            if d is None:
                if input.info.is_finished:
                    worker = input.info.assigned_workers[0]
                    d = self._download(worker, input)
                else:
                    not_complete = True
            deps.append(d)

        def _helper():
            yield self.env.all_of(deps)
            self.ready_tasks.put(task)

        if not_complete:
            return
        if not deps:
            self.ready_tasks.put(task)
        else:
            self.env.process(_helper())

    def run(self, env, simulator, connector):
        self.env = env
        self.simulator = simulator
        self.connector = connector
        self.ready_tasks = Store(env)

        while True:
            task = yield self.ready_tasks.get()
            simulator.add_trace_event(TaskStartTraceEvent(self.env.now, self, task))
            yield env.timeout(task.duration)
            simulator.add_trace_event(TaskEndTraceEvent(self.env.now, self, task))
            self.assigned_tasks.remove(task)
            self.data.add(task)
            simulator.on_task_finished(self, task)

    def __repr__(self):
        return "<Worker {}>".format(id(self))