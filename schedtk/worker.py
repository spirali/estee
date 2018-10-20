from simpy import Store
from .trace import TaskStartTraceEvent, TaskEndTraceEvent, FetchStartTraceEvent, FetchEndTraceEvent


class Worker:

    def __init__(self, cpus=1):
        self.cpus = cpus
        self.assignments = []
        self.ready_store = None

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

    def assign_task(self, assignment):
        if assignment.task.cpus > self.cpus:
            raise Exception("Task {} allocated on worker with {} cpus"
                            .format(assignment.task, self.cpus))
        assert assignment.worker == self
        self.assignments.append(assignment)
        self._init_downloads(assignment)

    def update_task(self, task):
        for assignment in self.assignments:
            if task == assignment.task:
                self._init_downloads(assignment)
                return
        raise Exception("Updating non assigned task {}, worker={}".format(task, self))

    @property
    def assigned_tasks(self):
        return [a.task for a in self.assignments]

    def _init_downloads(self, assignment):
        deps = []
        not_complete = False
        for input in assignment.task.inputs:
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
            self.ready_store.put(assignment)

        if not_complete:
            return
        if not deps:
            self.ready_store.put(assignment)
        else:
            self.env.process(_helper())

    def run(self, env, simulator, connector):
        self.env = env
        self.simulator = simulator
        self.connector = connector
        self.ready_store = Store(env)

        free_cpus = self.cpus

        prepared_assignments = []
        events = [self.ready_store.get()]

        while True:
            finished = yield env.any_of(events)
            for event in finished.keys():
                if event == events[0]:
                    prepared_assignments.append(event.value)
                    events[0] = self.ready_store.get()
                    continue

                assignment = event.value
                task = assignment.task
                free_cpus += task.cpus
                self.assignments.remove(assignment)
                events.remove(event)
                simulator.add_trace_event(TaskEndTraceEvent(self.env.now, self, task))
                self.data.add(task)
                simulator.on_task_finished(self, task)

            for assignment in prepared_assignments[:]:
                task = assignment.task
                if task.cpus <= free_cpus:
                    free_cpus -= task.cpus
                    simulator.add_trace_event(TaskStartTraceEvent(self.env.now, self, task))
                    events.append(env.timeout(task.duration, assignment))
                    prepared_assignments.remove(assignment)

    def __repr__(self):
        return "<Worker {}>".format(id(self))