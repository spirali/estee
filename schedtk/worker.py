from simpy import Store

from .trace import FetchEndTraceEvent, FetchStartTraceEvent, \
    TaskEndTraceEvent, TaskStartTraceEvent


class RunningTask:

    __slots__ = ("task", "start_time")

    def __init__(self, task, start_time):
        self.task = task
        self.start_time = start_time

    def running_time(self, now):
        return now - self.start_time

    def remaining_time(self, now):
        return self.task.duration - self.running_time(now)


class RunningDownload:

    __slots__ = ("process", "task", "start_time")

    def __init__(self, process, task, start_time):
        self.process = process
        self.task = task
        self.start_time = start_time

    def running_time(self, now):
        return now - self.start_time

    def naive_remaining_time_estimate(self, simulator):
        return self.task.size / simulator.connector.bandwidth + self.start_time - simulator.env.now


class Worker:

    def __init__(self, cpus=1):
        self.cpus = cpus
        self.assignments = []
        self.ready_store = None

        self.data = set()
        self.running_tasks = {}
        self.running_downloads = {}

        self.free_cpus = cpus

    def _download(self, worker, task):
        def _helper():
            yield self.connector.download(worker, self, task.size)
            self.simulator.add_trace_event(FetchEndTraceEvent(self.env.now, self, worker, task))
            del self.running_downloads[task]
            self.data.add(task)

        assert worker != self
        self.simulator.add_trace_event(FetchStartTraceEvent(self.env.now, self, worker, task))
        process = self.env.process(_helper())
        self.running_downloads[task] = RunningDownload(process, task, self.env.now)
        return process

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

            d = self.running_downloads.get(input)
            if d is None:
                if input.info.is_finished:
                    worker = input.info.assigned_workers[0]
                    d = self._download(worker, input)
                else:
                    not_complete = True
            else:
                d = d.process
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

        self.free_cpus = self.cpus

        prepared_assignments = []
        events = [self.ready_store.get()]

        while True:
            finished = yield env.any_of(events)
            for event in finished.keys():
                if event == events[0]:
                    prepared_assignments.append(event.value)
                    prepared_assignments.sort(key=lambda a: a.priority, reverse=True)
                    events[0] = self.ready_store.get()
                    continue

                assignment = event.value
                task = assignment.task
                self.free_cpus += task.cpus
                self.assignments.remove(assignment)
                events.remove(event)
                del self.running_tasks[task]
                simulator.add_trace_event(TaskEndTraceEvent(self.env.now, self, task))
                self.data.add(task)
                simulator.on_task_finished(self, task)

            for assignment in prepared_assignments[:]:
                task = assignment.task
                if task.cpus <= self.free_cpus:
                    self.free_cpus -= task.cpus
                    self.running_tasks[task] = RunningTask(task, self.env.now)
                    simulator.add_trace_event(TaskStartTraceEvent(self.env.now, self, task))
                    events.append(env.timeout(task.duration, assignment))
                    prepared_assignments.remove(assignment)

    def __repr__(self):
        return "<Worker {}>".format(id(self))
