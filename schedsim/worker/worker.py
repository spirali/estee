from simpy import Store, Event

from ..simulator.trace import FetchEndTraceEvent, FetchStartTraceEvent, \
    TaskEndTraceEvent, TaskStartTraceEvent

import logging


logger = logging.getLogger(__name__)


class RunningTask:

    __slots__ = ("task", "start_time")

    def __init__(self, task, start_time):
        self.task = task
        self.start_time = start_time

    def running_time(self, now):
        return now - self.start_time

    def remaining_time(self, now):
        return self.task.duration - self.running_time(now)


class Download:

    __slots__ = ("event", "output", "source", "start_time", "priority")

    def __init__(self, event, output, priority):
        self.event = event
        self.output = output
        self.start_time = None
        self.source = None
        self.priority = priority

    def update_priority(self, priority):
        self.priority = max(self.priority, priority)

    def running_time(self, now):
        if self.start_time is None:
            return None
        return now - self.start_time

    def naive_remaining_time_estimate(self, simulator):
        if self.start_time is None:
            return None
        return (self.output.size / simulator.netmodel.bandwidth +
                self.start_time - simulator.env.now)


class Worker:

    def __init__(self, cpus=1, max_downloads=4, max_downloads_per_worker=2):
        self.cpus = cpus
        self.assignments = []
        self.ready_store = None

        self.data = set()
        self.running_tasks = {}
        self.scheduled_downloads = {}
        self.running_downloads = []

        self.free_cpus = cpus
        self.max_downloads = max_downloads
        self.max_downloads_per_worker = max_downloads_per_worker
        self.id = None

    def _download(self, output, priority):
        logger.info("Worker %s: scheduled downloading %s, priority=%s", self, output, priority)
        assert output not in self.data
        download = Download(Event(self.env), output, priority)
        self.scheduled_downloads[output] = download
        return download

    def assign_tasks(self, assignments):
        for assignment in assignments:
            if assignment.task.cpus > self.cpus:
                raise Exception("Task {} allocated on worker with {} cpus"
                                .format(assignment.task, self.cpus))
            assert assignment.worker == self
            self.assignments.append(assignment)
            self._init_downloads(assignment)

        if not self.download_wakeup.triggered:
            self.download_wakeup.succeed()

    def update_tasks(self, tasks):
        for task in tasks:
            for assignment in self.assignments:
                if task == assignment.task:
                    self._init_downloads(assignment)
                    break
            else:
                raise Exception("Updating non assigned task {}, worker={}".format(task, self))
        if not self.download_wakeup.triggered:
            self.download_wakeup.succeed()

    @property
    def assigned_tasks(self):
        return [a.task for a in self.assignments]

    def _init_downloads(self, assignment):
        deps = []
        not_complete = False
        for input in assignment.task.inputs:
            if input in self.data:
                continue

            d = self.scheduled_downloads.get(input)
            if d is None:
                info = self.simulator.output_info(input)
                if info.placing:
                    if input.size == 0:
                        self.data.add(input)
                        continue
                    d = self._download(input, assignment.priority)
                    deps.append(d.event)
                else:
                    not_complete = True
            else:
                d.update_priority(assignment.priority)
                deps.append(d.event)

        def _helper():
            yield self.env.all_of(deps)
            self.ready_store.put(assignment)

        if not_complete:
            return
        if not deps:
            self.ready_store.put(assignment)
        else:
            self.env.process(_helper())

    def _download_process(self):
        events = [self.download_wakeup]
        env = self.env

        while True:
            finished = yield env.any_of(events)
            for event in finished.keys():
                if event == events[0]:
                    self.download_wakeup = Event(self.simulator.env)
                    events[0] = self.download_wakeup
                    continue
                events.remove(event)
                download = event.value
                assert download.output not in self.data
                self.data.add(download.output)
                self.running_downloads.remove(download)
                del self.scheduled_downloads[download.output]
                download.event.succeed(download)
                self.simulator.add_trace_event(
                    FetchEndTraceEvent(self.env.now, self, download.source, download.output))

            if len(self.running_downloads) < self.max_downloads:
                # We need to sort any time, as it priority may changed in background
                downloads = list(o for o in self.scheduled_downloads.values()
                                 if o not in self.running_downloads)
                downloads.sort(key=lambda d: d.priority, reverse=True)
                for d in downloads[:]:
                    count = 0
                    worker = self.simulator.output_info(d.output).placing[0]
                    for rd in self.running_downloads:
                        if worker == rd.source:
                            count += 1
                    if count >= self.max_downloads_per_worker:
                        continue
                    d.start_time = self.env.now
                    d.source = worker
                    self.running_downloads.append(d)
                    event = self.netmodel.download(worker, self, d.output.size, d)
                    events.append(event)
                    self.simulator.add_trace_event(
                        FetchStartTraceEvent(self.env.now, self, worker, d.output))
                    if len(self.running_downloads) >= self.max_downloads:
                        break

    def run(self, env, simulator, netmodel):
        self.env = env
        self.simulator = simulator
        self.netmodel = netmodel
        self.ready_store = Store(env)
        self.download_wakeup = Event(self.simulator.env)

        self.free_cpus = self.cpus
        env.process(self._download_process())

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
                for output in task.outputs:
                    self.data.add(output)
                simulator.on_task_finished(self, task)

            block = float("-inf")
            for assignment in prepared_assignments[:]:
                if assignment.priority < block:
                    continue
                task = assignment.task
                if task.cpus <= self.free_cpus:
                    self.free_cpus -= task.cpus
                    self.running_tasks[task] = RunningTask(task, self.env.now)
                    simulator.add_trace_event(TaskStartTraceEvent(self.env.now, self, task))
                    events.append(env.timeout(task.duration, assignment))
                    prepared_assignments.remove(assignment)
                else:
                    block = max(block, assignment.block)

    def __repr__(self):
        return "<Worker {}>".format(self.id)
