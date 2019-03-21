import logging

from simpy import Event, Store

from ..simulator.trace import FetchStartTraceEvent, \
    TaskEndTraceEvent, TaskStartTraceEvent

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

    __slots__ = ("output", "source", "start_time", "priority", "consumer_count")

    def __init__(self, output, priority):
        self.output = output
        self.start_time = None
        self.source = None
        self.priority = priority
        self.consumer_count = 0

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

    DOWNLOAD_PRIORITY_BOOST_FOR_READY_TASK = 100000

    def __init__(self, cpus=1, max_downloads=4, max_downloads_per_worker=2):
        self.cpus = cpus
        self.assignments = {}
        self.ready_store = None

        self.data = set()
        self.running_tasks = {}
        self.scheduled_downloads = {}
        self.running_downloads = []

        self.free_cpus = cpus
        self.max_downloads = max_downloads
        self.max_downloads_per_worker = max_downloads_per_worker
        self.id = None

    def to_dict(self):
        return {
            "id": self.id,
            "cpus": self.cpus
        }

    def copy(self):
        return Worker(cpus=self.cpus,
                      max_downloads=self.max_downloads,
                      max_downloads_per_worker=self.max_downloads_per_worker)

    def try_retract_task(self, task):
        if task in self.running_tasks:
            logging.debug("Retracting task %s from worker %s cancelled because task is running",
                          task, self)
            return False

        logging.debug("Retracting task %s from worker %s", task, self)
        a = self.assignments[task]
        a.cancelled = True

        for inp in task.inputs:
            d = self.scheduled_downloads.get(inp)
            if d is None:
                continue
            d.consumer_count -= 1
            if d.consumer_count <= 0:
                logging.debug("Cancelling download of %s", inp)
                assert d.consumer_count == 0
                if d.source is None:  # is not running
                    assert d not in self.running_downloads
                    del self.scheduled_downloads[inp]

                # This is necessary to cleanup cache
                if not self.download_wakeup.triggered:
                    self.download_wakeup.succeed()

        del self.assignments[a.task]
        return True

    def assign_tasks(self, assignments):
        runtime_state = self.simulator.runtime_state
        for assignment in assignments:
            assert assignment.worker == self
            # assert assignment not in self.assignments

            if assignment.task.cpus > self.cpus:
                raise Exception("Task {} allocated on worker with {} cpus"
                                .format(assignment.task, self.cpus))

            self.assignments[assignment.task] = assignment
            need_inputs = 0
            for inp in assignment.task.inputs:
                if inp in self.data:
                    continue
                if runtime_state.object_info(inp).placing:
                    self._schedule_download(assignment, inp,
                                            runtime_state.task_info(assignment.task).is_ready)
                need_inputs += 1
            assignment.remaining_inputs_count = need_inputs
            if need_inputs == 0:
                self.ready_store.put(assignment)
            logger.info("Task %s scheduled on %s (%s ri)", assignment.task, self, need_inputs)

    def update_tasks(self, updates):
        runtime_state = self.simulator.runtime_state
        for task, obj in updates:
            a = self.assignments[task]
            if obj not in self.data:
                if obj.size == 0:
                    self._add_data(obj)
                else:
                    self._schedule_download(a, obj, runtime_state.task_info(task).is_ready)

    @property
    def assigned_tasks(self):
        return iter(self.assignments)

    def _add_data(self, obj):
        if obj in self.data:
            raise Exception("Object {} is already on worker {}".format(obj, self))
        self.data.add(obj)
        for t in obj.consumers:
            a = self.assignments.get(t)
            if a is None:
                continue
            a.remaining_inputs_count -= 1
            if a.remaining_inputs_count <= 0:
                assert a.remaining_inputs_count == 0
                self.ready_store.put(a)

    def _schedule_download(self, assignment, obj, ready):
        priority = assignment.priority
        if ready:
            priority += self.DOWNLOAD_PRIORITY_BOOST_FOR_READY_TASK
        d = self.scheduled_downloads.get(obj)
        if d is None:
            logger.info("Worker %s: scheduled downloading %s, priority=%s", self, obj, priority)
            assert obj not in self.data
            d = Download(obj, priority)
            self.scheduled_downloads[obj] = d
        else:
            d.update_priority(priority)
        d.consumer_count += 1
        if not self.download_wakeup.triggered:
            self.download_wakeup.succeed()

    def _download_process(self):
        events = [self.download_wakeup]
        env = self.env
        runtime_state = self.simulator.runtime_state

        while True:
            finished = yield env.any_of(events)
            for event in finished.keys():
                if event == events[0]:
                    self.download_wakeup = Event(self.simulator.env)
                    events[0] = self.download_wakeup
                    downloads = None
                    continue
                events.remove(event)
                download = event.value
                self._add_data(download.output)
                self.running_downloads.remove(download)
                del self.scheduled_downloads[download.output]

                self.simulator.fetch_finished(self, download.source, download.output)

            if len(self.running_downloads) < self.max_downloads:
                # We need to sort any time, as it priority may changed in background

                if downloads is None:
                    downloads = list(o for o in self.scheduled_downloads.values()
                                     if o not in self.running_downloads)
                    downloads.sort(key=lambda d: d.priority, reverse=True)

                for d in downloads[:]:
                    count = 0
                    worker = runtime_state.object_info(d.output).placing[0]
                    for rd in self.running_downloads:
                        if worker == rd.source:
                            count += 1
                    if count >= self.max_downloads_per_worker:
                        continue
                    downloads.remove(d)
                    assert d.start_time is None
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
                    events[0] = self.ready_store.get()
                    assignment = event.value
                    if assignment.cancelled:
                        continue
                    prepared_assignments.append(assignment)
                    prepared_assignments.sort(key=lambda a: a.priority, reverse=True)
                    continue

                assignment = event.value
                task = assignment.task
                self.free_cpus += task.cpus
                assert not assignment.cancelled
                del self.assignments[assignment.task]
                events.remove(event)
                del self.running_tasks[task]
                simulator.add_trace_event(TaskEndTraceEvent(self.env.now, self, task))
                for output in task.outputs:
                    self._add_data(output)
                simulator.on_task_finished(self, task)

            block = float("-inf")
            for assignment in prepared_assignments[:]:
                if assignment.priority < block:
                    continue
                task = assignment.task
                if task.cpus <= self.free_cpus:
                    prepared_assignments.remove(assignment)
                    if assignment.cancelled:
                        continue
                    self.free_cpus -= task.cpus
                    self.running_tasks[task] = RunningTask(task, self.env.now)
                    simulator.add_trace_event(TaskStartTraceEvent(self.env.now, self, task))
                    events.append(env.timeout(task.duration, assignment))
                    self.simulator.on_task_start(self, assignment.task)
                else:
                    block = max(block, assignment.block)

    def __repr__(self):
        return "<Worker {}>".format(self.id)
