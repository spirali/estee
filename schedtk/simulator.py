
from enum import Enum
from simpy import Environment, Event
from .trace import TaskAssignTraceEvent, build_trace_html


class TaskAssignment:

    def __init__(self, worker, task, priority=0):
        self.worker = worker
        self.task = task
        self.priority = priority


class TaskState(Enum):
    Waiting = 1
    Ready = 2
    Assigned = 3
    Finished = 4


class TaskRuntimeInfo:

    __slots__ = ("state",
                 "assign_time",
                 "end_time",
                 "assigned_workers",
                 "unfinished_inputs")

    def __init__(self, task):
        self.state = TaskState.Waiting
        self.assign_time = None
        self.end_time = None
        self.assigned_workers = []
        self.unfinished_inputs = len(task.inputs)

    @property
    def is_running(self):
        return self.state == TaskState.Running

    @property
    def is_ready(self):
        return self.state == TaskState.Ready

    @property
    def is_finished(self):
        return self.state == TaskState.Finished

    @property
    def is_waiting(self):
        return self.state == TaskState.Waiting


class OutputRuntimeInfo:

    __slots__ = ("placing")

    def __init__(self, output):
        self.placing = []


class Simulator:

    def __init__(self, task_graph, workers, scheduler, connector, trace=False):
        self.workers = workers
        self.task_graph = task_graph
        self.connector = connector
        self.scheduler = scheduler
        scheduler.simulator = self
        self.new_finished = []
        self.new_ready = []
        self.wakeup_event = None
        self.env = None
        if trace:
            self.trace_events = []
        else:
            self.trace_events = None

        for i, worker in enumerate(workers):
            assert worker.id is None
            worker.id = i

    def task_info(self, task):
        return self.task_infos[task.id]

    def output_info(self, output):
        return self.output_infos[output.id]

    def add_trace_event(self, trace_event):
        if self.trace_events is not None:
            self.trace_events.append(trace_event)

    def schedule(self, ready_tasks, finished_tasks):
        for assignment in self.scheduler.schedule(ready_tasks, finished_tasks):
            assert isinstance(assignment, TaskAssignment)
            info = self.task_info(assignment.task)
            if info.state == TaskState.Finished:
                raise Exception("Scheduler tries to assign a finished task ({})"
                                .format(assignment.task))
            if info.state == TaskState.Assigned:
                raise Exception("Scheduler reassigns already assigned task ({})"
                                .format(assignment.task))
            info.state = TaskState.Assigned
            info.assigned_workers.append(assignment.worker)
            assignment.worker.assign_task(assignment)
            self.add_trace_event(TaskAssignTraceEvent(
                self.env.now, assignment.worker, assignment.task))

    def _master_process(self, env):
        self.schedule(self.task_graph.source_tasks(), [])

        while self.unprocessed_tasks > 0:
            self.wakeup_event = Event(env)
            yield self.wakeup_event
            self.schedule(self.new_ready, self.new_finished)
            self.new_finished = []
            self.new_ready = []

    def on_task_finished(self, worker, task):
        info = self.task_info(task)
        assert info.state == TaskState.Assigned
        assert worker in info.assigned_workers
        info.state = TaskState.Finished
        self.new_finished.append(task)
        self.unprocessed_tasks -= 1

        for o in task.outputs:
            self.output_info(o).placing.append(worker)
            tasks = sorted(o.consumers, key=lambda t: t.id)
            for t in tasks:
                t_info = self.task_info(t)
                t_info.unfinished_inputs -= 1
                if t_info.unfinished_inputs <= 0:
                    if t_info.unfinished_inputs < 0:
                        raise Exception("Invalid number of unfinished inputs: {}, task {}".format(
                            t_info.unfinished_inputs, t
                        ))
                    assert t_info.unfinished_inputs == 0
                    if t_info.state == TaskState.Waiting:
                        t_info.state = TaskState.Ready
                    self.new_ready.append(t)

            for t in tasks:
                for w in self.task_info(t).assigned_workers:
                    w.update_task(t)

        if not self.wakeup_event.triggered:
            self.wakeup_event.succeed()

    def make_trace_report(self, filename):
        build_trace_html(self.trace_events or [], self.workers, filename)

    def run(self):
        assert not self.trace_events

        self.task_infos = [TaskRuntimeInfo(task) for task in self.task_graph.tasks]
        self.output_infos = [OutputRuntimeInfo(task) for task in self.task_graph.outputs]

        self.unprocessed_tasks = self.task_graph.task_count

        env = Environment()
        self.env = env
        self.connector.init(self.env, self.workers)

        for worker in self.workers:
            env.process(worker.run(env, self, self.connector))

        master_process = env.process(self._master_process(env))
        self.scheduler.init(self)

        env.run(master_process)
        return env.now
