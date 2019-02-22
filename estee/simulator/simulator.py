
from simpy import Environment, Event

from .commands import TaskAssignment
from .runtimeinfo import RuntimeState, TaskState
from .trace import TaskAssignTraceEvent


class Simulator:

    def __init__(self,
                 task_graph,
                 workers,
                 scheduler,
                 netmodel,
                 min_scheduling_interval=None,
                 scheduling_time=None,
                 trace=False):
        self.workers = workers
        self.task_graph = task_graph
        self.netmodel = netmodel
        self.scheduler = scheduler
        scheduler.simulator = self
        self.new_finished = []
        self.new_ready = []
        self.wakeup_event = None
        self.env = None
        self.min_scheduling_interval = min_scheduling_interval
        self.scheduling_time = scheduling_time

        if trace:
            self.trace_events = []
            netmodel.set_event_listener(lambda e: self.trace_events.append(e))
        else:
            self.trace_events = None

        for i, worker in enumerate(workers):
            assert worker.id is None
            worker.id = i

    """
        def task_info(self, task):
            return self.task_infos[task.id]

        def output_info(self, output):
            return self.output_infos[output.id]
    """

    def add_trace_event(self, trace_event):
        if self.trace_events is not None:
            self.trace_events.append(trace_event)

    def schedule(self, ready_tasks, finished_tasks):
        schedule = self.scheduler.schedule(ready_tasks, finished_tasks)
        if not schedule:
            return None
        schedule.sort(key=lambda a: a.priority, reverse=True)
        return schedule

    def apply_schedule(self, schedule):
        worker_loads = {}
        for assignment in schedule:
            assert isinstance(assignment, TaskAssignment)
            info = self.runtime_state.task_info(assignment.task)
            if info.state == TaskState.Finished:
                raise Exception("Scheduler tries to assign a finished task ({})"
                                .format(assignment.task))
            if info.state == TaskState.Assigned:
                raise Exception("Scheduler reassigns already assigned task ({})"
                                .format(assignment.task))
            info.state = TaskState.Assigned
            info.assigned_workers.append(assignment.worker)
            worker = assignment.worker
            lst = worker_loads.get(worker)
            if lst is None:
                lst = []
                worker_loads[worker] = lst
            lst.append(assignment)
            self.add_trace_event(TaskAssignTraceEvent(
                self.env.now, assignment.worker, assignment.task))
        for worker in worker_loads:
            worker.assign_tasks(worker_loads[worker])

    def _master_process(self, env):
        timeout = self.env.timeout
        min_scheduling_interval = self.min_scheduling_interval
        scheduling_time = self.scheduling_time

        schedule = self.schedule(self.task_graph.source_tasks(), [])
        if scheduling_time:
            yield timeout(scheduling_time)
        if schedule:
            self.apply_schedule(schedule)

        while self.unprocessed_tasks > 0:
            self.wakeup_event = Event(env)
            if min_scheduling_interval:
                yield self.wakeup_event & timeout(min_scheduling_interval)
            else:
                yield self.wakeup_event
            schedule = self.schedule(self.new_ready, self.new_finished)
            self.new_finished = []
            self.new_ready = []
            if scheduling_time:
                yield timeout(scheduling_time)
            if schedule:
                self.apply_schedule(schedule)

    def on_task_finished(self, worker, task):
        runtime_state = self.runtime_state
        info = runtime_state.task_info(task)
        assert info.state == TaskState.Assigned
        assert worker in info.assigned_workers
        info.state = TaskState.Finished
        info.end_time = self.env.now
        self.new_finished.append(task)
        self.unprocessed_tasks -= 1

        worker_updates = {}
        for o in task.outputs:
            runtime_state.output_info(o).placing.append(worker)
            tasks = sorted(o.consumers, key=lambda t: t.id)
            for t in tasks:
                t_info = runtime_state.task_info(t)
                t_info.unfinished_inputs -= 1
                if t_info.unfinished_inputs <= 0:
                    if t_info.unfinished_inputs < 0:
                        raise Exception("Invalid number of unfinished inputs: {}, task {}".format(
                            t_info.unfinished_inputs, t
                        ))
                    assert t_info.unfinished_inputs == 0
                    self.new_ready.append(t)

            for t in tasks:
                for w in runtime_state.task_info(t).assigned_workers:
                    task_set = worker_updates.get(w)
                    if task_set is None:
                        task_set = set()
                        worker_updates[w] = task_set
                    task_set.add(t)

        for w in worker_updates:
            w.update_tasks(worker_updates[w])
        if not self.wakeup_event.triggered:
            self.wakeup_event.succeed()

    def run(self):
        assert not self.trace_events

        self.runtime_state = RuntimeState(self.task_graph)
        self.unprocessed_tasks = self.task_graph.task_count

        env = Environment()
        self.env = env
        self.netmodel.init(self.env, self.workers)

        for worker in self.workers:
            env.process(worker.run(env, self, self.netmodel))

        master_process = env.process(self._master_process(env))
        self.scheduler.init(self)

        env.run(master_process)
        return env.now
