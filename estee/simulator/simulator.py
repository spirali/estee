
from simpy import Environment, Event

from .commands import TaskAssignment
from .runtimeinfo import RuntimeState, TaskState
from .trace import TaskAssignTraceEvent

import logging

logger = logging.getLogger(__name__)


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
        self.new_finished = []
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

        self.tasks_updated = set()
        self.objects_updated = set()
        self.new_workers = []
        self.new_tasks = []
        self.new_objects = []
        self.update_bandwidth = True

    def add_trace_event(self, trace_event):
        if self.trace_events is not None:
            self.trace_events.append(trace_event)

    def read_assignment(self, obj):
        task = self.task_graph.tasks[obj["task"]]
        worker = self.workers[obj["worker"]]
        priority = obj.get("priority", 0)
        blocking = obj.get("blocking", 0)
        return TaskAssignment(worker, task, priority, blocking)

    def apply_schedule(self, schedule):
        worker_loads = {}
        assignments = []
        for obj in schedule:
            # TODO: Filter invalid assignemnts
            assignments.append(self.read_assignment(obj))
        assignments.sort(key=lambda a: a.priority, reverse=True)
        for assignment in assignments:
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

    def send_update(self):
        runtime_state = self.runtime_state

        def make_task_update(task):
            info = runtime_state.task_info(task)
            return {
                "id": task.id,
                "state": info.state,
            }

        def make_object_update(obj):
            info = runtime_state.object_info(obj)
            placing = info.placing
            result = {
              "id": obj.id,
              "placing": [w.id for w in placing]
            }
            if info.placing or runtime_state.task_info(obj.parent).state == TaskState.Finished:
                result["size"] = obj.size
            return result

        message = {
            "type": "update",
            "tasks_update": [make_task_update(t) for t in self.tasks_updated],
            "objects_update": [make_object_update(o) for o in self.objects_updated]
        }

        self.objects_updated.clear()
        self.tasks_updated.clear()

        if self.new_workers:
            message["new_workers"] = [worker.to_dict() for worker in self.new_workers]
            self.new_workers = []

        if self.update_bandwidth:
            message["network_bandwidth"] = self.netmodel.bandwidth
            self.update_bandwidth = False

        if self.new_tasks:
            message["new_tasks"] = [o.to_dict() for o in self.new_tasks]
            self.new_tasks = []

        if self.new_objects:
            message["new_objects"] = [t.to_dict() for t in self.new_objects]
            self.new_objects = []

        logger.debug("Sending update %s", message)
        schedule = self.scheduler.send_message(message)
        logger.debug("Scheduler result %s", schedule)
        return schedule

    def _master_process(self, env):
        min_scheduling_interval = self.min_scheduling_interval
        scheduling_time = self.scheduling_time
        timeout = self.env.timeout

        # We are here intentionally separate registering workers
        # and task submit, as it is usually separated in real-word
        # reactors.

        self.new_workers += self.workers
        schedule = self.send_update()
        assert not schedule

        self.new_tasks += list(self.task_graph.tasks.values())
        self.new_objects += list(self.task_graph.objects.values())

        schedule = self.send_update()
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

            schedule = self.send_update()
            if scheduling_time:
                yield timeout(scheduling_time)
            if schedule:
                self.apply_schedule(schedule)

    def on_task_finished(self, worker, task):
        logger.debug("Task %s finished on %s", task, worker)
        runtime_state = self.runtime_state
        info = runtime_state.task_info(task)
        assert info.state == TaskState.Assigned
        assert worker in info.assigned_workers
        info.state = TaskState.Finished
        info.end_time = self.env.now
        self.new_finished.append(task)
        self.unprocessed_tasks -= 1

        worker_updates = {}

        self.tasks_updated.add(task)
        objects_updated = self.objects_updated

        for o in task.outputs:
            runtime_state.object_info(o).placing.append(worker)
            objects_updated.add(o)
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

    def start_scheduler(self):
        self.scheduler._simulator = self
        message = self.scheduler.start()
        if message.get("type") != "register":
            raise Exception("Invalid registeration message from scheduler")
        logger.info("Scheduler '%s', version '%s'",
                    message.get("scheduler_name"),
                    message.get("scheduler_version"))

    def stop_scheduler(self):
        self.scheduler.stop()
        self.scheduler._simulator = None
        logger.info("Scheduler stopped")

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

        self.start_scheduler()
        env.run(master_process)
        self.stop_scheduler()
        return env.now