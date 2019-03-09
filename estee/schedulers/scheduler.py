
from ..simulator.runtimeinfo import TaskState
from .tasks import SchedulerTaskGraph, SchedulerTask, SchedulerDataObject

import logging
logger = logging.getLogger(__name__)

class SchedulerInterface:

    def send_message(self, message):
        raise NotImplementedError()

    def start(self):
        raise NotImplementedError()

    def stop(self):
        pass


class SchedulerWorker:

    __slots__ = ("worker_id", "cpus")

    def __init__(self, worker_id, cpus):
        self.worker_id = worker_id
        self.cpus = cpus


class SchedulerBase(SchedulerInterface):

    PROTOCOL_VERSION = 0

    _simulator = None  # If running in simulator, this variable is filled before calling start()
                       # Only for testing purpose, scheduler should not depends on this variable

    _disable_cleanup = False

    def __init__(self, name, version):
        self.workers = {}
        self.task_graph = SchedulerTaskGraph()
        self._name = name
        self._version = version
        self.network_bandwidth = None
        self.assignments = None


    def send_message(self, message):
        message_type = message["type"]
        if message_type == "update":
            return self._process_update(message)
        else:
            raise Exception("Unkown message type: '{}'".format(message_type))

    def start(self):
        return {
            "type": "register",
            "protocol_version": self.PROTOCOL_VERSION,
            "scheduler_name": self._name,
            "scheduler_version": self._version,
        }

    def schedule(self, ready_tasks, finished_tasks, graph_changed, cluster_changed):
        raise NotImplementedError()

    def _process_update(self, message):

        cluster_changed = False
        graph_changed = False
        ready_tasks = []
        finished_tasks = []

        if message.get("new_workers"):
            cluster_changed = True
            for w in message["new_workers"]:
                worker_id = w["id"]
                if worker_id in self.workers:
                    raise Exception(
                        "Registering already registered worker '{}'".format(worker_id))
                self.workers[worker_id] = SchedulerWorker(worker_id, w["cpus"])

        if "network_bandwidth" in message:
            bandwidth = message["network_bandwidth"]
            if bandwidth != self.network_bandwidth:
                cluster_changed = True
                self.network_bandwidth = bandwidth

        if message.get("new_objects"):
            graph_changed = True
            objects = self.task_graph.objects
            for o in message["new_objects"]:
                object_id = o["id"]
                objects[object_id] = SchedulerDataObject(object_id, o["expected_size"], o.get("size"))

        if message.get("new_tasks"):
            graph_changed = True
            tasks = self.task_graph.tasks
            objects = self.task_graph.objects
            for t in message["new_tasks"]:
                task_id = t["id"]
                inputs = [objects[o] for o in t["inputs"]]
                outputs = [objects[o] for o in t["outputs"]]
                task = SchedulerTask(
                    task_id,
                    inputs,
                    outputs,
                    t["expected_duration"],
                    t["cpus"])
                for o in outputs:
                    o.parent = task
                for o in inputs:
                    o.consumers.add(task)
                if task.unfinished_inputs == 0:
                    ready_tasks.append(task)
                tasks[task_id] = task

        task_graph = self.task_graph
        workers = self.workers

        for tu in message.get("tasks_update", ()):
            assert tu["state"] == TaskState.Finished
            task = task_graph.tasks[tu["id"]]
            task.state = TaskState.Finished
            finished_tasks.append(task)
            for o in task.outputs:
                for t in o.consumers:
                    t.unfinished_inputs -= 1
                    if t.unfinished_inputs <= 0:
                        assert t.unfinished_inputs == 0
                        ready_tasks.append(t)

        for ou in message.get("objects_update", ()):
            o = task_graph.objects[ou["id"]]
            o.placing = [workers[w] for w in ou["placing"]]
            o.availability = [workers[w] for w in ou["availability"]]
            size = ou.get("size")
            if size is not None:
                o.size = size

        self.assignments = {}
        self.schedule(ready_tasks, finished_tasks, graph_changed, cluster_changed)
        return list(self.assignments.values())

    def assign(self, worker, task, priority=None, blocking=None):
        task.state = TaskState.Assigned
        task.worker = worker

        for o in task.inputs:
            o.scheduled.add(worker)

        for o in task.outputs:
            o.scheduled.add(worker)

        result = {
            "worker": worker.worker_id,
            "task": task.id,
        }
        if priority is not None:
            result["priority"] = priority
        if blocking is not None:
            result["blocking"] = blocking
        self.assignments[task] = result

    def stop(self):
        if self._disable_cleanup:
            return
        self.workers.clear()
        self.task_graph.tasks.clear()
        self.task_graph.objects.clear()
        self.network_bandwidth = None


class StaticScheduler(SchedulerBase):

    def schedule(self, new_ready, new_finished, graph_changed, cluster_changed):
        if graph_changed or cluster_changed:
            return self.static_schedule()
        else:
            return ()

    def static_schedule(self):
        raise NotImplementedError()


class FixedScheduler(StaticScheduler):
    def __init__(self, schedules):
        super().__init__()
        self.schedules = schedules

    def static_schedule(self):
        return self.schedules


class TracingScheduler(SchedulerBase):
    def __init__(self, scheduler):
        self.scheduler = scheduler

    def init(self, simulator):
        self.schedules = []
        self.scheduler.init(simulator)

    def schedule(self, new_ready, new_finished):
        results = self.scheduler.schedule(new_ready, new_finished)
        self.schedules += results
        return results
