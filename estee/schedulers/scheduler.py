
import logging
import time

from estee.simulator import Simulator
from .tasks import SchedulerTaskGraph, SchedulerTask, SchedulerDataObject, TaskState

logger = logging.getLogger(__name__)


class SchedulerInterface:
    """
        Generic interface of scheduler as expected by simulator
    """

    """
    If running in a simulator, this variable is filled by simulator before calling start().
    Only for testing purposes, scheduler should not depend on this variable.
    """
    _simulator: Simulator = None

    def send_message(self, message):
        """ Send message to scheduler

        Right now only "update" message is supported:

        {
            "type": "update",
            "new_workers": [WORKER_ID, ...]  # Optional
            "network_bandwidth": FLOAT  # Optional
            "new_tasks": [TASK_DEF, ...]  # Optional
            "new_objects": [OBJECT_DEF, ...]  # Optional
            "tasks_update": [TASK_UPDATE, ...]  # Optional
            "objects_update": [OBJECT_UPDATE, ...]  # Optional
            "reassign_failed": [REASSIGN_FAILED, ...]  # Optinal
        }

        TASK_UPDATE = {
            "id": TASK_ID,
            "state": TaskState
            "worker": WORKER_ID
        }

        OBJECT_UPDATE = {
            "id": OBJECT_ID,
            "placing": [WORKER_ID, ...]
            "availability": [WORKER_ID, ...]
        }

        REASSIGN_FAILED = {
            "id": TASK_ID
            "assigned_workers": [WORKER_ID, ...]  # Ground truth from simulator
        }
        """
        raise NotImplementedError()

    def start(self):
        """ Start scheduler and register it

        Simulator call the method at the beginning of the computation.
        It should return registration message as follows:

        {
            "type": "register",
            "protocol_version": PROTOCOL_VERSION,
            "scheduler_name": SCHEDULER_NAME,
            "scheduler_version": SCHEDULER_VERSION,
            "reassigning": REASSIGNING_FLAG
        }

        REASSIGNING_FLAG has to be True if scheduler may reassign
        already scheduled tasks
        """
        raise NotImplementedError()

    def stop(self):
        """ Stop scheduler """
        pass


class SchedulerWorker:

    def __init__(self, worker_id, cpus):
        self.worker_id = worker_id
        self.cpus = cpus

        # metadata, may not be used
        self.running_tasks = set()
        self.scheduled_tasks = []

    def simple_copy(self):
        return SchedulerWorker(self.worker_id, self.cpus)

    def __repr__(self):
        return "<SW id={} cpus={}>".format(self.worker_id, self.cpus)


class Update:

    def __init__(self,
                 new_workers,
                 network_update,
                 new_objects,
                 new_tasks,
                 new_ready_tasks,
                 new_finished_tasks,
                 reassign_failed,
                 new_started_tasks):

        self.new_workers = new_workers
        self.network_update = network_update
        self.new_objects = new_objects
        self.new_tasks = new_tasks
        self.new_ready_tasks = new_ready_tasks
        self.new_finished_tasks = new_finished_tasks
        self.reassign_failed = reassign_failed
        self.new_started_tasks = new_started_tasks

    @property
    def graph_changed(self):
        return bool(self.new_objects or self.new_tasks)

    @property
    def cluster_changed(self):
        return bool(self.new_workers or self.network_update)


class SchedulerBase(SchedulerInterface):
    """
    Base class for Python implemented schedulers
    """

    PROTOCOL_VERSION = 0

    _disable_cleanup = False  # Disable clean in stop(), for testing purposes

    def __init__(self, name, version,
                 reassigning=False,
                 task_start_notification=False,
                 only_in_simulator=False):

        self.workers = {}
        self.task_graph = SchedulerTaskGraph()
        self._name = name
        self._version = version
        self.network_bandwidth = None
        self.assignments = None
        self.reassigning = reassigning
        self.task_start_notification = task_start_notification
        self.only_in_simulator = only_in_simulator

    def now(self):
        return self._simulator.env.now if self._simulator else time.time()

    def send_message(self, message):
        message_type = message["type"]
        if message_type == "update":
            return self._process_update(message)
        else:
            raise Exception("Unkown message type: '{}'".format(message_type))

    def start(self):
        if self.only_in_simulator and self._simulator is None:
            raise Exception("Scheduler '{}' can be run only in simulator".format(self._name))

        return {
            "type": "register",
            "protocol_version": self.PROTOCOL_VERSION,
            "scheduler_name": self._name,
            "scheduler_version": self._version,
            "reassigning": self.reassigning,
            "task_start_notification": self.task_start_notification,
        }

    def schedule(self, update: Update):
        raise NotImplementedError()

    def _process_update(self, message):

        task_graph = self.task_graph
        workers = self.workers

        ready_tasks = []
        finished_tasks = []
        started_tasks = []

        if "new_workers" in message:
            new_workers = []
            for w in message["new_workers"]:
                worker_id = w["id"]
                if worker_id in workers:
                    raise Exception(
                        "Registering already registered worker '{}'".format(worker_id))
                worker = SchedulerWorker(worker_id, w["cpus"])
                new_workers.append(worker)
                workers[worker_id] = worker
        else:
            new_workers = ()

        network_update = False
        if "network_bandwidth" in message:
            bandwidth = message["network_bandwidth"]
            if bandwidth != self.network_bandwidth:
                network_update = True
                self.network_bandwidth = bandwidth

        if "new_objects" in message:
            objects = self.task_graph.objects
            new_objects = []
            for o in message["new_objects"]:
                object_id = o["id"]
                obj = SchedulerDataObject(object_id, o["expected_size"], o.get("size"))
                new_objects.append(obj)
                objects[object_id] = obj
        else:
            new_objects = ()

        if "new_tasks" in message:
            tasks = self.task_graph.tasks
            objects = self.task_graph.objects
            new_tasks = []
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
                new_tasks.append(task)
                for o in outputs:
                    o.parent = task
                for o in inputs:
                    o.consumers.add(task)
                if task.unfinished_inputs == 0:
                    ready_tasks.append(task)
                tasks[task_id] = task
        else:
            new_tasks = ()

        reassign_failed = ()
        if "reassign_failed" in message:
            reassign_failed = []
            for tu in message["reassign_failed"]:
                task = self.task_graph.tasks[tu["id"]]
                ws = [self.workers[w] for w in tu["assigned_workers"]]
                task.scheduled_worker = ws[0]
                reassign_failed.append(task)
                self._fix_implied_schedule(task)

        for tu in message.get("tasks_update", ()):
            state = tu["state"]
            assert state == TaskState.Finished or state == TaskState.Assigned
            task = task_graph.tasks[tu["id"]]
            task.state = state
            task.computed_by = workers[tu["worker"]]
            was_running = task.running
            running = bool(tu["running"])
            task.running = running

            if state == TaskState.Finished:
                finished_tasks.append(task)
                for o in task.outputs:
                    for t in o.consumers:
                        t.unfinished_inputs -= 1
                        if t.unfinished_inputs <= 0:
                            assert t.unfinished_inputs == 0
                            ready_tasks.append(t)

            if not was_running and running:
                task.start_time = self.now()
                started_tasks.append(task)

        for ou in message.get("objects_update", ()):
            o = task_graph.objects[ou["id"]]
            o.placing = [workers[w] for w in ou["placing"]]
            o.availability = [workers[w] for w in ou["availability"]]
            size = ou.get("size")
            if size is not None:
                o.size = size

        self.assignments = {}
        self.schedule(Update(
            new_workers,
            network_update,
            new_objects,
            new_tasks,
            ready_tasks,
            finished_tasks,
            reassign_failed,
            started_tasks))

        return list(self.assignments.values())

    def _fix_implied_schedule_of_object(self, obj):
        s = set()
        if obj.parent.scheduled_worker:
            s.add(obj.parent.scheduled_worker)
        for c in obj.consumers:
            if c.scheduled_worker:
                s.add(c.scheduled_worker)
        obj.scheduled = s

    def _fix_implied_schedule(self, task):
        for obj in task.inputs:
            self._fix_implied_schedule_of_object(obj)
        for obj in task.outputs:
            self._fix_implied_schedule_of_object(obj)

    def assign(self, worker: SchedulerWorker, task: SchedulerTask, priority=None, blocking=None):
        """
            Assign a task to a worker

            This method can be called from "schedule" method
            This method can be called multiple times, the last
            call is accepted.
        """
        task.state = TaskState.Assigned
        task.scheduled_worker = worker

        for o in task.inputs:
            o.scheduled.add(worker)

        for o in task.outputs:
            o.scheduled.add(worker)

        result = {
            "worker": worker.worker_id if worker else None,
            "task": task.id,
        }

        if priority is not None:
            result["priority"] = priority
        if blocking is not None:
            result["blocking"] = blocking

        if task in self.assignments:
            existing_worker = self.assignments[task]["worker"]
            if existing_worker is not None and existing_worker != worker.worker_id:
                self.workers[existing_worker].scheduled_tasks.remove(task)
            self._fix_implied_schedule(task)

        if worker:
            worker.scheduled_tasks.append(task)

        self.assignments[task] = result

    def stop(self):
        if self._disable_cleanup:
            return
        self.workers.clear()
        self.task_graph.tasks.clear()
        self.task_graph.objects.clear()
        self.network_bandwidth = None


class StaticScheduler(SchedulerBase):

    """ Base class for static schedulers

        method `static_schedule()` is invoked when cluster or task graph
        is changed
    """

    def schedule(self, update):
        if update.graph_changed or update.cluster_changed:
            return self.static_schedule()

    def static_schedule(self):
        """
        Create a static schedule

        It has to assign (via .assign) a worker for each task
        """
        raise NotImplementedError()
