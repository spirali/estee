
from .task import Task, DataObject
from .taskbase import TaskGraphBase


class TaskGraph(TaskGraphBase):

    def _copy_tasks_and_objects(self):
        self.validate()
        tasks = [task.simple_copy() for task in self.tasks.values()]
        objects = {o.id: o for t in tasks for o in t.outputs}

        for o in self.objects.values():
            if o.parent is not None:
                continue
            assert o.id not in objects
            objects[o.id] = DataObject(o.id, o.size, o.expected_size)

        for old_task, task in zip(self.tasks.values(), tasks):
            for inp in old_task.inputs:
                task.add_input(objects[inp.id])
        return (tasks, objects)

    def copy(self):
        tasks, objects = self._copy_tasks_and_objects()
        return TaskGraph({t.id: t for t in tasks}, objects)

    def new_task(self,
                 name=None,
                 outputs=(),
                 duration=1,
                 expected_duration=None,
                 cpus=1,
                 output_size=None):
        task_id = len(self.tasks)
        task = Task(task_id, name, outputs, duration, cpus, output_size, expected_duration)
        self.tasks[task_id] = task

        output_id = len(self.objects)
        for o in task.outputs:
            o.id = output_id
            self.objects[output_id] = o
            output_id += 1
        return task

    @staticmethod
    def merge(task_graphs):
        tasks = []
        objects = []
        for g in task_graphs:
            t, o = g._copy_tasks_and_objects()
            tasks += t
            objects += o.values()

        ts = {}
        for (i, t) in enumerate(tasks):
            t.id = i
            ts[i] = t

        os = {}
        for (i, o) in enumerate(objects):
            o.id = i
            os[i] = o

        return TaskGraph(ts, os)
