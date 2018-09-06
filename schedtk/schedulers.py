import random


class SchedulerBase:

    def init(self, simulator):
        self.simulator = simulator

    def schedule(self, new_ready, new_finished):
        return ()


class DoNothingScheduler(SchedulerBase):
    pass


class AllOnOneScheduler(SchedulerBase):

    def schedule(self, new_ready, new_finished):
        worker = self.simulator.workers[0]
        return [(worker, task) for task in new_ready]


class QueueScheduler(SchedulerBase):

    def __init__(self):
        self.ready = []
        self.queue = None

    def init(self, simulator):
        super().init(simulator)
        self.queue = self.make_queue()

    def make_queue(self, simulator):
        raise NotImplementedError()

    def schedule(self, new_ready, new_finished):
        self.ready += new_ready
        workers = [w for w in self.simulator.workers if not w.assigned_tasks]
        results = []
        while workers and self.ready:
            w = workers.pop()
            for t in self.queue[:]:
                if t in self.ready:
                    self.ready.remove(t)
                    self.queue.remove(t)
                    results.append((w, t))
                    break
        return results


class RandomScheduler(QueueScheduler):

    def make_queue(self):
        tasks = self.simulator.task_graph.tasks[:]
        random.shuffle(tasks)
        return tasks
