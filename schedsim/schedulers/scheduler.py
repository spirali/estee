from ..communication import SimpleNetModel
from ..schedulers.utils import assign_expected_values
from ..simulator import Simulator, TaskAssignment


class SchedulerBase:

    def init(self, simulator):
        self.simulator = simulator

    def schedule(self, new_ready, new_finished):
        return ()


class StaticScheduler(SchedulerBase):

    def init(self, simulator):
        super().init(simulator)
        self.scheduled = False

    def schedule(self, new_ready, new_finished):
        if self.scheduled:
            return ()
        self.scheduled = True
        return self.static_schedule()

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


def make_static_scheduler(cls):
    class Static(StaticScheduler):
        def __init__(self, *args, **kwargs):
            self.scheduler = cls(*args, **kwargs)

        def init(self, simulator):
            super().init(simulator)
            self.scheduler.init(simulator)

        def static_schedule(self):
            tracer = TracingScheduler(self.scheduler)
            graph = self.simulator.task_graph.copy()
            assign_expected_values(graph)
            simulator = Simulator(graph, [w.copy() for w in self.simulator.workers],
                                  tracer,
                                  SimpleNetModel(self.simulator.netmodel.bandwidth),
                                  min_scheduling_interval=self.simulator.min_scheduling_interval,
                                  scheduling_time=self.simulator.scheduling_time)
            simulator.run()
            return [TaskAssignment(self.simulator.workers[sched.worker.id],
                                   self.simulator.task_graph.tasks[sched.task.id],
                                   sched.priority, sched.block)
                    for sched in tracer.schedules]
    return Static
