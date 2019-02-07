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
