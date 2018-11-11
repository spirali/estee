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
