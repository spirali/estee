class ExactImode():
    def process(self, graph):
        for t in graph.tasks:
            t.expected_duration = t.duration

        for o in graph.outputs:
            o.expected_size = o.size


class BlindImode():
    def process(self, graph):
        for t in graph.tasks:
            t.expected_duration = None

        for o in graph.outputs:
            o.expected_size = None
