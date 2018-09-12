
import collections

TaskAssignTraceEvent = collections.namedtuple("TaskAssign", ["time", "worker", "task"])
TaskStartTraceEvent = collections.namedtuple("TaskStart", ["time", "worker", "task"])
TaskEndTraceEvent = collections.namedtuple("TaskEnd", ["time", "worker", "task"])

FetchStartTraceEvent = collections.namedtuple(
    "FetchStart", ["time", "target_worker", "source_worker", "task"])

FetchEndTraceEvent = collections.namedtuple(
    "FetchEnd", ["time", "target_worker", "source_worker", "task"])


def merge_trace_events(trace_events, start_type, end_type, merge_fn, key_fn):
    open_events = {}

    for event in trace_events:
        if isinstance(event, start_type):
            key = key_fn(event)
            assert key not in open_events
            open_events[key] = event

        if isinstance(event, end_type):
            key = key_fn(event)
            event2 = open_events[key]
            yield merge_fn(event2, event)

"""
def make_blocks(trace_events):
    blocks = []
    blocks += merge_trace_events(
        trace_events, "start", "end", lambda e1, e2: (e1.time, e2.time, e1.task.label))
    return blocks
"""

def build_trace_html(trace_events, workers, filename):
    ##print(trace_events)
    ##blocks = make_blocks(trace_events)
    ##print(blocks)

    import bokeh.plotting
    import bokeh.io
    from pandas import DataFrame

    #from bokeh.io import output_file, save
    #from bokeh.models import ColumnDataSource, LabelSet
    #from bokeh.plotting import figure


    plot = bokeh.plotting.figure(plot_width=1200, plot_height=850,
                  title='CPU schedules')


    frame = bokeh.models.ColumnDataSource(DataFrame(merge_trace_events(
            trace_events, TaskStartTraceEvent, TaskEndTraceEvent,
            lambda e1, e2: (workers.index(e1.worker), e1.time, e2.time, e1.task.label),
            lambda e: (e.task, e.worker)),
        columns=["worker", "start", "end", "label"]))

    plot.hbar(y='worker', left='start', right='end', height=0.025, source=frame, line_color="black", line_width=2)

    labels = bokeh.models.LabelSet(x='start', y='worker', text='label',
                                   x_offset=5, y_offset=5, source=frame, text_color="black", render_mode='canvas')
    plot.add_layout(labels)

    frame = bokeh.models.ColumnDataSource(DataFrame(merge_trace_events(
            trace_events, FetchStartTraceEvent, FetchEndTraceEvent,
            lambda e1, e2: (workers.index(e1.source_worker),
                            e1.time,
                            workers.index(e2.target_worker),
                            e2.time,
                            e1.task.label),
            lambda e: (e.task, e.target_worker, e.source_worker)),
        columns=["worker", "start", "worker2", "end", "label"]))

    plot.segment(y0='worker', x0='start', y1='worker2', x1='end', source=frame, line_color="black", line_width=2)
    plot.circle(y="worker2", x="end", source=frame, size=15, color="black")

    labels = bokeh.models.LabelSet(x='end', y='worker2', text='label',
                                   x_offset=-10, source=frame, text_color="black", text_align="right", render_mode='canvas')
    plot.add_layout(labels)

    plot.yaxis.axis_label = 'Worker'
    plot.xaxis.axis_label = 'Time'

    bokeh.io.output_file(filename)
    bokeh.io.save(plot)

