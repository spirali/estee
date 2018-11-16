import collections
import math

TaskAssignTraceEvent = collections.namedtuple("TaskAssign", ["time", "worker", "task"])
TaskStartTraceEvent = collections.namedtuple("TaskStart", ["time", "worker", "task"])
TaskEndTraceEvent = collections.namedtuple("TaskEnd", ["time", "worker", "task"])

FetchStartTraceEvent = collections.namedtuple(
    "FetchStart", ["time", "target_worker", "source_worker", "output"])

FetchEndTraceEvent = collections.namedtuple(
    "FetchEnd", ["time", "target_worker", "source_worker", "output"])

BandwidthChangeEvent = collections.namedtuple("BandwidthChange",
                                              ["time", "out", "worker", "value"])


def merge_trace_events(trace_events, start_pred, end_pred, key_fn, start_map=None, end_map=None):
    """
    Produces a stream of matched starts and ends of events in the form of
    merge_fn(start_event, end_event).
    """
    open_events = {}

    if not start_map:
        def start_map(e): return e
    if not end_map:
        def end_map(e): return e

    for event in trace_events:
        if start_pred(event):
            key = key_fn(event)
            assert key not in open_events
            open_events[key] = start_map(event)
        elif end_pred(event):
            key = key_fn(event)
            start_event = open_events[key]
            yield end_map(start_event, event)


def build_task_locations(trace_events, worker):
    slots = []

    def find_slot(height):
        h = 0
        for (index, (start, end)) in enumerate(slots):
            if h + height <= start:
                slots.insert(index, (h, h + height))
                return (h, h + height)
            h = end
        last = slots[-1][1] if slots else 0
        slots.append((last, last + height))
        return (last, last + height)

    def map_start(event):
        (start, end) = find_slot(event.task.cpus)
        return (event, (start, end))

    def map_end(start_event, end_event):
        event, slot = start_event
        slots.remove(slot)
        return (event.task,
                (event.time, event.time + event.task.duration, slot[0], slot[1]))

    yield from merge_trace_events(
        trace_events,
        lambda t: isinstance(t, TaskStartTraceEvent) and t.worker == worker,
        lambda t: isinstance(t, TaskEndTraceEvent) and t.worker == worker,
        lambda e: e.task,
        map_start,
        map_end
    )


def build_worker_usage(trace_events, worker):
    rectangles = []
    start_time = 0
    height = 0

    def map_start(event):
        nonlocal height, start_time

        if height != 0 and start_time != event.time:
            rectangles.append((start_time, event.time, 0, height))
        start_time = event.time
        height += event.task.cpus
        return event

    def map_end(start_event, end_event):
        nonlocal height, start_time
        if height != 0 and start_time != end_event.time:
            rectangles.append((start_time, end_event.time, 0, height))
        start_time = end_event.time
        height -= end_event.task.cpus

    list(merge_trace_events(
        trace_events,
        lambda t: isinstance(t, TaskStartTraceEvent) and t.worker == worker,
        lambda t: isinstance(t, TaskEndTraceEvent) and t.worker == worker,
        lambda e: e.task,
        map_start,
        map_end
    ))

    return rectangles


def build_worker_transfer_size(trace_events, worker):
    transfers = (  # in, out
        [(0, 0)],
        [(0, 0)]
    )

    def map_end(start_event, end_event):
        bw_index = 0 if start_event.target_worker == worker else 1
        transfers[bw_index].append((end_event.time,
                                    transfers[bw_index][-1][1] + start_event.output.size))

    def check_worker(event):
        return worker in (event.target_worker, event.source_worker)

    list(merge_trace_events(
        trace_events,
        lambda t: isinstance(t, FetchStartTraceEvent) and check_worker(t),
        lambda t: isinstance(t, FetchEndTraceEvent) and check_worker(t),
        lambda e: (e.output, e.target_worker, e.source_worker),
        end_map=map_end
    ))

    return transfers


def plot_task_communication(trace_events, workers, show_communication=False):
    """
    Plots individual tasks on workers into a grid chart (one chart per worker).

    :param show_communication: Merge all worker charts into one and plot communication edges.
    """
    from bokeh import models, plotting
    from bokeh.layouts import gridplot
    from pandas import DataFrame

    end_time = math.ceil(max([e.time for e in trace_events]))

    plots = []
    if show_communication:
        plot = plotting.figure(plot_width=1200, plot_height=850,
                               x_range=(0, end_time),
                               title='CPU schedules')
        plot.yaxis.axis_label = 'Worker'
        plot.xaxis.axis_label = 'Time'
        plots.append(plot)

    def get_worker_plot():
        if show_communication:
            return plot
        else:
            p = plotting.figure(plot_width=600, plot_height=300,
                                x_range=(0, end_time),
                                title='Worker task execution')
            p.yaxis.axis_label = 'Task'
            p.xaxis.axis_label = 'Time'
            plots.append(p)
            return p

    task_to_loc = {}

    # render task rectangles
    for index, worker in enumerate(workers):
        locations = list(build_task_locations(trace_events, worker))

        def normalize_height(height):
            if show_communication:
                return height / (worker.cpus * 2) + index
            return height

        worker_plot = get_worker_plot()
        rectangles = [(
            rect[0],
            rect[1],
            normalize_height(rect[2]),
            normalize_height(rect[3]))
            for (task, rect) in locations
        ]

        render_rectangles(worker_plot, rectangles)

        for i, (task, _) in enumerate(locations):
            task_to_loc[task] = rectangles[i]

        frame = DataFrame()
        frame["label"] = [t[0].name for t in locations]
        frame["bottom"] = [normalize_height(t[1][2]) for t in locations]
        frame["left"] = [t[1][0] for t in locations]

        source = models.ColumnDataSource(frame)
        labels = models.LabelSet(x='left', y='bottom',
                                 x_offset=2, y_offset=2,
                                 text='label', source=source,
                                 text_color="white", render_mode='canvas')
        worker_plot.add_layout(labels)

    if show_communication:
        frame = DataFrame(merge_trace_events(
            trace_events, lambda e: isinstance(e, FetchStartTraceEvent),
            lambda e: isinstance(e, FetchEndTraceEvent),
            lambda e: (e.output, e.target_worker, e.source_worker),
            end_map=lambda e1, e2: (workers.index(e1.source_worker),
                                    e1.time,
                                    workers.index(e2.target_worker),
                                    e2.time,
                                    "{}/{:.2f}".format(e1.output.parent.name, e1.output.size),
                                    e1.output)),
            columns=["worker", "start", "worker2", "end", "label", "output"])
        frame["src_task_y"] = frame["output"].map(
            lambda o: (task_to_loc[o.parent][2] + task_to_loc[o.parent][3]) / 2)

        plot = plots[-1]
        source = models.ColumnDataSource(frame.drop(columns=["output"]))
        plot.segment(y0='src_task_y', x0='start', y1='worker2', x1='end',
                     source=source, line_color="black", line_width=2)
        plot.circle(y="worker2", x="end", source=source, size=15, color="black")

        labels = models.LabelSet(x='end', y='worker2', text='label',
                                 x_offset=-10, source=source, text_color="black",
                                 text_align="right", render_mode='canvas')
        plot.add_layout(labels)

    return gridplot(plots, ncols=2)


def plot_worker_usage(trace_events, workers):
    """
    Plots aggregated worker core usage into a grid chart (one chart per worker).
    """
    from bokeh.plotting import figure
    from bokeh.layouts import gridplot

    plots = []

    end_time = math.ceil(max([e.time for e in trace_events]))

    for index, worker in enumerate(workers):
        rectangles = list(build_worker_usage(trace_events, worker))
        plot = figure(plot_width=600,
                      plot_height=300,
                      x_range=(0, end_time),
                      y_range=(0, worker.cpus),
                      title='Worker {} core usage'.format(index))
        plot.yaxis.axis_label = 'Cores used'
        plot.yaxis.ticker = list(range(worker.cpus + 1))
        plot.xaxis.axis_label = 'Time'

        render_rectangles(plot, rectangles, fill_color="blue", line_color="blue")
        plots.append(plot)

    return gridplot(plots, ncols=2)


def plot_worker_transfer_size(trace_events, workers):
    """
    Plots cumulative worker in/out transfer size into a grid chart (one chart per worker).
    """
    from bokeh.plotting import figure
    from bokeh.layouts import gridplot

    plots = []

    end_time = math.ceil(max([e.time for e in trace_events]))

    for index, worker in enumerate(workers):
        (bandwidth_in, bandwidth_out) = list(build_worker_transfer_size(trace_events, worker))

        bandwidth_in.append((end_time, bandwidth_in[-1][1]))
        bandwidth_out.append((end_time, bandwidth_out[-1][1]))

        plot = figure(plot_width=600,
                      plot_height=300,
                      x_range=(0, end_time),
                      title='Worker {} transfer usage'.format(index))
        plot.yaxis.axis_label = 'Transfer'
        plot.xaxis.axis_label = 'Time'

        plot.step([s[0] for s in bandwidth_in], [s[1] for s in bandwidth_in],
                  mode="after",
                  line_color="green",
                  legend="In")
        plot.step([s[0] for s in bandwidth_out], [s[1] for s in bandwidth_out],
                  mode="after",
                  line_color="red",
                  legend="Out")
        plots.append(plot)

    return gridplot(plots, ncols=2)


def plot_worker_bandwidth(trace_events, workers):
    """
    Plots network bandwidth usage into a grid chart (one chart per worker).
    """
    from bokeh.plotting import figure
    from bokeh.layouts import gridplot

    plots = []

    end_time = math.ceil(max([e.time for e in trace_events]))

    for index, worker in enumerate(workers):
        events = [e for e in trace_events if isinstance(e, BandwidthChangeEvent)
                  and e.worker == worker]

        plot = figure(plot_width=600,
                      plot_height=300,
                      x_range=(0, end_time),
                      title='Worker {} bandwidth usage'.format(index))
        plot.yaxis.axis_label = 'Bandwidth'
        plot.xaxis.axis_label = 'Time'

        plot.step([e.time for e in events if not e.out], [e.value for e in events if not e.out],
                  mode="after",
                  legend="In",
                  line_color="green")
        plot.step([e.time for e in events if e.out], [e.value for e in events if e.out],
                  mode="after",
                  legend="Out",
                  line_color="red")
        plots.append(plot)

    return gridplot(plots, ncols=2)


def plot_tabs(trace_events, workers, plot_fns, labels):
    from bokeh.models import Panel, Tabs
    return Tabs(tabs=[Panel(child=fn(trace_events, workers), title=label)
                      for (fn, label) in zip(plot_fns, labels)])


def plot_all(trace_events, workers):
    return plot_tabs(trace_events, workers,
                     [plot_task_communication,
                      lambda *arg: plot_task_communication(*arg, show_communication=True),
                      plot_worker_usage,
                      plot_worker_bandwidth,
                      plot_worker_transfer_size],
                     ["Tasks", "Tasks + communication", "Core usage", "Bandwidth usage",
                      "Transfer size"])


def build_trace_html(trace_events, workers, filename, plot_fn):
    """
    Render trace events into a HTML file according to the given plot function.
    :type plot_fn: Callable[[List[Event], List[Worker]], bokeh.plotting.figure.Figure]
    """
    import bokeh.io

    # put end events before start events
    trace_events = sorted(trace_events, key=lambda e: (
        e.time, 0 if isinstance(e, TaskEndTraceEvent) else 1))

    plot = plot_fn(trace_events, workers)

    bokeh.io.output_file(filename)
    bokeh.io.save(plot)


def simulator_trace_to_html(simulator, filename):
    build_trace_html(simulator.trace_events, simulator.workers, filename, plot_all)


def render_rectangles(plot, locations, fill_color="blue", line_color="black"):
    left = [r[0] for r in locations]
    right = [r[1] for r in locations]
    bottom = [r[2] for r in locations]
    top = [r[3] for r in locations]

    plot.quad(left=left, right=right, top=top, bottom=bottom, fill_color=fill_color,
              line_color=line_color, line_width=2)
