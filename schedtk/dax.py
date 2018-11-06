from .taskgraph import TaskGraph

import xml.etree.ElementTree


XMLNS_PREFIX = "{http://pegasus.isi.edu/schema/DAX}"


def load(path):
    tasks = {}
    tg = TaskGraph()

    root = xml.etree.ElementTree.parse(path).getroot()

    for job in root.findall("{}job".format(XMLNS_PREFIX)):
        files = job.findall("{}uses".format(XMLNS_PREFIX))
        output_size = sum([int(f.get("size")) for f in files if f.get("link") == "output"])
        id = job.get("id")
        runtime = float(job.get("runtime"))
        assert id not in tasks
        t = tg.new_task(name=id, duration=runtime, size=output_size)
        tasks[id] = t

    for child in root.findall("{}child".format(XMLNS_PREFIX)):
        child_task = tasks[child.get("ref")]
        parents = [tasks[p.get("ref")] for p in child.findall("{}parent".format(XMLNS_PREFIX))]
        child_task.add_inputs(parents)

    tg.validate()

    return tg
