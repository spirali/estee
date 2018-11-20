import io
import os

import pytest

from schedsim.common import TaskGraph
from schedsim.serialization.dax import dax_deserialize, dax_serialize


def test_load_graph():
    tg = dax_deserialize(os.path.join(os.path.dirname(__file__), "graph.dax"))

    assert isinstance(tg, TaskGraph)
    assert len(tg.tasks) == 3

    tasks = {}
    for t in tg.tasks:
        tasks[t.name] = t

    assert tasks["SpatialClustering"].name == "SpatialClustering"
    assert tasks["SpatialClustering"].duration == 129
    assert tasks["SpatialClustering"].expected_duration is None
    assert tasks["SpatialClustering"].cpus == 4

    assert tasks["RemoveAttributes"].name == "RemoveAttributes"
    assert tasks["RemoveAttributes"].duration == 66
    assert tasks["RemoveAttributes"].expected_duration is None
    assert tasks["RemoveAttributes"].cpus == 1

    inputs = tasks["RemoveAttributes"].inputs
    assert len(inputs) == 1
    assert inputs[0].parent == tasks["StormDetection"]
    assert inputs[0].size == pytest.approx(0.9765625)
    assert inputs[0].expected_size is None

    outputs = tasks["RemoveAttributes"].outputs
    assert len(outputs) == 1
    assert outputs[0].size == pytest.approx(0.9765625)
    assert outputs[0].expected_size == pytest.approx(0.0057487487)

    assert tasks["StormDetection"].name == "StormDetection"
    assert tasks["StormDetection"].duration == 35
    assert tasks["StormDetection"].expected_duration == 30
    assert tasks["StormDetection"].cpus == 1

    assert set(tasks["StormDetection"].consumers()) == {tasks["RemoveAttributes"],
                                                        tasks["SpatialClustering"]}


def test_serialize_deserialize(plan1):
    f = io.BytesIO()
    dax_serialize(plan1, f)

    f.seek(0)
    graph = dax_deserialize(f)

    assert len(graph.tasks) == len(plan1.tasks)


def test_serialize_plan1(plan1):
    f = io.BytesIO()
    dax_serialize(plan1, f)

    f.seek(0)
    xml = f.read().decode()
    assert xml == """<?xml version='1.0' encoding='UTF-8'?>
<adag>
  <job cores="1" expectedRuntime="2" id="task-0" name="a1" runtime="2">
    <uses expectedSize="1" file="task-0-o0" link="output" size="1"/>
  </job>
  <job cores="1" expectedRuntime="3" id="task-1" name="a2" runtime="3">
    <uses expectedSize="3" file="task-1-o0" link="output" size="3"/>
  </job>
  <job cores="1" expectedRuntime="2" id="task-2" name="a3" runtime="2">
    <uses expectedSize="1" file="task-2-o0" link="output" size="1"/>
    <uses expectedSize="1" file="task-2-o1" link="output" size="1"/>
    <uses expectedSize="1" file="task-0-o0" link="input" size="1"/>
  </job>
  <job cores="1" expectedRuntime="1" id="task-3" name="a4" runtime="1">
    <uses expectedSize="6" file="task-3-o0" link="output" size="6"/>
  </job>
  <job cores="1" expectedRuntime="1" id="task-4" name="a5" runtime="1">
    <uses expectedSize="1" file="task-4-o0" link="output" size="1"/>
    <uses expectedSize="3" file="task-1-o0" link="input" size="3"/>
    <uses expectedSize="1" file="task-2-o0" link="input" size="1"/>
    <uses expectedSize="6" file="task-3-o0" link="input" size="6"/>
  </job>
  <job cores="1" expectedRuntime="6" id="task-5" name="a6" runtime="6">
    <uses expectedSize="1" file="task-5-o0" link="output" size="1"/>
    <uses expectedSize="6" file="task-3-o0" link="input" size="6"/>
  </job>
  <job cores="1" expectedRuntime="1" id="task-6" name="a7" runtime="1">
    <uses expectedSize="2" file="task-6-o0" link="output" size="2"/>
  </job>
  <job cores="1" expectedRuntime="1" id="task-7" name="a8" runtime="1">
    <uses expectedSize="1" file="task-2-o1" link="input" size="1"/>
    <uses expectedSize="1" file="task-4-o0" link="input" size="1"/>
    <uses expectedSize="1" file="task-5-o0" link="input" size="1"/>
    <uses expectedSize="2" file="task-6-o0" link="input" size="2"/>
  </job>
  <child ref="task-2">
    <parent ref="task-0"/>
  </child>
  <child ref="task-4">
    <parent ref="task-1"/>
    <parent ref="task-2"/>
    <parent ref="task-3"/>
  </child>
  <child ref="task-5">
    <parent ref="task-3"/>
  </child>
  <child ref="task-7">
    <parent ref="task-2"/>
    <parent ref="task-4"/>
    <parent ref="task-5"/>
    <parent ref="task-6"/>
  </child>
</adag>
"""
