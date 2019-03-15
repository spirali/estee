# Estee

Estee is a discrete event simulation environment for executing task graphs. It is designed for benchmarking and experimenting with schedulers. Estee is created as an open-ended simulator so most of its components can be extended or replaced by a custom implementation, hence it is possible to experiment with different kinds of schedulers, workers, and network models. However, it also comes battery-included and provides a standard implementation for all its components.

## Usage

### Installation

```
python3 setup.py install
```

### Example

Estee requires users to define:
 * Task graph
 * Scheduler
 * Workers
 * Network model

The following code demostrates usage of Estee using built in implementation of scheduler and network model.

```
from estee.common import TaskGraph
from estee.schedulers import BlevelGtScheduler
from estee.simulator import Simulator, Worker, MaxMinFlowNetModel

# Create task graph containing 3 tasks
# (each task runs 1s and requires 1 CPU)
# t0 -> t1
#  L -> t2
task_graph = TaskGraph()
t0 = task_graph.new_task(duration=1, cpus=1, outputs=(1,))
t1 = task_graph.new_task(duration=1, cpus=1)
t1.add_input(t0)
t2 = task_graph.new_task(duration=1, cpus=1)
t2.add_input(t0)

# Create B-level scheduler
scheduler = BlevelGtScheduler()

# Define cluster with 2 workers (1 CPU each)
workers = [Worker(cpus=1) for _ in range(2)]

# Define MaxMinFlow network model (100MB/s)
netmodel = MaxMinFlowNetModel(bandwidth=100)

# Create a simulator
simulator = Simulator(task_graph, workers, scheduler, netmodel)

# Run simulation, returns the makespan in seconds
makespan = simulator.run()

# Print simulation time
print("Task graph execution makespan = {}s".format(makespan))
```

## Built-in implementations

We provide implementations of multiple scheduling heuristics and network models that can be used out of the box. 

### Schedulers

Built-in schedulers can be imported as follows:

`from estee.schedulers import`:
 * `StaticScheduler`
 * `AllOnOneScheduler`
 * `DoNothingScheduler`
 * `RandomAssignScheduler`
 * `RandomScheduler`
 * `RandomGtScheduler`
 * `BlevelGtScheduler`
 * `DLSScheduler`
 * `ETFScheduler`
 * `LASTScheduler`
 * `MCPScheduler`
 * `Camp2Scheduler`

### Network models

Built-in network models can be imported as follows:

`from estee.communication import`:
 * `InstantNetModel`
 * `SimpleNetModel`
 * `MaxMinFlowNetModel`
