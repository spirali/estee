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
from schedsim.common import TaskGraph
from schedsim.schedulers import BlevelGtScheduler
from schedsim.worker import Worker
from schedsim.simulator import Simulator
from schedsim.communication import MinMaxFlowNetModel

# Create task graph containing 3 tasks
# (each task runs 1s and requires 1 CPU)
# t0 -> t1
#  L -> t2
tg = TaskGraph()
t0 = tg.new_task(duration=1, cpus=1, outputs=(1,))
t1 = tg.new_task(duration=1, cpus=1)
t1.add_input(t0)
t2 = tg.new_task(duration=1, cpus=1)
t2.add_input(t0)

# Create B-level scheduler
scheduler = BlevelGtScheduler()

# Define cluster with 2 workers (1 CPU each)
workers = [Worker(cpus=1) for _ in range(2)]

# Define MinMaxFlow network model (100MB/s)
netmodel = MinMaxFlowNetModel(bandwidth=100)

# Create simulator
simulator = Simulator(tg, workers, scheduler, netmodel)

# Run simulation
result = simulator.run()

# Print simulation time
print("Task graph execution makespan = {}".format(result))
```

## Built-in implementations

We provide implementations of multiple scheduling heuristics and network models that can be used out of the box. 

### Schedulers

Built-in schedulers can be imported as follows:

`from schedsim.schedulers import`:
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

`from schedsim.communication import`:
 * `InstantNetModel`
 * `SimpleNetModel`
 * `MinMaxFlowNetModel`