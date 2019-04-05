### Task scheduler performance survey
This dataset contains results of task graph scheduler performance survey.
The results are stored in the following files, which correspond to simulations performed on 
the `elementary`, `irw` and `pegasus` task graph datasets published at https://doi.org/10.5281/zenodo.2630384.

* elementary-result.zip
* irw-result.zip
* pegasus-result.zip

The files contain compressed pandas dataframes in CSV format, you can read them with the
following Python code:
```python
import pandas as pd
frame = pd.read_csv("elementary-result.zip")
```

Each row in the frame corresponds to a single instance of a task graph that
was simulated with a specific configuration (network model, scheduler etc.).
The list below summarizes the meaning of the individual columns.

* **graph_name** - name of the benchmarked task graph
* graph_set - name of the task graph dataset from which the graph originates
* graph_id - unique ID of the graph
* **cluster_name** - type of cluster used in this instance
  * the format is <number-of-workers>x<number-of-cores>
  * 32x16 means 32 workers, each with 16 cores
* **bandwidth** - network bandwidth [MiB]
* **netmodel** - network model (simple or maxmin)
* **scheduler_name** - name of the scheduler
* **imode** - information mode
* **min_sched_interval** - minimal scheduling delay [s]
* sched_time - duration of each scheduler invocation [s]
* **time** - simulated makespan of the task graph execution [s]
* execution_time - real duration of all scheduler invocations [s]
* total_transfer - amount of data transferred amongst workers [MiB]


### Reproducing the results
##### 1. Download and install Estee (https://github.com/It4innovations/estee)
```bash
$ git clone https://github.com/It4innovations/estee
$ cd estee
$ pip install .
```
##### 2. Generate task graphs
You can either use the provided script `benchmarks/generate.py` to generate graphs
from three categories (elementary, irw and pegasus):
```bash
$ cd benchmarks
$ python generate.py elementary.zip elementary
$ python generate.py irw.zip irw
$ python generate.py pegasus.zip pegasus
```
or use our task graph dataset that is provided at https://doi.org/10.5281/zenodo.2630384.

##### 3. Run benchmarks
To run a benchmark suite, you should prepare a JSON file describing the benchmark.
The file that was used to run experiments from the paper is provided in
`benchmark.json`. Then you can run the benchmark using this command:
```bash
$ python pbs.py compute benchmark.json
```

The benchmark script can be interrupted at any time (for example using Ctrl+C).
When interrupted, it will store the computed results to the result file and restore
the computation when launched again.

#### 3. Visualizing results
```bash
$ python view.py --all <result-file>
```
The resulting plots will appear in a folder called `outputs`.
