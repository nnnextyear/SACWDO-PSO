# Simulation Data

This directory contains the UE4/AirSim simulation result data used in the manuscript analysis.

## Directory Layout

```text
improved_no_obstacle/          # SACWDO-PSO, obstacle-free scenario
discrete_pso_no_obstacle/      # DPSO, obstacle-free scenario
improved_obstacle/             # SACWDO-PSO, obstacle scenario
discrete_pso_obstacle/         # DPSO, obstacle scenario
```

Each subdirectory contains CSV files for the five virtual UGVs used in the corresponding simulation scenario.

These files provide the simulation result records used for the obstacle-free and obstacle execution-feasibility analysis in the manuscript.

The UE4/AirSim simulation is used as an execution-feasibility demonstration after the benchmark-based task-allocation results are obtained. The complete UE4/AirSim project files are not included because the algorithmic reproducibility of SACWDO-PSO is supported by the released source code, benchmark dataset, parameter settings, and fixed random seed.
