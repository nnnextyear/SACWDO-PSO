# SACWDO-PSO for VRPTW

Clarke-Wright savings initialization + improved particle swarm optimization with adaptive parameters and simulated annealing for the Vehicle Routing Problem with Time Windows (VRPTW).

## Structure

```text
data/                          # CSV benchmark dataset used by the released implementation
new_pso/
├── main.py                    # Entry point
├── problem/
│   ├── model.py               # VRPTW problem model with soft time-window penalties
│   └── data_loader.py         # CSV data loader
├── components/
│   ├── clarke_wright.py       # Clarke-Wright savings initialization
│   ├── adaptive.py            # Adaptive parameter control
│   ├── sa_search.py           # Simulated annealing local search
│   └── local_search.py        # Neighborhood operators
└── solvers/
    └── improved_pso.py        # SACWDO-PSO solver
```

## Usage

```bash
cd new_pso
python main.py
```

## Reproducibility

The released implementation uses a fixed random seed in `main.py`:

```python
random.seed(42)
```

The repository includes the core source code, the benchmark dataset file used by the released implementation, and fixed random seed information. Changing or removing the seed will restore stochastic behavior.

## Experimental Settings in the Released Implementation

The released implementation follows the soft time-window setting used in the manuscript:

- early-arrival penalty coefficient: `30.0`
- late-arrival penalty coefficient: `100.0`
- vehicle capacity: `200`
- population size: `240`
- maximum iterations: `600`
- simulated-annealing frequency: every `10` iterations
- simulated-annealing intensity: `20` neighborhood moves

## Algorithm

The released code implements a hybrid SACWDO-PSO workflow:

1. Clarke-Wright savings initialization is used to improve the quality of the initial route population.
2. Discrete PSO updates are used to modify task assignment and route order.
3. Adaptive parameter control adjusts the search behavior during iterations.
4. Simulated annealing local search refines candidate routes.

The simulated-annealing local search uses four basic neighborhood operations consistent with the manuscript description:

- reassignment
- swap
- insertion
- route reversal

## Data Availability

The repository provides the core source code, the benchmark dataset file used by the released implementation, and fixed random seed information to support reproducibility of the released code.

## License

MIT
