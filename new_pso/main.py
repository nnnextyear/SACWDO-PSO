import os
import sys
import time
import random

random.seed(42)

sys.path.insert(0, os.path.dirname(__file__))

from problem.data_loader import load_solomon_csv, DATA_DIR
from problem.model import VRPTWProblem
from solvers.improved_pso import ImprovedPSO


def main():
    files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
    if not files:
        print(f'No CSV dataset found in {DATA_DIR}')
        sys.exit(1)

    dataset_file = files[0]
    print('=' * 60)
    print(f'SACWDO-PSO for VRPTW')
    print('=' * 60)

    data = load_solomon_csv(dataset_file)
    problem = VRPTWProblem(data)

    print(f'Customers: {problem.num_customers}')
    print(f'Max vehicles: {problem.max_vehicles}')
    print(f'Capacity: {problem.vehicle_capacity}')
    print(f'Penalties: early={problem.penalty_early}, late={problem.penalty_late}')

    t0 = time.time()
    solver = ImprovedPSO(problem, pop_size=240, max_iterations=600,
                         sa_frequency=10, sa_intensity=20, verbose=True)
    result = solver.run()
    elapsed = time.time() - t0

    details = result['best_details']
    print()
    print('=' * 60)
    print('FINAL RESULTS')
    print('=' * 60)
    print(f'Total cost:      {details["total_cost"]:.2f}')
    print(f'Distance:        {details["total_distance"]:.2f}')
    print(f'Penalty:         {details["penalty_cost"]:.2f}')
    print(f'Vehicles:        {details["num_vehicles"]}')
    print(f'Tardiness:       {details["total_tardiness"]:.2f}')
    print(f'Violations:      {details["late_violations"]}')
    print(f'Time:            {elapsed:.1f}s')
    print('=' * 60)


if __name__ == '__main__':
    main()
