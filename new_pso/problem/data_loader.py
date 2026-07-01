import csv
import os
import math

_current_dir = os.path.dirname(os.path.abspath(__file__))
_DATA_ROOT = os.path.dirname(os.path.dirname(_current_dir))
DATA_DIR = os.path.join(_DATA_ROOT, 'data')


def load_solomon_csv(filename):
    """Load Solomon-format CSV. Returns dict with coords, demands, time_windows, service_times."""
    filepath = os.path.join(DATA_DIR, filename)
    coords = []
    demands = []
    time_windows = []
    service_times = []

    with open(filepath) as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            _, x, y, demand, ready, due, service = row
            coords.append((float(x), float(y)))
            demands.append(float(demand))
            time_windows.append((float(ready), float(due)))
            service_times.append(float(service))

    num_customers = len(coords) - 1

    return {
        'coords': coords,
        'demands': demands,
        'time_windows': time_windows,
        'service_times': service_times,
        'num_customers': num_customers,
        'depot': 0
    }


def compute_distance_matrix(coords):
    n = len(coords)
    dist = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            dist[i][j] = math.sqrt((coords[i][0] - coords[j][0])**2 +
                                   (coords[i][1] - coords[j][1])**2)
    return dist
