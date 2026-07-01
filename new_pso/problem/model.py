class VRPTWProblem:
    def __init__(self, data):
        self.coords = data['coords']
        self.demands = data['demands']
        self.time_windows = data['time_windows']
        self.service_times = data['service_times']
        self.num_customers = data['num_customers']
        self.depot = 0

        self.cost_per_distance = 1.0
        self.penalty_early = 0.0
        self.penalty_late = 100.0
        self.vehicle_capacity = 200.0
        self.max_vehicles = 13
        self.huge_penalty = 999999.0

        self._dist_cache = None

    def set_penalties(self, early=30.0, late=100.0, cost_per_dist=1.0):
        self.penalty_early = early
        self.penalty_late = late
        self.cost_per_distance = cost_per_dist

    def dist(self, i, j):
        if self._dist_cache is None:
            import math
            n = len(self.coords)
            self._dist_cache = [[0.0] * n for _ in range(n)]
            for a in range(n):
                for b in range(n):
                    self._dist_cache[a][b] = math.sqrt(
                        (self.coords[a][0] - self.coords[b][0])**2 +
                        (self.coords[a][1] - self.coords[b][1])**2)
        return self._dist_cache[i][j]

    def decode_routes(self, solution):
        """solution is a list with 0 as depot separator, e.g. [0, 3, 7, 0, 1, 5, 0]"""
        routes = []
        current = []
        for x in solution[1:-1] if solution[0] == 0 and solution[-1] == 0 else solution:
            if x == 0:
                if current:
                    routes.append(current)
                    current = []
            else:
                current.append(x)
        if current:
            routes.append(current)
        return routes

    def evaluate(self, solution, return_details=False):
        routes = self.decode_routes(solution)

        total_distance = 0.0
        total_penalty = 0.0
        total_tardiness = 0.0
        late_violations = 0
        overload_cost = 0.0
        vehicle_count = len(routes)

        if vehicle_count > self.max_vehicles:
            penalty = self.huge_penalty + vehicle_count * 10000
            if return_details:
                return {
                    'total_cost': penalty,
                    'transport_cost': 0,
                    'total_distance': 0,
                    'penalty_cost': penalty,
                    'total_tardiness': 0,
                    'late_violations': 0,
                    'overload_cost': 0,
                    'num_vehicles': vehicle_count,
                    'routes': routes
                }
            return penalty

        for route in routes:
            load = sum(self.demands[i] for i in route)
            if load > self.vehicle_capacity:
                overload_cost += self.huge_penalty

            prev = self.depot
            elapsed = 0.0
            for cust in route:
                d = self.dist(prev, cust)
                total_distance += d
                elapsed += d
                ready, due = self.time_windows[cust]
                if elapsed < ready:
                    total_penalty += self.penalty_early * (ready - elapsed)
                    elapsed = ready
                elif elapsed > due:
                    tardiness = elapsed - due
                    total_tardiness += tardiness
                    total_penalty += self.penalty_late * tardiness
                    late_violations += 1
                elapsed += self.service_times[cust]
                prev = cust
            total_distance += self.dist(prev, self.depot)

        transport_cost = total_distance * self.cost_per_distance
        total_cost = transport_cost + total_penalty + overload_cost

        if return_details:
            return {
                'total_cost': total_cost,
                'transport_cost': transport_cost,
                'total_distance': total_distance,
                'penalty_cost': total_penalty,
                'total_tardiness': total_tardiness,
                'late_violations': late_violations,
                'overload_cost': overload_cost,
                'num_vehicles': vehicle_count,
                'routes': routes
            }
        return total_cost

    def count_time_window_violations(self, solution):
        routes = self.decode_routes(solution)
        violations = 0
        for route in routes:
            prev = self.depot
            elapsed = 0.0
            for cust in route:
                elapsed += self.dist(prev, cust)
                ready, due = self.time_windows[cust]
                if elapsed < ready:
                    elapsed = ready
                elif elapsed > due:
                    violations += 1
                elapsed += self.service_times[cust]
                prev = cust
        return violations
