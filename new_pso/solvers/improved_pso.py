import random
import copy
import math
from components.clarke_wright import build_cw_solution, build_random_solution
from components.sa_search import simulated_annealing
from components.adaptive import AdaptiveParams


class ImprovedPSO:
    """Improved PSO for VRPTW.

    Key components:
    - CW savings algorithm for high-quality initialization
    - Discrete velocity as operation sequences with directed learning
    - Adaptive parameters (tanh for w, cosh for c1/c2)
    - Simulated annealing for local search refinement
    """

    def __init__(self, problem, pop_size=240, max_iterations=600,
                 sa_frequency=10, sa_intensity=20, verbose=True):
        self.problem = problem
        self.pop_size = pop_size
        self.max_iterations = max_iterations
        self.sa_frequency = sa_frequency
        self.sa_intensity = sa_intensity
        self.verbose = verbose
        self.adaptive = AdaptiveParams()

        self.particles = []
        self.pbest = []
        self.pbest_costs = []
        self.gbest = None
        self.gbest_cost = float('inf')
        self.history = []
        self.stagnation = 0

    def _init_particle(self):
        r = random.random()
        if r < 0.6:
            return build_cw_solution(self.problem, randomize=True)
        return build_random_solution(self.problem)

    def initialize(self):
        for _ in range(self.pop_size):
            sol = self._init_particle()
            cost = self.problem.evaluate(sol)
            self.particles.append(sol)
            self.pbest.append(copy.deepcopy(sol))
            self.pbest_costs.append(cost)
            if cost < self.gbest_cost:
                self.gbest_cost = cost
                self.gbest = copy.deepcopy(sol)

        gbest_routes = self.problem.decode_routes(self.gbest)
        gbest_routes = [self._reorder_route_ci(r) for r in gbest_routes]
        self.gbest = self._routes_to_sol(gbest_routes)
        self.gbest_cost = self.problem.evaluate(self.gbest)
        self.history.append(self.gbest_cost)

    def _sort_by_tw(self, route):
        """Sort route by time window center for TW feasibility."""
        if len(route) <= 2:
            return route
        return sorted(route, key=lambda c: sum(self.problem.time_windows[c]) / 2)

    def _reorder_route_ci(self, route):
        """Cheapest-insertion reordering for a single route (used for gbest only)."""
        if len(route) <= 2:
            return route
        unrouted = set(route)
        seed = min(route, key=lambda c: self.problem.dist(self.problem.depot, c))
        unrouted.remove(seed)
        ordered = [self.problem.depot, seed, self.problem.depot]
        while unrouted:
            best_cust = None
            best_pos = None
            best_cost = float('inf')
            for cust in unrouted:
                for pos in range(1, len(ordered)):
                    trial = ordered[:pos] + [cust] + ordered[pos:]
                    cost = self._eval_partial_route(trial)
                    if cost < best_cost:
                        best_cost = cost
                        best_cust = cust
                        best_pos = pos
            ordered.insert(best_pos, best_cust)
            unrouted.remove(best_cust)
        return ordered[1:-1]

    def _eval_partial_route(self, route):
        """Fast evaluation of a depot-wrapped partial route."""
        total = 0.0
        prev = route[0]
        elapsed = 0.0
        for cust in route[1:]:
            if cust == self.problem.depot:
                total += self.problem.dist(prev, cust)
                prev = cust
                elapsed = 0.0
                continue
            elapsed += self.problem.dist(prev, cust)
            ready, due = self.problem.time_windows[cust]
            if elapsed < ready:
                total += self.problem.penalty_early * (ready - elapsed)
                elapsed = ready
            elif elapsed > due:
                total += self.problem.penalty_late * (elapsed - due)
            elapsed += self.problem.service_times[cust]
            total += self.problem.dist(prev, cust)
            prev = cust
        return total

    def _compute_diff(self, routes, target_routes):
        """Compute operations to move from routes toward target_routes.
        Returns a list of (cust, target_route_idx, target_pos) tuples."""
        current_pos = {}
        for ri, route in enumerate(routes):
            for pi, cust in enumerate(route):
                current_pos[cust] = (ri, pi)

        target_pos = {}
        for ri, route in enumerate(target_routes):
            for pi, cust in enumerate(route):
                target_pos[cust] = (ri, pi)

        ops = []
        all_custs = set(current_pos.keys()) | set(target_pos.keys())
        for cust in all_custs:
            if cust not in current_pos or cust not in target_pos:
                continue
            cr, cp = current_pos[cust]
            tr, tp = target_pos[cust]
            if cr != tr or abs(cp - tp) > 1:
                ops.append((cust, tr, tp))

        random.shuffle(ops)
        return ops

    def _apply_learn_ops(self, routes, target_routes, num_ops):
        """Apply directed learning operations with TW-aware acceptance.

        Moves customers toward their target positions but rejects moves
        that would significantly worsen TW feasibility.
        """
        diffs = self._compute_diff(routes, target_routes)
        if not diffs:
            return

        applied = 0
        for cust, target_route_idx, target_pos in diffs:
            if applied >= num_ops:
                break

            src_route_idx = None
            src_pos = None
            for ri, route in enumerate(routes):
                if cust in route:
                    src_route_idx = ri
                    src_pos = route.index(cust)
                    break

            if src_route_idx is None:
                continue

            if target_route_idx >= len(routes):
                target_route_idx = random.randrange(len(routes))

            # Remember TW cost before move
            tw_before = self._route_tw_cost(routes[src_route_idx])
            if target_route_idx != src_route_idx and target_route_idx < len(routes):
                tw_before += self._route_tw_cost(routes[target_route_idx])

            routes[src_route_idx].pop(src_pos)
            src_became_empty = not routes[src_route_idx]
            if src_became_empty:
                routes.pop(src_route_idx)
                if src_route_idx < target_route_idx:
                    target_route_idx -= 1
                if target_route_idx >= len(routes):
                    target_route_idx = len(routes) - 1

            if target_route_idx < 0:
                target_route_idx = 0
            if target_route_idx >= len(routes):
                routes.append([])
                target_route_idx = len(routes) - 1

            insert_pos = min(target_pos, len(routes[target_route_idx]))
            routes[target_route_idx].insert(insert_pos, cust)

            load = sum(self.problem.demands[c] for c in routes[target_route_idx])
            if load > self.problem.vehicle_capacity * 1.1:
                # Revert: capacity violation
                routes[target_route_idx].pop(insert_pos)
                if not routes[target_route_idx]:
                    routes.pop(target_route_idx)
                if src_became_empty:
                    routes.insert(src_route_idx, [cust])
                else:
                    routes[src_route_idx].insert(src_pos, cust)
                continue

            # TW check: compute cost after move
            tw_after = self._route_tw_cost(routes[target_route_idx])
            if not src_became_empty and src_route_idx < len(routes):
                tw_after += self._route_tw_cost(routes[src_route_idx])

            if tw_after > tw_before * 1.2 + 100:
                # Revert: TW cost increased too much
                routes[target_route_idx].pop(insert_pos)
                if not routes[target_route_idx]:
                    routes.pop(target_route_idx)
                if src_became_empty:
                    routes.insert(src_route_idx, [cust])
                else:
                    routes[src_route_idx].insert(src_pos, cust)
                continue

            applied += 1

    def _route_tw_cost(self, route):
        """Fast TW cost (lateness * penalty_late) for a single route."""
        cost = 0.0
        elapsed = 0.0
        prev = self.problem.depot
        for cust in route:
            elapsed += self.problem.dist(prev, cust)
            _, due = self.problem.time_windows[cust]
            if elapsed > due:
                cost += (elapsed - due) * self.problem.penalty_late
            ready, _ = self.problem.time_windows[cust]
            if elapsed < ready:
                elapsed = ready
            elapsed += self.problem.service_times[cust]
            prev = cust
        return cost

    def _apply_random_ops(self, routes, num_ops):
        """Apply random perturbation operations with TW-aware acceptance.

        Each op checks the TW cost of affected routes before and after.
        Changes that significantly worsen TW (>20% increase) are rejected.
        Ops 5 (TW-split) and 6 (TW-relocate) are weighted 2x.
        """
        for _ in range(num_ops):
            op = random.choices([1, 2, 3, 4, 5, 6], weights=[1, 1, 1, 1, 2, 2])[0]

            if op == 1:
                non_empty = [r for r in routes if len(r) >= 2]
                if not non_empty:
                    continue
                ri = random.randrange(len(routes))
                while len(routes[ri]) < 2:
                    ri = random.randrange(len(routes))
                route = routes[ri]
                a, b = random.sample(range(len(route)), 2)
                tw_before = self._route_tw_cost(route)
                route[a], route[b] = route[b], route[a]
                tw_after = self._route_tw_cost(route)
                if tw_after > tw_before * 1.2 + 50:
                    route[a], route[b] = route[b], route[a]  # revert

            elif op == 2:
                non_empty = [i for i, r in enumerate(routes) if len(r) >= 1]
                if len(non_empty) < 1:
                    continue
                src_idx = random.choice(non_empty)
                src = routes[src_idx]
                pos = random.randrange(len(src))
                cust = src[pos]
                dst_idx = random.randrange(len(routes))
                if dst_idx == src_idx and len(routes) < 2:
                    continue

                tw_before = self._route_tw_cost(src)
                if dst_idx != src_idx:
                    tw_before += self._route_tw_cost(routes[dst_idx])

                src.pop(pos)
                insert_pos = random.randrange(len(routes[dst_idx]) + 1)
                routes[dst_idx].insert(insert_pos, cust)

                load = sum(self.problem.demands[c] for c in routes[dst_idx])
                if load > self.problem.vehicle_capacity * 1.1:
                    routes[dst_idx].pop(insert_pos)
                    src.insert(pos, cust)
                    continue

                tw_after = self._route_tw_cost(src) if src else 0
                if dst_idx != src_idx or src:
                    tw_after += self._route_tw_cost(routes[dst_idx])
                # If src became empty, it will be cleaned up later

                if tw_after > tw_before * 1.2 + 100:
                    routes[dst_idx].pop(insert_pos)
                    src.insert(pos, cust)  # revert

            elif op == 3:
                non_empty = [r for r in routes if len(r) >= 2]
                if not non_empty:
                    continue
                ri = random.randrange(len(routes))
                while len(routes[ri]) < 2:
                    ri = random.randrange(len(routes))
                route = routes[ri]
                a, b = sorted(random.sample(range(len(route)), 2))
                tw_before = self._route_tw_cost(route)
                route[a:b+1] = reversed(route[a:b+1])
                tw_after = self._route_tw_cost(route)
                if tw_after > tw_before * 1.2 + 50:
                    route[a:b+1] = reversed(route[a:b+1])  # revert

            elif op == 4:
                if len(routes) < 2:
                    continue
                eligible = [i for i, r in enumerate(routes) if len(r) >= 1]
                if len(eligible) < 2:
                    continue
                i1 = random.choice(eligible)
                i2 = random.choice([j for j in eligible if j != i1])
                r1, r2 = routes[i1], routes[i2]
                p1 = random.randrange(len(r1))
                p2 = random.randrange(len(r2))
                tw_before = self._route_tw_cost(r1) + self._route_tw_cost(r2)
                tail1, tail2 = r1[p1:], r2[p2:]
                r1[p1:], r2[p2:] = tail2, tail1
                load1 = sum(self.problem.demands[c] for c in r1)
                load2 = sum(self.problem.demands[c] for c in r2)
                if load1 > self.problem.vehicle_capacity * 1.1 or \
                   load2 > self.problem.vehicle_capacity * 1.1:
                    r1[p1:], r2[p2:] = tail1, tail2
                    continue
                tw_after = self._route_tw_cost(r1) + self._route_tw_cost(r2)
                if tw_after > tw_before * 1.2 + 100:
                    r1[p1:], r2[p2:] = tail1, tail2  # revert

            elif op == 5:
                # TW-based route split: find a route with violations and split it
                eligible = []
                for ri, r in enumerate(routes):
                    if len(r) < 2:
                        continue
                    # Quick TW violation count
                    elapsed = 0.0
                    prev = self.problem.depot
                    for pos, cust in enumerate(r):
                        elapsed += self.problem.dist(prev, cust)
                        _, due = self.problem.time_windows[cust]
                        if elapsed > due:
                            eligible.append((ri, pos, elapsed - due))
                            break
                        ready, _ = self.problem.time_windows[cust]
                        if elapsed < ready:
                            elapsed = ready
                        elapsed += self.problem.service_times[cust]
                        prev = cust

                if eligible and len(routes) < self.problem.max_vehicles:
                    ri, pos, _ = random.choice(eligible)
                    route = routes[ri]
                    part1 = route[:pos]
                    part2 = route[pos:]
                    if part1 and part2:
                        routes[ri] = part1
                        routes.append(part2)

            elif op == 6:
                # TW-targeted relocate: find the latest customer and move to
                # a route where it arrives earlier (or less late).
                worst_cust = None
                worst_lateness = 0
                worst_src = None
                worst_pos = None
                for ri, r in enumerate(routes):
                    elapsed = 0.0
                    prev = self.problem.depot
                    for pos, cust in enumerate(r):
                        elapsed += self.problem.dist(prev, cust)
                        _, due = self.problem.time_windows[cust]
                        if elapsed > due:
                            lateness = elapsed - due
                            if lateness > worst_lateness:
                                worst_lateness = lateness
                                worst_cust = cust
                                worst_src = ri
                                worst_pos = pos
                        ready, _ = self.problem.time_windows[cust]
                        if elapsed < ready:
                            elapsed = ready
                        elapsed += self.problem.service_times[cust]
                        prev = cust

                if worst_cust is not None and len(routes) > 1:
                    # Try each target route, find the one where this customer
                    # arrives earliest (minimizing lateness)
                    best_dst = None
                    best_pos_dst = None
                    best_arrival = float('inf')
                    for ti, target in enumerate(routes):
                        if ti == worst_src:
                            continue
                        elapsed = 0.0
                        prev = self.problem.depot
                        for tp in range(len(target) + 1):
                            # Simulate arrival at worst_cust if inserted at tp
                            if tp == 0:
                                arr = self.problem.dist(self.problem.depot, worst_cust)
                            else:
                                # Compute arrival through the target route
                                e = 0.0
                                p = self.problem.depot
                                for c in target[:tp]:
                                    e += self.problem.dist(p, c)
                                    rd, _ = self.problem.time_windows[c]
                                    if e < rd:
                                        e = rd
                                    e += self.problem.service_times[c]
                                    p = c
                                e += self.problem.dist(p, worst_cust)
                                arr = e
                            if arr < best_arrival:
                                # Check capacity
                                trial_load = sum(self.problem.demands[c] for c in target) + self.problem.demands[worst_cust]
                                if trial_load <= self.problem.vehicle_capacity * 1.05:
                                    best_arrival = arr
                                    best_dst = ti
                                    best_pos_dst = tp

                    if best_dst is not None:
                        routes[worst_src].pop(worst_pos)
                        if not routes[worst_src]:
                            routes.pop(worst_src)
                            if worst_src < best_dst:
                                best_dst -= 1
                        if best_dst >= len(routes):
                            routes.append([worst_cust])
                        else:
                            routes[best_dst].insert(best_pos_dst, worst_cust)

    def _routes_to_sol(self, routes):
        routes = [r for r in routes if r]
        if len(routes) > self.problem.max_vehicles:
            routes = self._merge_smallest_routes(routes)
        sol = [0]
        for r in routes:
            sol.extend(r)
            sol.append(0)
        return sol

    def _merge_smallest_routes(self, routes):
        """Merge smallest routes until within vehicle limit, considering TW compatibility."""
        while len(routes) > self.problem.max_vehicles:
            routes.sort(key=len)
            smallest = routes[0]
            best_target = None
            best_tw_cost = float('inf')

            for target in routes[1:]:
                combined_load = sum(self.problem.demands[c] for c in smallest + target)
                if combined_load > self.problem.vehicle_capacity:
                    continue
                tw_cost = self._route_tw_cost(target + smallest)
                if tw_cost < best_tw_cost:
                    best_tw_cost = tw_cost
                    best_target = target

            if best_target is not None:
                best_target.extend(smallest)
                routes.pop(0)
            else:
                routes.pop(0)
        return [r for r in routes if r]

    def _repair(self, routes):
        """Local repair: fix overloaded routes and TW-violating routes by splitting."""
        new_routes = []
        for route in routes:
            load = sum(self.problem.demands[c] for c in route)
            if load > self.problem.vehicle_capacity:
                # Capacity split
                current_route = []
                current_load = 0.0
                for cust in route:
                    d = self.problem.demands[cust]
                    if current_load + d <= self.problem.vehicle_capacity:
                        current_route.append(cust)
                        current_load += d
                    else:
                        if current_route:
                            new_routes.append(current_route)
                        current_route = [cust]
                        current_load = d
                if current_route:
                    new_routes.append(current_route)
            else:
                new_routes.append(route)

        # TW-based split: if a route has >5 violations and we have vehicle budget, split it
        result = []
        for route in new_routes:
            violations, worst_pos = 0, None
            worst_late = 0
            elapsed = 0.0
            prev = self.problem.depot
            for pos, cust in enumerate(route):
                elapsed += self.problem.dist(prev, cust)
                _, due = self.problem.time_windows[cust]
                if elapsed > due:
                    violations += 1
                    if elapsed - due > worst_late:
                        worst_late = elapsed - due
                        worst_pos = pos
                ready, _ = self.problem.time_windows[cust]
                if elapsed < ready:
                    elapsed = ready
                elapsed += self.problem.service_times[cust]
                prev = cust

            if violations > 5 and worst_pos is not None and worst_pos > 0:
                part1 = route[:worst_pos]
                part2 = route[worst_pos:]
                if part1:
                    result.append(part1)
                if part2:
                    result.append(part2)
            else:
                result.append(route)

        return [r for r in result if r]

    def _tw_improve(self, routes, iterations=30):
        """TW improvement: move late customers to routes where they arrive earlier.

        Examines the top-K latest customers and tries all target routes/positions,
        picking the best cost-improving move each iteration (steepest descent).
        Uses top-12 cutoff for speed while maintaining effectiveness.
        """
        for _ in range(iterations):
            # Find all late customers with their lateness
            late_customers = []
            for ri, route in enumerate(routes):
                if len(route) < 1:
                    continue
                elapsed = 0.0
                prev = self.problem.depot
                for pos, cust in enumerate(route):
                    elapsed += self.problem.dist(prev, cust)
                    _, due = self.problem.time_windows[cust]
                    if elapsed > due:
                        late_customers.append((ri, pos, cust, elapsed - due))
                    ready, _ = self.problem.time_windows[cust]
                    if elapsed < ready:
                        elapsed = ready
                    elapsed += self.problem.service_times[cust]
                    prev = cust

            if not late_customers:
                break

            # Focus on top 12 latest customers for speed
            late_customers.sort(key=lambda x: x[3], reverse=True)
            candidates = late_customers[:12]

            current_cost = self.problem.evaluate(self._routes_to_sol(routes))
            best_move = None
            best_cost = current_cost

            for src_ri, src_pos, cust, _ in candidates:
                for tj, target in enumerate(routes):
                    if tj == src_ri:
                        continue
                    for tp in range(len(target) + 1):
                        trial = target[:tp] + [cust] + target[tp:]
                        load = sum(self.problem.demands[c] for c in trial)
                        if load > self.problem.vehicle_capacity * 1.05:
                            continue
                        temp_routes = [list(r) for r in routes]
                        temp_routes[src_ri].pop(src_pos)
                        if not temp_routes[src_ri]:
                            continue
                        temp_routes[tj] = trial
                        sol = self._routes_to_sol(temp_routes)
                        c = self.problem.evaluate(sol)
                        if c < best_cost:
                            best_cost = c
                            best_move = (src_ri, src_pos, tj, tp)

            if best_move is None:
                break

            ri, pos, tj, tp = best_move
            cust = routes[ri].pop(pos)
            if not routes[ri]:
                routes.pop(ri)
                if ri < tj:
                    tj -= 1
            if tj >= len(routes):
                routes.append([cust])
            else:
                routes[tj].insert(tp, cust)

        return routes

    def run(self):
        self.initialize()
        if self.verbose:
            details = self.problem.evaluate(self.gbest, return_details=True)
            print(f'  Initial: cost={self.gbest_cost:.2f}, '
                  f'vehicles={details["num_vehicles"]}, '
                  f'dist={details["total_distance"]:.2f}, '
                  f'penalty={details["penalty_cost"]:.2f}')

        for it in range(self.max_iterations):
            improved = False

            for i in range(self.pop_size):
                pbest_c = self.pbest_costs[i]
                w, c1, c2 = self.adaptive.compute(pbest_c, self.gbest_cost, it, self.max_iterations)

                routes = self.problem.decode_routes(self.particles[i])
                routes_pbest = self.problem.decode_routes(self.pbest[i])
                routes_gbest = self.problem.decode_routes(self.gbest)

                num_random = max(1, int(w * 5))
                num_cognitive = max(1, int(c1 * 3))
                num_social = max(1, int(c2 * 3))

                self._apply_random_ops(routes, num_random)
                self._apply_learn_ops(routes, routes_pbest, num_cognitive)
                self._apply_learn_ops(routes, routes_gbest, num_social)

                routes = self._repair(routes)
                # TW-order each route for time-window coherence
                routes = [self._sort_by_tw(r) for r in routes]
                new_sol = self._routes_to_sol(routes)
                new_cost = self.problem.evaluate(new_sol)
                self.particles[i] = new_sol

                if new_cost < self.pbest_costs[i]:
                    self.pbest[i] = copy.deepcopy(new_sol)
                    self.pbest_costs[i] = new_cost

                if new_cost < self.gbest_cost:
                    self.gbest = copy.deepcopy(new_sol)
                    self.gbest_cost = new_cost
                    improved = True

            # TW-improvement on gbest every 5 iterations
            if it % 5 == 4:
                gbest_routes = self.problem.decode_routes(self.gbest)
                gbest_routes = self._tw_improve(gbest_routes, iterations=30)
                self.gbest = self._routes_to_sol(gbest_routes)
                self.gbest_cost = self.problem.evaluate(self.gbest)

            # SA refinement every sa_frequency iterations
            if it % self.sa_frequency == self.sa_frequency - 1:

                refined = simulated_annealing(
                    self.problem, self.gbest,
                    initial_temp=120.0, cooling_rate=0.97,
                    iterations_per_temp=self.sa_intensity, min_temp=0.1
                )
                refined_cost = self.problem.evaluate(refined)
                if refined_cost < self.gbest_cost:
                    self.gbest = refined
                    self.gbest_cost = refined_cost
                    improved = True

            self.history.append(self.gbest_cost)

            if improved:
                self.stagnation = 0
            else:
                self.stagnation += 1

            if self.stagnation > 80:
                routes = self.problem.decode_routes(self.gbest)
                self._apply_random_ops(routes, 15)
                routes = self._repair(routes)
                perturbed = self._routes_to_sol(routes)
                perturbed_cost = self.problem.evaluate(perturbed)
                if perturbed_cost < self.gbest_cost * 1.05:
                    self.gbest = perturbed
                    self.gbest_cost = perturbed_cost
                self.stagnation = 0

            if self.verbose and it % 100 == 0:
                details = self.problem.evaluate(self.gbest, return_details=True)
                print(f'  Iter {it}: cost={self.gbest_cost:.2f}, '
                      f'vehicles={details["num_vehicles"]}, '
                      f'dist={details["total_distance"]:.2f}, '
                      f'penalty={details["penalty_cost"]:.2f}')

        return {
            'best_solution': self.gbest,
            'best_cost': self.gbest_cost,
            'best_details': self.problem.evaluate(self.gbest, return_details=True),
            'history': self.history
        }
