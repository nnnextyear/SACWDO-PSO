import random
import math
import copy


def simulated_annealing(problem, solution, initial_temp=120.0, cooling_rate=0.97,
                        iterations_per_temp=20, min_temp=0.1):
    """Simulated annealing local search with neighborhood operations.

    Temperature decays geometrically each move to control acceptance
    probability via Metropolis criterion.
    """
    current = copy.deepcopy(solution)
    current_cost = problem.evaluate(current)
    best = copy.deepcopy(current)
    best_cost = current_cost

    temp = initial_temp

    for _ in range(iterations_per_temp):
        neighbor = _random_neighbor(current[:], problem)
        neighbor_cost = problem.evaluate(neighbor)

        delta = neighbor_cost - current_cost

        if delta < 0:
            current = neighbor
            current_cost = neighbor_cost
            if current_cost < best_cost:
                best = copy.deepcopy(current)
                best_cost = current_cost
        elif random.random() < math.exp(-delta / temp):
            current = neighbor
            current_cost = neighbor_cost

        temp *= cooling_rate

    return best


def _random_neighbor(solution, problem):
    """Randomly apply one of five neighborhood operations with TW checks."""
    routes = problem.decode_routes(solution)

    if len(routes) < 1:
        return solution

    # Weight TW ops more heavily
    op = random.choices([1, 2, 3, 4, 5, 6], weights=[1, 1, 1, 1, 3, 3])[0]

    if op == 1:
        return _intra_route_swap(solution, routes, problem)
    elif op == 2:
        return _inter_route_relocate(solution, routes, problem)
    elif op == 3:
        return _intra_route_two_opt(solution, routes, problem)
    elif op == 4:
        return _inter_route_cross(solution, routes, problem)
    elif op == 5:
        return _tw_split(solution, routes, problem)
    else:
        return _tw_relocate(solution, routes, problem)


def _intra_route_swap(solution, routes, problem):
    """Swap two customers within the same route."""
    long_routes = [(i, r) for i, r in enumerate(routes) if len(r) >= 2]
    if not long_routes:
        return solution
    idx, route = random.choice(long_routes)
    a, b = random.sample(range(len(route)), 2)
    route[a], route[b] = route[b], route[a]
    return _routes_to_solution(routes)


def _inter_route_relocate(solution, routes, problem):
    """Move a customer from one route to another."""
    if len(routes) < 2:
        return solution
    non_empty = [(i, r) for i, r in enumerate(routes) if len(r) >= 1]
    if len(non_empty) < 2:
        return solution
    src_idx, src_route = random.choice(non_empty)
    dst_idx = random.choice([j for j in range(len(routes)) if j != src_idx])
    dst_route = routes[dst_idx]

    pos = random.randrange(len(src_route))
    cust = src_route.pop(pos)

    if not src_route:
        routes.pop(src_idx)
        if src_idx < dst_idx:
            dst_idx -= 1

    load = sum(problem.demands[c] for c in dst_route) + problem.demands[cust]
    if load > problem.vehicle_capacity * 1.05:
        src_route.insert(pos, cust)
        return solution

    insert_pos = random.randrange(len(dst_route) + 1)
    dst_route.insert(insert_pos, cust)
    return _routes_to_solution(routes)


def _intra_route_two_opt(solution, routes, problem):
    """2-opt reversal within a single route."""
    long_routes = [(i, r) for i, r in enumerate(routes) if len(r) >= 2]
    if not long_routes:
        return solution
    idx, route = random.choice(long_routes)
    if len(route) < 2:
        return solution
    a, b = sorted(random.sample(range(len(route)), 2))
    route[a:b+1] = reversed(route[a:b+1])
    return _routes_to_solution(routes)


def _inter_route_cross(solution, routes, problem):
    """Cross-exchange: swap tails between two routes."""
    if len(routes) < 2:
        return solution
    eligible = [(i, r) for i, r in enumerate(routes) if len(r) >= 1]
    if len(eligible) < 2:
        return solution
    i1, r1 = random.choice(eligible)
    i2 = random.choice([j for j in range(len(routes)) if j != i1])
    r2 = routes[i2]

    pos1 = random.randrange(len(r1))
    pos2 = random.randrange(len(r2))

    tail1 = r1[pos1:]
    tail2 = r2[pos2:]
    r1[pos1:] = tail2
    r2[pos2:] = tail1

    load1 = sum(problem.demands[c] for c in r1)
    load2 = sum(problem.demands[c] for c in r2)
    if load1 > problem.vehicle_capacity * 1.05 or load2 > problem.vehicle_capacity * 1.05:
        r1[pos1:] = tail1
        r2[pos2:] = tail2
        return solution

    return _routes_to_solution(routes)


def _routes_to_solution(routes):
    solution = [0]
    for route in routes:
        if route:
            solution.extend(route)
            solution.append(0)
    return solution


def _tw_relocate(solution, routes, problem):
    """Move the latest customer to a route where they arrive earlier."""
    # Find the latest customer across all routes
    worst_cust = None
    worst_lateness = 0
    worst_src = None
    worst_pos = None

    for ri, route in enumerate(routes):
        elapsed = 0.0
        prev = problem.depot
        for pos, cust in enumerate(route):
            elapsed += problem.dist(prev, cust)
            _, due = problem.time_windows[cust]
            if elapsed > due:
                lateness = elapsed - due
                if lateness > worst_lateness:
                    worst_lateness = lateness
                    worst_cust = cust
                    worst_src = ri
                    worst_pos = pos
            ready, _ = problem.time_windows[cust]
            if elapsed < ready:
                elapsed = ready
            elapsed += problem.service_times[cust]
            prev = cust

    if worst_cust is None or len(routes) < 2:
        return solution

    # Find best insertion position in any other route
    best_dst = None
    best_tp = None
    best_arrival = float('inf')

    for ti, target in enumerate(routes):
        if ti == worst_src:
            continue
        for tp in range(len(target) + 1):
            # Compute arrival time at worst_cust if inserted at tp
            e = 0.0
            p = problem.depot
            for c in target[:tp]:
                e += problem.dist(p, c)
                rd, _ = problem.time_windows[c]
                if e < rd:
                    e = rd
                e += problem.service_times[c]
                p = c
            e += problem.dist(p, worst_cust)
            if e < best_arrival:
                trial = target[:tp] + [worst_cust] + target[tp:]
                load = sum(problem.demands[c] for c in trial)
                if load <= problem.vehicle_capacity * 1.05:
                    best_arrival = e
                    best_dst = ti
                    best_tp = tp

    if best_dst is not None:
        routes[worst_src].pop(worst_pos)
        if not routes[worst_src]:
            routes.pop(worst_src)
            if worst_src < best_dst:
                best_dst -= 1
        if best_dst >= len(routes):
            routes.append([worst_cust])
        else:
            routes[best_dst].insert(best_tp, worst_cust)

    return _routes_to_solution(routes)


def _tw_split(solution, routes, problem):
    """Split a route at the worst TW violation point."""
    # Find route with worst TW violation
    worst_info = None
    worst_lateness = 0
    for ri, route in enumerate(routes):
        if len(route) < 2:
            continue
        elapsed = 0.0
        prev = problem.depot
        for pos, cust in enumerate(route):
            elapsed += problem.dist(prev, cust)
            _, due = problem.time_windows[cust]
            if elapsed > due:
                lateness = elapsed - due
                if lateness > worst_lateness:
                    worst_lateness = lateness
                    worst_info = (ri, pos)
            ready, _ = problem.time_windows[cust]
            if elapsed < ready:
                elapsed = ready
            elapsed += problem.service_times[cust]
            prev = cust

    if worst_info is None:
        return solution

    ri, pos = worst_info
    route = routes[ri]
    if pos <= 0 or pos >= len(route):
        return solution

    part1 = route[:pos]
    part2 = route[pos:]
    if not part1 or not part2:
        return solution

    if len(routes) >= problem.max_vehicles:
        return solution

    routes[ri] = part1
    routes.append(part2)
    return _routes_to_solution(routes)
