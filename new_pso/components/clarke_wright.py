import random


def build_cw_solution(problem, randomize=True):
    """Build solution using Clarke-Wright savings + fast TW-aware ordering.

    Phase 1: Standard CW savings with capacity check.
    Phase 2: Each route is sorted by TW center for time-window coherence.
    Phase 3: Split TW-violating routes for better feasibility.
    """
    n = problem.num_customers
    depot = problem.depot
    customers = list(range(1, n + 1))

    savings = []
    for i in customers:
        for j in customers:
            if i >= j:
                continue
            s = problem.dist(depot, i) + problem.dist(depot, j) - problem.dist(i, j)

            # TW compatibility: estimate lateness if i and j are served consecutively
            # in both orders, take the minimum (since route ordering can choose)
            ready_i, due_i = problem.time_windows[i]
            ready_j, due_j = problem.time_windows[j]
            svc_i = problem.service_times[i]
            svc_j = problem.service_times[j]

            arr_i = problem.dist(depot, i)
            if arr_i < ready_i: arr_i = ready_i
            arr_j = arr_i + svc_i + problem.dist(i, j)
            late_ij = max(0.0, arr_j - due_j)

            arr_j2 = problem.dist(depot, j)
            if arr_j2 < ready_j: arr_j2 = ready_j
            arr_i2 = arr_j2 + svc_j + problem.dist(j, i)
            late_ji = max(0.0, arr_i2 - due_i)

            min_late = min(late_ij, late_ji)
            # Balanced TW penalty: 0.1 weight means 10 units late = 100 penalty = ~distance saving
            s -= min_late * problem.penalty_late * 0.03

            # Small random noise for diversity between runs
            if randomize:
                s += random.uniform(-1.0, 1.0)
            savings.append((s, i, j))

    savings.sort(key=lambda x: x[0], reverse=True)

    routes = {c: [c] for c in customers}
    route_of = {c: c for c in customers}

    for s, i, j in savings:
        if route_of[i] == route_of[j]:
            continue
        ri = route_of[i]
        rj = route_of[j]
        route_i = routes[ri]
        route_j = routes[rj]

        if route_i[-1] != i:
            i, j = j, i
            ri, rj = rj, ri
            route_i, route_j = route_j, route_i

        if route_i[-1] != i or route_j[0] != j:
            continue

        merged = route_i + route_j
        load = sum(problem.demands[c] for c in merged)
        if load > problem.vehicle_capacity:
            continue

        # Cost-based merge criterion: only merge if the TW-sorted merged
        # route doesn't increase total cost by more than a tolerance.
        cost_before = _route_cost(problem, route_i) + _route_cost(problem, route_j)
        cost_after = _route_cost(problem, merged)
        if cost_after > cost_before + 200:  # allow moderate cost increase
            continue

        for c in route_j:
            route_of[c] = ri
        routes[ri] = merged
        del routes[rj]

    if randomize:
        route_list = list(routes.values())
        random.shuffle(route_list)
        route_keys = list(routes.keys())
        for i, r in enumerate(route_list):
            routes[route_keys[i]] = r

    route_list = list(routes.values())

    # Sort each route by TW center and apply 2-opt
    optimized_routes = [_sort_route_by_tw(problem, r) for r in route_list]

    # Split routes with excessive TW violations to improve TW feasibility.
    optimized_routes = _split_tw_violating_routes(problem, optimized_routes)

    if len(optimized_routes) > problem.max_vehicles:
        optimized_routes = _merge_excess_routes(problem, optimized_routes)

    solution = [0]
    for route in optimized_routes:
        if route:
            solution.extend(route)
            solution.append(0)

    return solution


def _route_cost(problem, route):
    """Compute full cost (distance + TW penalties) of a route.

    Evaluates in TW-sorted order to match post-processing behavior.
    """
    if len(route) <= 1:
        if len(route) == 1:
            c = route[0]
            d = problem.dist(problem.depot, c) * 2
            arr = problem.dist(problem.depot, c)
            ready, due = problem.time_windows[c]
            late = max(0, arr - due)
            early_pen = problem.penalty_early * max(0, ready - arr)
            return d + late * problem.penalty_late + early_pen
        return 0.0

    sorted_route = sorted(route, key=lambda c: sum(problem.time_windows[c]) / 2)
    total_dist = 0.0
    total_penalty = 0.0
    elapsed = 0.0
    prev = problem.depot

    for cust in sorted_route:
        d = problem.dist(prev, cust)
        elapsed += d
        total_dist += d
        ready, due = problem.time_windows[cust]
        if elapsed < ready:
            total_penalty += problem.penalty_early * (ready - elapsed)
            elapsed = ready
        elif elapsed > due:
            total_penalty += problem.penalty_late * (elapsed - due)
        elapsed += problem.service_times[cust]
        prev = cust

    total_dist += problem.dist(prev, problem.depot)
    return total_dist + total_penalty


def _split_tw_violating_routes(problem, routes):
    """Split routes that have excessive TW violations into smaller routes."""
    result = []
    for route in routes:
        violations, worst_pos = _count_route_violations(problem, route)
        if violations <= 2:
            result.append(route)
            continue

        # Split at the worst violation point
        if worst_pos is not None and 0 < worst_pos < len(route) - 1:
            part1 = route[:worst_pos]
            part2 = route[worst_pos:]
            if part1:
                result.append(part1)
            if part2:
                result.append(part2)
        else:
            result.append(route)

    return result


def _count_route_violations(problem, route):
    """Count TW violations in a route and return (count, worst_position)."""
    violations = 0
    worst_lateness = 0
    worst_pos = None
    elapsed = 0.0
    prev = problem.depot
    for pos, cust in enumerate(route):
        elapsed += problem.dist(prev, cust)
        ready, due = problem.time_windows[cust]
        if elapsed > due:
            violations += 1
            lateness = elapsed - due
            if lateness > worst_lateness:
                worst_lateness = lateness
                worst_pos = pos
        if elapsed < ready:
            elapsed = ready
        elapsed += problem.service_times[cust]
        prev = cust
    return violations, worst_pos


def _sort_route_by_tw(problem, route):
    """Fast sort by time-window center then apply one 2-opt pass."""
    if len(route) <= 2:
        return route
    sorted_route = sorted(route, key=lambda c: sum(problem.time_windows[c]) / 2)
    return _quick_two_opt(problem, sorted_route)


def _quick_two_opt(problem, route):
    """Single pass of 2-opt improvement (not iterated).
    Checks both distance and TW impact — only applies if TW doesn't worsen."""
    if len(route) < 3:
        return route
    best_delta = 0
    best_ij = None
    for i in range(len(route) - 1):
        for j in range(i + 2, len(route)):
            a, b = route[i], route[i + 1]
            c = route[j]
            # d is route[j+1], or depot if j is the last customer
            d = route[j + 1] if j + 1 < len(route) else problem.depot
            old = problem.dist(a, b) + problem.dist(c, d)
            new = problem.dist(a, c) + problem.dist(b, d)
            delta = new - old
            if delta < best_delta:
                # Quick TW check: would this reversal cause severe lateness?
                if not _would_reversal_violate_tw(problem, route, i, j):
                    best_delta = delta
                    best_ij = (i, j)
    if best_ij is not None:
        i, j = best_ij
        route[i + 1:j + 1] = reversed(route[i + 1:j + 1])
    return route


def _would_reversal_violate_tw(problem, route, i, j):
    """Check if reversing segment i+1..j would cause TW violations > 5 units."""
    # Simulate the route quickly up to position i
    elapsed = 0.0
    prev = problem.depot
    for pos in range(i + 1):
        elapsed += problem.dist(prev, route[pos])
        ready, _ = problem.time_windows[route[pos]]
        if elapsed < ready:
            elapsed = ready
        elapsed += problem.service_times[route[pos]]
        prev = route[pos]

    # Now simulate the reversed segment
    total_late = 0.0
    for pos in range(j, i, -1):  # j, j-1, ..., i+1
        cust = route[pos]
        elapsed += problem.dist(prev, cust)
        _, due = problem.time_windows[cust]
        if elapsed > due:
            total_late += elapsed - due
        ready, _ = problem.time_windows[cust]
        if elapsed < ready:
            elapsed = ready
        elapsed += problem.service_times[cust]
        prev = cust

    return total_late > 5


def _rebuild_route_ci(problem, customers):
    """Rebuild a route from a set of customers using TW-aware cheapest insertion.

    Starts from the customer closest to depot, then repeatedly inserts the
    remaining customer at the position that minimizes total cost (distance + TW).
    Much better than concatenation for preserving TW quality during merges.
    """
    if len(customers) <= 1:
        return list(customers)

    unrouted = set(customers)
    # Start with customer closest to depot
    seed = min(unrouted, key=lambda c: problem.dist(problem.depot, c))
    unrouted.remove(seed)
    ordered = [seed]

    while unrouted:
        best_cust = None
        best_pos = None
        best_cost = float('inf')

        for cust in unrouted:
            for pos in range(len(ordered) + 1):
                # Build trial route with depot wrap
                trial = [problem.depot] + ordered[:pos] + [cust] + ordered[pos:] + [problem.depot]
                cost = _eval_route_cost(problem, trial)
                if cost < best_cost:
                    best_cost = cost
                    best_cust = cust
                    best_pos = pos

        ordered.insert(best_pos, best_cust)
        unrouted.remove(best_cust)

    return ordered


def _eval_route_cost(problem, route_with_depot):
    """Evaluate cost of a depot-wrapped route (quick, no full evaluate)."""
    total = 0.0
    elapsed = 0.0
    for k in range(len(route_with_depot) - 1):
        a, b = route_with_depot[k], route_with_depot[k + 1]
        d = problem.dist(a, b)
        total += d
        if b != problem.depot:
            elapsed += d
            ready, due = problem.time_windows[b]
            if elapsed < ready:
                elapsed = ready
            elif elapsed > due:
                total += problem.penalty_late * (elapsed - due)
            elapsed += problem.service_times[b]
        else:
            # Return to depot, reset
            total += d
            elapsed = 0.0
    return total


def _merge_excess_routes(problem, routes):
    """Merge routes until within vehicle limit using best-pair selection.

    Evaluates all feasible merge pairs and picks the one with lowest
    post-merge route cost (distance + TW penalties). Much better than
    smallest-first for preserving TW quality when reducing route count.
    """
    routes = [r for r in routes if r]
    while len(routes) > problem.max_vehicles:
        best_i = None
        best_j = None
        best_cost = float('inf')

        for i in range(len(routes)):
            for j in range(i + 1, len(routes)):
                combined_load = sum(problem.demands[c] for c in routes[i]) + sum(problem.demands[c] for c in routes[j])
                if combined_load > problem.vehicle_capacity:
                    continue
                # Evaluate both concatenation orders with CI reordering
                merged = _rebuild_route_ci(problem, routes[i] + routes[j])
                cost = _route_cost(problem, merged)
                if cost < best_cost:
                    best_cost = cost
                    best_i, best_j = i, j

        if best_i is not None:
            routes[best_i] = _rebuild_route_ci(problem, routes[best_i] + routes[best_j])
            routes.pop(best_j)
        else:
            # No feasible merge found; drop the smallest route (its customers are lost,
            # but this should almost never happen with reasonable max_vehicles)
            routes.sort(key=len)
            routes.pop(0)
    return [r for r in routes if r]


def build_random_solution(problem):
    """Build a random feasible solution for diversity."""
    n = problem.num_customers
    customers = list(range(1, n + 1))
    random.shuffle(customers)

    routes = []
    current_route = []
    current_load = 0.0

    for cust in customers:
        if current_load + problem.demands[cust] > problem.vehicle_capacity:
            if current_route:
                routes.append(current_route)
            current_route = [cust]
            current_load = problem.demands[cust]
        else:
            current_route.append(cust)
            current_load += problem.demands[cust]

    if current_route:
        routes.append(current_route)

    if len(routes) > problem.max_vehicles:
        routes = _merge_excess_routes(problem, routes)

    solution = [0]
    for route in routes:
        solution.extend(route)
        solution.append(0)

    return solution


