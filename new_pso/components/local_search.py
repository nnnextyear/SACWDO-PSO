import random


def hill_climbing(problem, solution, iterations=200):
    """Simple hill-climbing: only accept improving moves.
    Weaker than SA (no Metropolis acceptance), but provides basic local search."""
    routes = problem.decode_routes(solution)
    best_routes = [r[:] for r in routes]
    best_cost = problem.evaluate(solution)

    for _ in range(iterations):
        improved = False

        for _ in range(10):
            op = random.randint(1, 4)
            new_routes = [r[:] for r in best_routes]

            if op == 1:
                _intra_swap(new_routes)
            elif op == 2:
                _inter_relocate(new_routes, problem)
            elif op == 3:
                _intra_two_opt(new_routes)
            else:
                _inter_cross(new_routes, problem)

            new_sol = _to_sol(new_routes)
            new_cost = problem.evaluate(new_sol)
            if new_cost < best_cost:
                best_routes = new_routes
                best_cost = new_cost
                improved = True

        if not improved:
            break

    return _to_sol(best_routes)


def _intra_swap(routes):
    long_routes = [(i, r) for i, r in enumerate(routes) if len(r) >= 2]
    if not long_routes:
        return
    idx, route = random.choice(long_routes)
    a, b = random.sample(range(len(route)), 2)
    route[a], route[b] = route[b], route[a]


def _inter_relocate(routes, problem):
    if len(routes) < 2:
        return
    non_empty = [(i, r) for i, r in enumerate(routes) if len(r) >= 1]
    if len(non_empty) < 2:
        return
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
        return
    insert_pos = random.randrange(len(dst_route) + 1)
    dst_route.insert(insert_pos, cust)


def _intra_two_opt(routes):
    long_routes = [(i, r) for i, r in enumerate(routes) if len(r) >= 3]
    if not long_routes:
        return
    idx, route = random.choice(long_routes)
    a, b = sorted(random.sample(range(len(route)), 2))
    route[a:b+1] = reversed(route[a:b+1])


def _inter_cross(routes, problem):
    if len(routes) < 2:
        return
    eligible = [(i, r) for i, r in enumerate(routes) if len(r) >= 1]
    if len(eligible) < 2:
        return
    i1, r1 = random.choice(eligible)
    i2 = random.choice([j for j in range(len(routes)) if j != i1])
    r2 = routes[i2]
    pos1 = random.randrange(len(r1))
    pos2 = random.randrange(len(r2))
    tail1, tail2 = r1[pos1:], r2[pos2:]
    r1[pos1:], r2[pos2:] = tail2, tail1
    load1 = sum(problem.demands[c] for c in r1)
    load2 = sum(problem.demands[c] for c in r2)
    if load1 > problem.vehicle_capacity * 1.05 or load2 > problem.vehicle_capacity * 1.05:
        r1[pos1:], r2[pos2:] = tail1, tail2


def _to_sol(routes):
    routes = [r for r in routes if r]
    sol = [0]
    for r in routes:
        sol.extend(r)
        sol.append(0)
    return sol
