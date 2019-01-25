from collections import Counter
from collections import defaultdict
from ortools.linear_solver import pywraplp
import numpy as np

def euclidean_distance(x, y):
    return sum((a - b)**2 for (a, b) in zip(x, y))

def earthmover_distance(p1, p2):
    dist1 = {x: count / len(p1) for (x, count) in Counter(p1).items()}
    dist2 = {x: count / len(p2) for (x, count) in Counter(p2).items()}
    solver = pywraplp.Solver('earthmover_distance', pywraplp.Solver.GLOP_LINEAR_PROGRAMMING)

    variables = dict()

    # for each pile in dist1, the constraint that says all the dirt must leave this pile
    dirt_leaving_constraints = defaultdict(lambda: 0)

    # for each hole in dist2, the constraint that says this hole must be filled
    dirt_filling_constraints = defaultdict(lambda: 0)

    # the objective
    objective = solver.Objective()
    objective.SetMinimization()

    for (x, dirt_at_x) in dist1.items():
        for (y, capacity_of_y) in dist2.items():
            amount_to_move_x_y = solver.NumVar(0, solver.infinity(), 'z_{%s, %s}' % (x, y))
            variables[(x, y)] = amount_to_move_x_y
            dirt_leaving_constraints[x] += amount_to_move_x_y
            dirt_filling_constraints[y] += amount_to_move_x_y
            objective.SetCoefficient(amount_to_move_x_y, euclidean_distance(x, y))

    for x, linear_combination in dirt_leaving_constraints.items():
        solver.Add(linear_combination == dist1[x])

    for y, linear_combination in dirt_filling_constraints.items():
        solver.Add(linear_combination == dist2[y])

    status = solver.Solve()
    if status not in [solver.OPTIMAL, solver.FEASIBLE]:
        raise Exception('Unable to find feasible solution')

    return objective.Value()


p1 = np.array([1,1,2,2])
p2 = np.array([1,1,2,3])

print earthmover_distance(p1, p2)