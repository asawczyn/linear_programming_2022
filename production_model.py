# A factory produces a product over the next K quarters. The demand, production cost,
# storage cost, and maximum production level in the jth quarter equal dj , cj , mj and sj ,
# respectively. The factory wants to minimize total cost and has to meet all demands on
# time. Determine an optimal production plan.

import pyomo.environ as pyo

model = pyo.AbstractModel()

model.K = pyo.Param()  # num of quarters

model.j = pyo.RangeSet(1, model.K)

model.c = pyo.Param(model.j)  # production cost
model.d = pyo.Param(model.j)  # demand
model.m = pyo.Param(model.j)  # storage cost
model.s = pyo.Param(model.j)  # maximum production level

model.x = pyo.Var(model.j, domain=pyo.NonNegativeIntegers)
model.y = pyo.Var(model.j, domain=pyo.NonNegativeIntegers)


model.cost = pyo.Objective(
    rule=lambda m: pyo.summation(m.c, m.x) + pyo.summation(m.m, m.y), sense=pyo.minimize
)


def production_rule(m, j):
    if j == 1:
        return 0 + m.x[j] == m.d[j] + m.y[j]
    else:
        return m.y[j - 1] + m.x[j] == m.d[j] + m.y[j]


model.production = pyo.Constraint(model.j, rule=production_rule)
model.max_production = pyo.Constraint(model.j, rule=lambda m, j: m.x[j] <= m.s[j])

data = {
    None: {
        "K": {None: 4},
        "c": {1: 1, 2: 1, 3: 100, 4: 1},
        "d": {1: 10, 2: 10, 3: 10, 4: 30},
        "m": {1: 10, 2: 10, 3: 10, 4: 10},
        "s": {1: 15, 2: 15, 3: 50, 4: 5},
    }
}

i = model.create_instance(data)

pyo.SolverFactory("glpk").solve(i).write()
i.pprint()

print("Cost = ", i.cost())
print("x = ", i.x.get_values())
print("y = ", i.y.get_values())
