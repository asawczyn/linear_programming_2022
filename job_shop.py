# We are given a set of jobs J = {J1 , . . . , Jn }. Job Ji consists of operations
# Oi1 , . . . Oini , which must be executed in order:
# Oi1 → Oi2 → · · · → Oini .
# Each operation Oij must be executed on a fixed machine from the set
# {M1 , . . . , Mm } and its processing time equals pij . We seek a schedule whose
# makespan (the completion time of the last job) is minimal.

import pandas as pd
import pyomo.environ as pyo
import pyomo.gdp as pyo_gdp
import operator

from utils import plot_gant

model = pyo.AbstractModel()

model.n_jobs = pyo.Param()  # num of jobs
model.n_operations = pyo.Param()  # num of operations of job j
model.ub_s = pyo.Param()  # upper bound of s (starting time)

model.i = pyo.RangeSet(1, model.n_jobs)
model.j = pyo.RangeSet(1, model.n_operations)

model.p = pyo.Param(model.i, model.j)  # processing time
model.m = pyo.Param(model.i, model.j)  # machine

# the starting time of operation Oij on a proper machine.
model.s = pyo.Var(model.i, model.j, domain=pyo.NonNegativeIntegers, bounds=(0, model.ub_s))

model.c = pyo.Var()  # cost
model.cost = pyo.Objective(rule=lambda m: m.c, sense=pyo.minimize)


def preceding_rule(m, i, j):
    if j == 1:
        return pyo.Constraint.Skip
    else:
        return m.s[i, j] >= m.s[i, j - 1] + m.p[i, j - 1]


model.next_starting_time_constraint = pyo.Constraint(model.i, model.j, rule=preceding_rule)


def disjunctions_machines_rule(m, i, j, k, l):
    if m.m[i, j] == m.m[k, l] and (i, j) != (k, l):
        return [
            [m.s[i, j] >= m.s[k, l] + m.p[k, l]],
            [m.s[k, l] >= m.s[i, j] + m.p[i, j]],
        ]
    else:
        return pyo.Constraint.Skip


model.disjunctions_machines = pyo_gdp.Disjunction(
    model.i, model.j, model.i, model.j, rule=disjunctions_machines_rule
)

model.cost_contraint = pyo.Constraint(
    model.i, model.j, rule=lambda m, i, j: m.c >= m.s[i, j] + m.p[i, j]
)


data = {
    None: {
        "n_jobs": {None: 3},
        "n_operations": {None: 3},
        "ub_s": {None: 1000},
        "p": {
            (1, 1): 3,
            (1, 2): 2,
            (1, 3): 2,
            # -------
            (2, 1): 2,
            (2, 2): 1,
            (2, 3): 0,  # placeholder with 0 processing time and machine -1
            # -------
            (3, 1): 4,
            (3, 2): 2,
            (3, 3): 3,
        },
        "m": {
            (1, 1): 1,
            (1, 2): 2,
            (1, 3): 3,
            # -------
            (2, 1): 2,
            (2, 2): 1,
            (2, 3): -1,  # placeholder with 0 processing time and machine -1
            # -------
            (3, 1): 1,
            (3, 2): 2,
            (3, 3): 3,
        },
    }
}


i = model.create_instance(data)
i.pprint()
pyo.TransformationFactory("gdp.hull").apply_to(i)
i.pprint()
pyo.SolverFactory("glpk").solve(i).write()

df = pd.DataFrame()
df["i,j"] = data[None]["m"].keys()
df["job"] = df["i,j"].map(operator.itemgetter(0))
df["operation"] = df["i,j"].map(operator.itemgetter(1))
df["processing_time"] = data[None]["p"].values()
df["machine"] = data[None]["m"].values()
df["starting_time"] = df["i,j"].map(i.s.get_values())

print("\n")
print(df)
print("\n")
print("Cost = ", i.cost())
print("s = ", i.s.get_values())

plot_gant(df)
