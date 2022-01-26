# We are given a graph G = (V , E). We seek a partition of the node set V into V1 and V2 ,
# so that the number of edges between the nodes in V1 and the nodes in V2 is maximal.

import pyomo.environ as pyo

from utils import plot_graph_cut

model = pyo.AbstractModel()

model.n_nodes = pyo.Param()  # num of quarters

model.i = pyo.RangeSet(1, model.n_nodes)

model.E = pyo.Set(dimen=2)  # set of edges

model.x = pyo.Var(model.i, domain=pyo.Binary)
model.y = pyo.Var(model.E, domain=pyo.Binary)  # y = x_i*x_j


def z_rule(m):
    return sum(m.x[i] - 2 * m.y[(i, j)] + m.x[j] for i, j in m.E)


model.z = pyo.Objective(sense=pyo.maximize, rule=z_rule)

model.y_constraint_i = pyo.Constraint(model.E, rule=lambda m, i, j: m.y[(i, j)] <= m.x[i])
model.y_constraint_j = pyo.Constraint(model.E, rule=lambda m, i, j: m.y[(i, j)] <= m.x[j])
model.y_constraint_ij = pyo.Constraint(
    model.E, rule=lambda m, i, j: m.y[(i, j)] >= m.x[i] + m.x[j] - 1
)

data = {
    None: {
        "n_nodes": {None: 6},
        "E": {None: (1, 6, 1, 3, 1, 2, 2, 3, 2, 4, 3, 4, 4, 5, 5, 3, 5, 6, 6, 3)},
    }
}
i = model.create_instance(data)
i.pprint()

solutions = pyo.SolverFactory("glpk").solve(i)  # .write()
i.solutions.store_to(solutions)
print(solutions)
i.display()
i.pprint()
print(i.z())
print(i.x.get_values())
print(i.y.get_values())

plot_graph_cut(i.x.get_values(), i.E)
