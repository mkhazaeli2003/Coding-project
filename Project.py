import math
import random
import pulp
import numpy as np
import copy


np.random.seed(42)
random.seed(42)

#Finding epsilons

def calculate_ideal_points(prob, objectives):
    ideal_points = []
    for obj in objectives:
        prob_obj = copy.deepcopy(prob)
        prob_obj += obj
        prob_obj.solve()
        if prob_obj.status == pulp.LpStatusOptimal:
            ideal_points.append(pulp.value(obj))
        else:
            print(f"Problem infeasible for objective: {obj}")
            ideal_points.append(float('inf'))  # Assign infinity for infeasible cases
    return ideal_points


def calculate_nadir_points(prob, objectives):
    nadir_points = []
    for obj in objectives:
        prob_obj = copy.deepcopy(prob)
        negated_obj = -obj  # Negate the objective
        prob_obj += negated_obj
        prob_obj.solve()
        if prob_obj.status == pulp.LpStatusOptimal:
            nadir_points.append(-pulp.value(obj))  # Negate back the value
        else:
            print(f"Problem infeasible for objective: {obj}")
            nadir_points.append(float('-inf'))  # Assign -inf for infeasible cases
    return nadir_points

# Sets
P = range(10)
V = range(7)
N = range(4)
K = range(3)

V_name= ["origin", "patient1", "patient2", "patient3", "patient4", "patient5", "destination"]
P_name= ["nurse1", "nurse2", "nurse3", "nurse4", "nurse5", "nurse6", "nurse7", "nurse8", "nurse9", "nurse10"]
N_name= ["Type1", "Type2", "Type3", "Type4"]
K_name= ["Institution", "Personal", "Public"]


X = [random.randint(5, 15) for _ in V]
Y = [random.randint(5, 15) for _ in V]


#Desicion variables
x = pulp.LpVariable.dicts("x", (V, V, N, P, K), cat='Binary')
at = pulp.LpVariable.dicts("at", (V, N, P, K), cat='integer')
z = pulp.LpVariable.dicts("z", (N, P), lowBound=0)
w = pulp.LpVariable.dicts("w", (V, N, P, K), lowBound=0)
z_max = pulp.LpVariable("z_max", lowBound=0)
w_prime = pulp.LpVariable.dicts("w_prime", (V, V, range(len(N)),P, K), lowBound=0)


# Parameters
sl = [8, 7, 6, 5]
vc = [300000, 200000, 100000, 75000]
v_c = [200, 400, 50]
v_s = [60, 75, 45]
v_e = [0.075, 0.100, 0.046]
st = [random.randint(30, 100) for _ in V]
print(st)
ot = [random.randint(0, 520) for _ in V]
print(ot)
pr = np.random.randint(0, 2, size=(len(V), len(set(N)))).tolist()
M = 100000

d = [[0] * len(V) for _ in range(len(V))]
for i in V:
    for j in V:
        d[i][j] = math.sqrt((X[i] - X[j])**2 + (Y[i] - Y[j])**2)


ct = [0] * len(V)
for i in V:
    ct[i] = ot[i] + random.uniform(50, 200)


c = [[[0] * len(K) for _ in range(len(V))] for _ in range(len(V))]
for i in V:
    for j in V:
        for k in K:
            c[i][j][k] = int(v_c[k] * d[i][j])


e = [[[0] * len(K) for _ in range(len(V))] for _ in range(len(V))]
for i in V:
    for j in V:
        for k in K:
            e[i][j][k] = int(v_e[k] * d[i][j])

t = [[[0] * len(K) for _ in range(len(V))] for _ in range(len(V))]
for i in V:
    for j in V:
        for k in K:
            t[i][j][k] = int(v_s[k] * d[i][j])





#  PuLP
prob = pulp.LpProblem("Minimize_cost", pulp.LpMinimize)


# Objectives
objectives = [
    pulp.lpSum([(vc[n] + c[i][j][k]) * x[i][j][n][p][k] for i in V for j in V for n in N for p in P for k in K]),
    pulp.lpSum([e[i][j][k] * x[i][j][n][p][k] for i in V for j in V if i != j for n in N for p in P for k in K]),
    z_max,
    pulp.lpSum([sl[n] * x[i][j][n][p][k] for i in V for j in V if i != j for n in N for p in P for k in K])
]




#Constraints

prob += (pulp.lpSum(
    [x[0][j][n][p][k]
    for j in V[1:] for n in N for p in P for k in K])
    >= 1)


prob += (pulp.lpSum(
    [x[j][len(V)-1][n][p][k]
    for j in V[:-1] for n in N for p in P for k in K])
    >= 1)


for i in V[1:-1]:
    prob += (pulp.lpSum(
        [x[i][j][n][p][k]
        for j in V if i!=j for n in N for p in P for k in K])
        == 1)


for i in V[1:-1]:
    for j in V[1:-1]:
        if i != j:
            prob += (pulp.lpSum(
                [x[i][j][n][p][k]
                for n in N for p in P for k in K])
                <= 1)


for p in P[1:-1]:
    prob += (pulp.lpSum(
        [x[0][j][n][p][k]
        for j in V for n in N for p in P for k in K])
        <= 1)


for i in V:
    for n in N:
        prob += (pulp.lpSum(
            [x[i][j][n][p][k]
            for j in V if i!=j for p in P for k in K])
            <= pr[i][n])


prob += (pulp.lpSum(
    [x[0][j][n][p][k]
    for j in V[1:] for n in N for p in P for k in K])
    == pulp.lpSum(
    [x[j][0][n][p][k]
    for j in V[:-1] for n in N for p in P for k in K]))


for h in V[1:-1]:
    for n in N:
        for p in P:
            for k in K:
                 prob += (pulp.lpSum(
                     [x[i][h][n][p][k]
                     for i in V if i!=h])
                     == pulp.lpSum(
                     [x[h][j][n][p][k]
                     for j in V if j!=h]))

for i in V:
    for j in V:
        if i != j:
            for n in N:
                for p in P:
                    for k in K:
                        prob += at[i][n][p][k] + w[i][n][p][k] + st[i] + t[i][j][k] <= M * (1 - x[i][j][n][p][k]) + at[j][n][p][k]

for i in V:
    for n in N:
        for p in P:
            for k in K:
                prob += ot[i] <= at[i][n][p][k] + w[i][n][p][k]

for i in V:
    for n in N:
        for p in P:
            for k in K:
                prob += at[i][n][p][k] + w[i][n][p][k] + st[i] <= ct[i]

for n in N:
    for p in P:
        for k in K:
            prob += at[0][n][p][k] == 0

for i in V:
    for j in V:
        if i != j:
            for n in range(len(N)):
                for p in P:
                    for k in K:
                        prob += w_prime[i][j][n][p][k] <= w[i][n][p][k]

                        prob += w_prime[i][j][n][p][k] <= M * x[i][j][n][p][k]

                        prob += w[i][n][p][k] - w_prime[i][j][n][p][k] <= M * (1 - x[i][j][n][p][k])


for n in range(len(N)):
    for p in P:
        prob += z[n][p] == pulp.lpSum(
            [
                (st[i] + t[i][j][k]) * x[i][j][n][p][k] + w_prime[i][j][n][p][k]
                for i in V for j in V if i!=j for k in K
            ]
        )

for n in range(len(N)):
    for p in P:
        prob += z[n][p] <= z_max


ideal_points = calculate_ideal_points(prob, objectives)
nadir_points = calculate_nadir_points(prob, objectives)



#Finding pareto solutions using e-constraint

prob += pulp.lpSum(
    [
        (vc[n] + c[i][j][k]) * x[i][j][n][p][k]
        for i in V for j in V for n in N for p in P for k in K
    ]
)


b = c = d =  4
while b >= 0 :
    eps1 = nadir_points[1]-(nadir_points[1]-ideal_points[1] / 4 * b)
    while c >= 0:
        eps2 = nadir_points[2]-(nadir_points[2]-ideal_points[2] / 4 * c)
        while d >= 0:
            eps3 =  nadir_points[3]-(nadir_points[3]-ideal_points[3] / 4 * d)
            prob += pulp.lpSum(
                [
                    e[i][j][k] * x[i][j][n][p][k]
                    for i in V for j in V if i!=j for n in range(len(N)) for p in P for k in K
                ]
            ) <= eps1

            prob += z_max <= eps2

            prob += pulp.lpSum(
                [
                 sl[n] * x[i][j][n][p][k]
                 for i in V for j in V if i!=j for n in N for p in P for k in K
                 ]
            ) <= eps3

            # Constraints

            prob += (pulp.lpSum(
                [x[0][j][n][p][k]
                 for j in V[1:] for n in N for p in P for k in K])
                     >= 1)

            prob += (pulp.lpSum(
                [x[j][len(V) - 1][n][p][k]
                 for j in V[:-1] for n in N for p in P for k in K])
                     >= 1)

            for i in V[1:-1]:
                prob += (pulp.lpSum(
                    [x[i][j][n][p][k]
                     for j in V if i != j for n in N for p in P for k in K])
                         == 1)

            for i in V[1:-1]:
                for j in V[1:-1]:
                    if i != j:
                        prob += (pulp.lpSum(
                            [x[i][j][n][p][k]
                             for n in N for p in P for k in K])
                                 <= 1)

            for p in P[1:-1]:
                prob += (pulp.lpSum(
                    [x[0][j][n][p][k]
                     for j in V for n in N for p in P for k in K])
                         <= 1)

            for i in V:
                for n in N:
                    prob += (pulp.lpSum(
                        [x[i][j][n][p][k]
                         for j in V if i != j for p in P for k in K])
                             <= pr[i][n])

            prob += (pulp.lpSum(
                [x[0][j][n][p][k]
                 for j in V[1:] for n in N for p in P for k in K])
                     == pulp.lpSum(
                        [x[j][0][n][p][k]
                         for j in V[:-1] for n in N for p in P for k in K]))

            for h in V[1:-1]:
                for n in N:
                    for p in P:
                        for k in K:
                            prob += (pulp.lpSum(
                                [x[i][h][n][p][k]
                                 for i in V if i != h])
                                     == pulp.lpSum(
                                        [x[h][j][n][p][k]
                                         for j in V if j != h]))

            for i in V:
                for j in V:
                    if i != j:
                        for n in N:
                            for p in P:
                                for k in K:
                                    prob += at[i][n][p][k] + w[i][n][p][k] + st[i] + t[i][j][k] <= M * (
                                                1 - x[i][j][n][p][k]) + at[j][n][p][k]

            for i in V:
                for n in N:
                    for p in P:
                        for k in K:
                            prob += ot[i] <= at[i][n][p][k] + w[i][n][p][k]

            for i in V:
                for n in N:
                    for p in P:
                        for k in K:
                            prob += at[i][n][p][k] + w[i][n][p][k] + st[i] <= ct[i]

            for n in N:
                for p in P:
                    for k in K:
                        prob += at[0][n][p][k] == 0

            for i in V:
                for j in V:
                    if i != j:
                        for n in range(len(N)):
                            for p in P:
                                for k in K:
                                    prob += w_prime[i][j][n][p][k] <= w[i][n][p][k]

                                    prob += w_prime[i][j][n][p][k] <= M * x[i][j][n][p][k]

                                    prob += w[i][n][p][k] - w_prime[i][j][n][p][k] <= M * (1 - x[i][j][n][p][k])

            for n in range(len(N)):
                for p in P:
                    prob += z[n][p] == pulp.lpSum(
                        [
                            (st[i] + t[i][j][k]) * x[i][j][n][p][k] + w_prime[i][j][n][p][k]
                            for i in V for j in V if i != j for k in K
                        ]
                    )

            for n in range(len(N)):
                for p in P:
                    prob += z[n][p] <= z_max

            prob.solve()

            print("Status:", pulp.LpStatus[prob.status])
            if prob.status == pulp.LpStatusOptimal:
                print("Objective Value:", pulp.value(prob.objective))
                for i in V:
                    for j in V:
                        if i != j:
                            for n in N:
                                for p in P:
                                    for k in K:
                                        if x[i][j][n][p][k].varValue > 0:
                                            print(
                                                f"{P_name[p]} travels from {V_name[i]} to {V_name[j]} "
                                                f"using {K_name[k]} in {N_name[n]}: {x[i][j][n][p][k].varValue}"
                                            )
            else:
                print("No optimal solution found.")

            d -= 1
        c -= 1
    b -= 1

