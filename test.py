import math
import random
import pulp
import numpy as np


np.random.seed(42)
random.seed(42)


# Sets
P = range(10)
V = range(6)
N = range(2)
K = range(3)

V_name= ["origin", "patient1", "patient2", "patient3", "patient4", "patient5"]
P_name= ["nurse1", "nurse2", "nurse3", "nurse4", "nurse5", "nurse6", "nurse7", "nurse8", "nurse9", "nurse10"]
N_name= ["Type1", "Type2", "Type3", "Type4"]
K_name= ["Institution", "Personal", "Public"]


X = [15, 9, 6.34, 5.3, 14.33, 5.5]
Y = [7.23,11.3, 8.29, 6.63, 6.54, 11.1]


#Desicion variables
x = pulp.LpVariable.dicts("x", (V, V, N, P, K), cat='Binary')
at = pulp.LpVariable.dicts("at", (V, N, P, K), cat='integer')
z = pulp.LpVariable.dicts("z", (N, P), lowBound=0)
w = pulp.LpVariable.dicts("w", (V, N, P, K), lowBound=0)
z_max = pulp.LpVariable("z_max", lowBound=0)
w_prime = pulp.LpVariable.dicts("w_prime", (V, V, range(len(N)),P, K), lowBound=0)


# Parameters
sl = [9, 8]
vc = [400, 300]
v_c = [2.5, 1, 5]
v_s = [.66, 1.2, .75]
v_e = [0.133, 0.069, 0.183]
st = [0,40,50,35,60,50]
ot = [0, 6, 50, 110, 210, 12]
ct = [720, 110, 390, 490, 650, 410]
pr = [[0, 0], [1, 1], [1, 1], [1, 0], [0, 1], [1, 1]]
M = 1000000

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


# Objective Functions

prob += pulp.lpSum(
    [
        (vc[n] + c[i][j][k]) * x[i][j][n][p][k]
        for i in V for j in V for n in N for p in P for k in K
    ]
)

prob += pulp.lpSum(
    [
        e[i][j][k] * x[i][j][n][p][k]
        for i in V for j in V if i!=j for n in range(len(N)) for p in P for k in K
    ]
)

prob += z_max

prob += pulp.lpSum(
    [
     sl[n] * x[i][j][n][p][k]
     for i in V for j in V if i!=j for n in N for p in P for k in K
     ]
)

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
