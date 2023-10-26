import matplotlib.pyplot as plt
import numpy as np

plt.style.use("Solarize_Light2")

# Number of simulations
num_sims = 1000

# Interval between h_hat calculations
h_interv = 10

# Random number generator seed. Set to 'None' for a random sequence
seed_num = 1234
gen = np.random.default_rng(seed=seed_num)

# Save plot to file or not?
save_plot = True

# Name of system
sys_name = "bridge"

# Number of components:
n = 8

# Length of shortest path
d = 3

# Length of shortest cut
c = 2

# Component reliabilities
px = [
    0,
    0.6,
    0.3,
    0.5,
    0.4,
    0.7,
    0.5,
    0.3,
    0.6,
]  # P(X_i = px[i]), i = 1, .., n. px[0] is not used


def coprod(x, y):
    return x + y - x * y


def phi(xx):
    system = (
        xx[3]
        * xx[6]
        * coprod(xx[1], xx[2])
        * coprod(xx[4], xx[5])
        * coprod(xx[7], xx[8])
    )

    +((1 - xx[3]) * xx[6] * coprod(xx[1] * xx[4], xx[2] * xx[5]) * coprod(xx[7], xx[8]))

    +((1 - xx[6]) * xx[3] * coprod(xx[1], xx[2]) * coprod(xx[4] * xx[7], xx[5] * xx[8]))

    +((1 - xx[3]) * (1 - xx[6]) * coprod(xx[1] * xx[4] * xx[7], xx[2] * xx[5] * xx[8]))
    return system


def hh(pp):
    rel = (
        pp[3]
        * pp[6]
        * coprod(pp[1], pp[2])
        * coprod(pp[4], pp[5])
        * coprod(pp[7], pp[8])
    )

    +((1 - pp[3]) * pp[6] * coprod(pp[1] * pp[4], pp[2] * pp[5]) * coprod(pp[7], pp[8]))

    +((1 - pp[6]) * pp[3] * coprod(pp[1], pp[2]) * coprod(pp[4] * pp[7], pp[5] * pp[8]))

    +((1 - pp[3]) * (1 - pp[6]) * coprod(pp[1] * pp[4] * pp[7], pp[2] * pp[5] * pp[8]))
    return rel


# Compute the distributions of S_1, ... , S_n, where S_m = X_m + ... + X_n, m = 1, ..., n
ps = np.zeros(
    [n + 1, n + 1]
)  # P(S_m = s) = ps[m,s], s = 0, 1, ..., (n-m+1). ps[0,s] is not used

ps[n, 0] = 1.0 - px[n]  # P(S_n = 0) = 1 - P(X_n = 1)
ps[n, 1] = px[n]  # P(S_n = 1) = P(X_n = 1)

for j in range(1, n):
    m = n - j
    ps[m, 0] = ps[m + 1, 0] * (1.0 - px[m])  # P(S_m = 0) = P(S_{m+1} = 0) * P(X_m = 0)

    for s in range(1, n - m + 1):
        ps[m, s] = ps[m + 1, s - 1] * px[m] + ps[m + 1, s] * (
            1.0 - px[m]
        )  # P(S_m = s) = P(S_{m+1} = s-1) * P(X_m = 1)
        # + P(S_{m+1} = s) * P(X_m = 0)

    ps[m, n - m + 1] = (
        ps[m + 1, n - m] * px[m]
    )  # P(S_m = n-m+1) = P(S_{m+1} = n-m) * P(X_m = 1)

# Print the distribution of S = S_1
for s in range(n + 1):
    print("P(S = " + str(s) + ") =", ps[1, s])

# Calculate pdc = P(d <= S_1 <= n-c)
pdc = 0
for s in range(d, n - c + 1):
    pdc += ps[1, s]

# Calculate pcn = P(n-c < S_1 <= n)
pcn = 0
for s in range(n - c + 1, n + 1):
    pcn += ps[1, s]


# Sample S_1 from the set {d, ... , n-c}
def sampleS():
    u = gen.uniform(0.0, pdc)
    for s in range(d, n - c):
        if u < ps[1, s]:
            return s
        else:
            u -= ps[1, s]
    return n - c


X = np.zeros(n + 1, dtype=int)  # The component state variables (X[0] is not used)
T = np.zeros(
    n + 1, dtype=int
)  # T[s] counts random path sets of size s = d, 1, ..., n-c
V = np.zeros(n + 1, dtype=int)  # V[s] counts random sets of size s = d, 1, ..., n-c

I = []
H = []
H_hat = []

# Calculate the true system reliability
h = hh(px)

# Run the simulations
for sim in range(num_sims):
    s = sampleS()
    V[s] += 1
    U = gen.uniform(0.0, 1.0, n + 1)  # Uniform variables (U[0] is not used)
    sumx = 0
    for m in range(1, n):
        if sumx < s:
            p = px[m] * ps[m + 1, s - sumx - 1] / ps[m, s - sumx]
            if U[m] <= p:
                X[m] = 1
            else:
                X[m] = 0
            sumx += X[m]
        else:
            X[m] = 0
    if sumx < s:
        X[n] = 1
    else:
        X[n] = 0
    T[s] += phi(X)
    if sim > 0 and sim % h_interv == 0:
        h_hat = pcn
        for s in range(d, n - c + 1):
            if V[s] > 0:
                h_hat += ps[1, s] * T[s] / V[s]
        I.append(sim)
        H.append(h)
        H_hat.append(h_hat)

# Print the estimated conditional reliabilities, theta_s, s = d, ..., n-c
for s in range(d, n - c + 1):
    print("theta_" + str(s) + " =", T[s] / V[s])

# Estimate final system reliability
h_hat = pcn

for s in range(d, n - c + 1):
    h_hat += ps[1, s] * T[s] / V[s]

print("h = ", h, ", h_hat = ", h_hat)

plt.plot(I, H, color="blue", label="True h")
plt.plot(I, H_hat, color="red", label="h_hat(iteration)")
plt.ylim(h - 0.2, h + 0.2)
plt.xlabel("iteration")
plt.ylabel("h")
plt.title("Convergence curve for " + sys_name + " system")
plt.legend()
if save_plot:
    plt.savefig("cmcgen/" + sys_name + ".pdf")
plt.show()
