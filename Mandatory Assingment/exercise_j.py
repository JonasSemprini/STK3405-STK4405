from scipy.stats import binom
import matplotlib.pyplot as plt
import numpy as np

plt.style.use("Solarize_Light2")
# Number of simulations
num_sims = 1000

# Random number generator seed. Set to 'None' for a random sequence
seed_num = 1234
gen = np.random.default_rng(seed=seed_num)

# Number of points
num_points = 101

# Save plot to file or not?
save_plot = True

# Name of system
sys_name = "bridge"

# Number of components:
n = 8


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
    rel = pp**2 * coprod(pp, pp) * coprod(pp, pp) * coprod(pp, pp)

    +(2 * pp * (1 - pp) * coprod(pp**2, pp**2) * coprod(pp, pp))

    +((1 - pp) ** 2 * coprod(pp**3, pp**3))
    return rel


C = list(range(1, n + 1))  # The component set [1, 2, ..., n]
X = np.zeros(n + 1, dtype=int)  # The component state variables (X[0] is not used)
T = np.zeros(
    n + 1, dtype=int
)  # T[s] counts the number of random path sets of size s = 0, 1, ..., n

for _ in range(num_sims):
    for i in range(1, n + 1):
        X[i] = 0
    sys_state = phi(
        X
    )  # phi(0,...0) will always be zero unless we have a trivial system
    T[0] += sys_state
    gen.shuffle(C)  # Generate a random permutation of the component set C
    for i in range(1, n + 1):
        X[C[i - 1]] = 1
        if (
            sys_state == 0
        ):  # If sys_state = 1 already, we know phi(X) = 1 since phi is non-decreasing
            sys_state = phi(X)
        T[i] += sys_state

s_values = list(range(n + 1))  # Set of possible values of S = X[1] + ... + X[n]

theta_hat = [
    T[s] / num_sims for s in s_values
]  # theta_hat[s] = Estimated conditional reliability given S = s
print(theta_hat)

p = np.linspace(0, 1, num_points)
h = np.zeros(num_points)  # True reliability function
h_hat = np.zeros(num_points)  # Estimated reliability function
for j in range(num_points):
    h[j] = hh(p[j])
    dist = [binom.pmf(s, n, p[j]) for s in s_values]
    for s in range(n + 1):
        h_hat[j] += theta_hat[s] * dist[s]

plt.plot(p, h, color="blue", label="h(p)")
plt.plot(p, h_hat, color="red", label="h_hat(p)")
plt.xlabel("p")
plt.ylabel("h")
plt.title("Reliability function of " + sys_name + " system")
plt.legend()
if save_plot:
    plt.savefig("cmc/" + sys_name + ".pdf")
plt.show()
