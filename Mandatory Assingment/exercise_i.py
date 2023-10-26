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


X = np.zeros(n + 1, dtype=int)  # The component state variables (X[0] is not used)

p = np.linspace(0, 1, num_points)
T = np.zeros(num_points, dtype=int)

for _ in range(num_sims):
    U = gen.uniform(0.0, 1.0, n + 1)  # Uniform variables (U[0] is not used)
    for j in range(num_points):
        for i in range(1, n + 1):
            if U[i] <= p[j]:
                X[i] = 1
            else:
                X[i] = 0
        T[j] += phi(X)

h_hat = [T[i] / num_sims for i in range(num_points)]

h = np.zeros(num_points)  # True reliability function

for j in range(num_points):
    h[j] = hh(p[j])


plt.plot(p, h, color="blue", label="h(p)")
plt.plot(p, h_hat, color="red", label="h_hat(p)")
plt.xlabel("p")
plt.ylabel("h")
plt.title("Reliability function of " + sys_name + " system")
plt.legend()
if save_plot:
    plt.savefig("crude/" + sys_name + ".pdf")
plt.show()
