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

# Component reliabilities
px = [
    0.0,
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


X = np.zeros(n + 1, dtype=int)  # The component state variables (X[0] is not used)

I = []
H = []
H_hat = []
T = 0

# Calculate the true system reliability
h = hh(px)

for sim in range(num_sims):
    U = gen.uniform(0.0, 1.0, n + 1)  # Uniform variables (U[0] is not used)
    for m in range(1, n + 1):
        if U[m] <= px[m]:
            X[m] = 1
        else:
            X[m] = 0
    T += phi(X)
    if sim > 0 and sim % h_interv == 0:
        h_hat = T / sim
        I.append(sim)
        H.append(h)
        H_hat.append(h_hat)

# Estimate final system reliability
h_hat = T / num_sims

print("h = ", h, ", h_hat = ", h_hat)

plt.plot(I, H, "b", label="True h")
plt.plot(I, H_hat, "r", label="h_hat(iteration)")

plt.ylim(h - 0.2, h + 0.2)
plt.xlabel("iteration")
plt.ylabel("h")

plt.title("Convergence curve for " + sys_name + " system")
plt.legend()
if save_plot:
    plt.savefig("crudegen/" + sys_name + ".pdf")
plt.show()
