from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm


def energy(w, s):
    return norm(w)**2/2 - 1/norm(s)

def energy_error(Energies, E_0):
    return np.abs((Energies-E_0)/E_0)



def turn_vectors_into_2_arrays(List):
    x_list = []
    y_list = []
    for i in List:
        x_list.append(i[0])
        y_list.append(i[1])
    return x_list, y_list

def calculate(s_0, w_0, h, steps):
    S = []
    W = []
    E = []
    for i in range(steps):
        s_i = s_0 + w_0*h
        w_i = w_0 - (s_0/norm(s_0)**3) * h
        W.append(w_i)
        S.append(s_i)
        e = energy(w_i, s_i)
        E.append(e)
        s_0 = s_i
        w_0 = w_i
    x_s, y_s = turn_vectors_into_2_arrays(S)
    return x_s, y_s, E

w = np.array([0, 1])
s = np.array([1, 0])

steps = 670

w = np.array([0, 0.6])
s = np.array([1, 0])
steps = 200
stepsize = 0.01

plt.figure(figsize=(11,8))

x_s, y_s, E = calculate(s, w, stepsize, steps)
rel_E_err = energy_error(E, energy(w,s))
delta_t = np.linspace(0.001, 1)
zero = np.zeros(len(x_s))
plt.xlim(-1.5, 1.5)
plt.ylim(-1.5, 1.5)
#plt.quiver(zero, zero, x_s, y_s, scale=1, angles="xy", scale_units="xy")
plt.scatter(x_s, y_s, c="gray", marker=".")
plt.gca().add_artist(plt.Circle((0, 0), 1, fill=False, color="r"))
#plt.show()
steps_array = np.linspace(0, steps, steps)
plt.figure(figsize=(11,8))
plt.plot(steps_array, rel_E_err)


plt.show()
