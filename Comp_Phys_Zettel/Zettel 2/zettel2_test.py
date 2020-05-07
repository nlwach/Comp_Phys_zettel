from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm

"""
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

#plt.figure(figsize=(11,8))

#x_s, y_s, E = calculate(s, w, stepsize, steps)
#rel_E_err = energy_error(E, energy(w,s))
#delta_t = np.linspace(0.001, 1)
#zero = np.zeros(len(x_s))
#plt.xlim(-1.5, 1.5)
#plt.ylim(-1.5, 1.5)
#plt.quiver(zero, zero, x_s, y_s, scale=1, angles="xy", scale_units="xy")
#plt.scatter(x_s, y_s, c="gray", marker=".")
#plt.gca().add_artist(plt.Circle((0, 0), 1, fill=False, color="r"))
#plt.show()
#steps_array = np.linspace(0, steps, steps)
#plt.figure(figsize=(11,8))
#plt.plot(steps_array, rel_E_err)


def calc_abs_energy_err(s, w, h):
    energy_error_time_step = []
    for t in h:
        _, _, E = calculate(s, w, t, steps)
        E_err = np.abs((E[-1]-energy(w,s))/energy(w,s))
        energy_error_time_step.append(E_err)
    return energy_error_time_step

w = np.array([0, 1])
s = np.array([1, 0])
stepsize = 0.001
steps = 800
h = np.linspace(0.001, 0.9, 300)


plt.figure(figsize=(11,8))
plt.title("Time Step Vectors")
plt.xlabel("X-Position")
plt.ylabel("Y-Position")
plt.yscale("log")
plt.xscale("log")
rel_E_err = calc_abs_energy_err(s, w, h)
plt.plot(h, rel_E_err)

print(rel_E_err)
plt.show()

import time


import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm


class Euler():
    def __init__(self, r_0, v_0, h, steps, threed=False):
        self.s_0 = np.array(r_0 / norm(r_0))
        self.w_0 = np.array(v_0 / np.sqrt(1 / norm(r_0)))
        self.h = h
        self.location = [self.s_0]
        self.velocity = [self.w_0]
        self.energies = [1 / 4 * norm(self.w_0) ** 2 - 1 / norm(self.s_0)]
        self.steps = steps
        self.y_location = []
        self.x_location = []
        self.z_location = []
        self.threed = threed
        self.rel_Error = []

    def loc(self, i):
        location_i = self.location[i - 1] + (self.velocity[i - 1] * self.h)
        self.location.append(location_i)

    def vel(self, i):
        velocity_i = self.velocity[i - 1] - ((self.location[i - 1] / norm(self.location[i - 1]) ** 3) * self.h)
        self.velocity.append(velocity_i)

    def energy(self, i):
        e = 1 / 4 * norm(self.velocity[i]) ** 2 - 1 / norm(self.location[i])
        self.energies.append(e)

    def split(self):
        if self.threed == False:
            for location in self.location:
                self.x_location.append(location[0])
                self.y_location.append(location[1])
        else:
            for location in self.location:
                self.x_location.append(location[0])
                self.y_location.append(location[1])
                self.z_location.append(location[2])

    def calc_rel_Error(self, i):
        rel_E_i = np.abs((self.energies[0] - self.energies[i]) / self.energies[0])
        self.rel_Error.append(rel_E_i)

    def calc(self):
        for i in range(1, self.steps):
            loc = self.loc(i)
            self.vel(i)
            self.energy(i)
            self.calc_rel_Error(i)

        self.split()


A = Euler([1, 0], [0, 1], 0.01, 5)
A.calc()
print(A.x_location, A.y_location)
print(A.steps, A.rel_Error)

plt.plot(np.linspace(0, A.steps, len(A.rel_Error)), A.rel_Error)
plt.show()
"""


class Euler():
    def __init__(self, r_0, v_0, h, steps, threed=False):
        self.s_0 = np.array(r_0 / norm(r_0))
        self.w_0 = np.array(v_0 / np.sqrt(1 / norm(r_0)))
        self.h = h
        self.location = [self.s_0]
        self.velocity = [self.w_0]
        self.energies = [1 / 4 * norm(self.w_0) ** 2 - 1 / norm(self.s_0)]
        self.steps = steps
        self.y_location = []
        self.x_location = []
        self.z_location = []
        self.threed = threed
        self.rel_Error = []

    def loc(self, i):
        location_i = self.location[i - 1] + (self.velocity[i - 1] * self.h)
        self.location.append(location_i)

    def vel(self, i):
        velocity_i = self.velocity[i - 1] - ((self.location[i - 1] / norm(self.location[i - 1]) ** 3) * self.h)
        self.velocity.append(velocity_i)

    def vel_half(self, i):
        velocity_i = self.velocity[i - 1] - ((self.location[i - 1] / norm(self.location[i - 1]) ** 3) * self.h * 0.5)
        self.velocity.append(velocity_i)

    def energy(self, i):
        e = 1 / 4 * norm(self.velocity[i]) ** 2 - 1 / norm(self.location[i])
        self.energies.append(e)

    def split(self):
        if self.threed == False:
            for location in self.location:
                self.x_location.append(location[0])
                self.y_location.append(location[1])
        else:
            for location in self.location:
                self.x_location.append(location[0])
                self.y_location.append(location[1])
                self.z_location.append(location[2])

    def calc_rel_Error(self, i):
        rel_E_i = np.abs((self.energies[0] - self.energies[i]) / self.energies[0])
        self.rel_Error.append(rel_E_i)

    def calc(self):
        for i in range(1, self.steps):
            self.loc(i)
            self.vel(i)
            self.energy(i)
            self.calc_rel_Error(i)

        self.split()

    def calc_leap(self):
        for i in range(1, self.steps):
            if i == 1:
                self.vel_half(i)
            else:
                self.vel(i)
            self.loc(i)
            self.energy(i)
            self.calc_rel_Error(i)
        self.split()






D = Euler([1, 0], [0,1], 0.01, 1000, threed=False)
D.calc_leap()
plt.figure(figsize=(8,8))


plt.scatter(D.x_location, D.y_location, c="red", cmap='RdYlGn_r', marker=".")

plt.show()

