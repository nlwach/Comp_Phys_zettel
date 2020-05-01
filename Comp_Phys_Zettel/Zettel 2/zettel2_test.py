from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm


def s_i(vector_s_i_1, vector_w_i_1, h):
    return vector_s_i_1 + vector_w_i_1*h

def w_i(vector_s_i_1 ,vector_w_i_1, h):
    return vector_w_i_1 - vector_s_i_1/norm(vector_s_i_1)**3*h


def calc(vector_s_0, vector_w_0, h, steps):
    for step in range(steps):
        W.append(vector_w_0)
        S.append(vector_s_0)
        e = energy(vector_w_0, vector_s_0)
        E.append(e)
        value_vector_s_0 = s_i(vector_s_0, vector_w_0, h)
        vector_w_0 = w_i(vector_s_0, vector_w_0, h)
        vector_s_0 = value_vector_s_0

def energy(w, s):
    return norm(w)**2/2 - 1/norm(s)

v = np.array([0, 1])
s = np.array([1, 0])

S = []
W = []
E = []

e_1 = energy(v, s)
w = w_i(s, v, 0.1)
s = s_i(s, v, 0.1)
e_2 = energy(w, s)
#print(s)
#print(w)
#print(e_1)
#print(e_2)
#calc(s, v, 100, 100)
print(S)
print(W)
print(E)

x_s = []
y_s = []
z_s = []
x_w = []
y_w = []
z_w = []
for i in S:
    x_s.append(i[0])
    y_s.append(i[1])
    z_s.append(i[2])
for i in W:
    x_w.append(i[0])
    y_w.append(i[1])
    z_w.append(i[2])

#ax.scatter(x_w, y_w, z_w)

#plt.show()

def calculate(s_0, w_0, h, steps):
    for i in range(steps):
        s_1 = s_0 + w_0*h
        w_1 = w_0 - (s_0/norm(s_0)**3) * h
        W.append(w_1)
        S.append(s_1)
        e = energy(w_1, s_1)
        E.append(e)
        s_0 = s_1
        w_0 = w_1

calculate(s, v, 0.01, 670)
print(S)
print(W)
print(E)
for i in S:
    x_s.append(i[0])
    y_s.append(i[1])
for i in W:
    x_w.append(i[0])
    y_w.append(i[1])
fig = plt.figure()

#plt.axis('scaled')
zero = np.zeros(len(x_s))
plt.xlim(-3, 3)
plt.ylim(-3, 3)
plt.quiver(zero, zero, x_s, y_s, scale=1, angles="xy", scale_units="xy")
plt.gca().add_artist(plt.Circle((0, 0), 1, fill=False, color="r"))
plt.show()