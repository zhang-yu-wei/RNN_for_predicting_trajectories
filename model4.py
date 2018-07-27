import numpy as np
import pickle
import matplotlib.pyplot as plt

input = []

dt = 0.1
num_traj = 300
num_points = 0
for i in range(0, num_traj):
    t = 0
    a = 0.5 + (5 - 0.5)*i/num_traj
    co_list = []
    num_points_ = 0
    while t <= 2*np.pi:
          t += dt
          x = a * np.cos(t)*np.sin(t)/t
          y = a * np.sin(t)*np.sin(t)/t
          co_list.append([x, y])
          num_points_ += 1
          if i%100 == 0:
              plt.plot(x, y, 'r*')

    num_points += num_points_
    print("trajectory "+str(i)+" done"+" with "+str(num_points_))
    input.append(co_list)
print('total points number:'+str(num_points))
plt.show()

with open('Sequences4.pk1', 'wb') as output:
    pickle.dump(input, output, pickle.HIGHEST_PROTOCOL)