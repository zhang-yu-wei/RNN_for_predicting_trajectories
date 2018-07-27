import pickle
import numpy as np
import matplotlib.pyplot as plt

co_list = []
num_points = 0
num_traj = 500
dtheta = np.pi/2/num_traj
theta = np.pi/2
"""create 1,000 physical model with 100 data points in each model"""
for i in range(1, num_traj + 1):
    theta -= dtheta
    num_points_ = 0
    x0 = 0
    y0 = 0
    v0 = 18

    g = 10
    cos_th = np.cos(theta)
    sin_th = np.sin(theta)
    t = 0
    # timestep
    dt = 0.05

    x = x0
    y = y0
    vx = v0*cos_th
    vy = v0*sin_th
    _co_list = []

    #generate a new trajectory
    while y>=0:
        _co_list.append([x, y])
        a = -g
        vy = vy + a*dt
        x = x + vx*dt
        y = y + vy*dt
        t += dt
        num_points_ += 1
        #print example trajectory
        if i%100 == 0:
           plt.plot(x, y, '*')

    if num_points_ < 50:
        break
    elif num_points_>100:
        break
    num_points += num_points_

    print(_co_list)
    print("trajectory "+str(i)+" done"+" with "+str(num_points_))
    co_list.append(_co_list)

print("there are " + str(len(co_list)) + " trajectories")
print("the total datapoints is " + str(num_points) + " data points.")

plt.show()
with open('Sequences.pk1', 'wb') as output:
    pickle.dump(co_list, output, pickle.HIGHEST_PROTOCOL)

