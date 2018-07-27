import numpy as np
import tensorflow as tf
import pickle
import random
import matplotlib.pyplot as plt
from itertools import chain, repeat, islice
import matplotlib.patches as mpatches

def pad_infinite(iterable, padding=None):
   return chain(iterable, repeat(padding))


def pad(iterable, size, padding=None):
   return islice(pad_infinite(iterable, padding), size)


def length(sequence):
    used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
    length = tf.reduce_sum(used, 1)
    length = tf.cast(length, tf.int32)
    return length

# load in data
with open('Sequences4.pk1','rb') as input:
    _in_data = pickle.load(input)

num_hidden = 32
tr_example = 10000
num_input = 10
max_length = 80
frame_size = 2

data = tf.placeholder(tf.float32, [None, max_length, 2])
target = tf.placeholder(tf.float32, [None, 2])

cell = tf.nn.rnn_cell.LSTMCell(num_hidden,state_is_tuple=True)
cell_dr = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=1, output_keep_prob=1)
out, state = tf.nn.dynamic_rnn(cell_dr, data, dtype=tf.float32)

out = tf.transpose(out, [1, 0, 2])
last = tf.gather(out, int(out.get_shape()[0]) - 1)

weight = tf.Variable(tf.truncated_normal([num_hidden, int(target.get_shape()[1])]))
bias = tf.Variable(tf.constant(0.1, shape=[target.get_shape()[1]]))

prediction = tf.matmul(last, weight) + bias

saver = tf.train.Saver()
sess = tf.Session()
saver.restore(sess, "./RNN_tra")

traj = _in_data[random.randint(0, len(_in_data))]
pred = traj[0:num_input]
for i in range(0, len(traj) - num_input):
    input = pred[0:num_input + i]
    input = list(pad(input, max_length, [0,0]))
    next = sess.run(prediction, {data:np.reshape(input, [1, max_length, 2])})
    next = next.tolist()
    pred.append(next[0])
    print("Final State =" + str(traj[num_input + i]) + ", Prediction =" +
                   str(next))
# print it out
for i in range(0, len(traj)):
    plt.plot(traj[i][0], traj[i][1], 'r*')
for i in range(0, num_input):
    plt.plot(pred[i][0], pred[i][1], 'b*')
for i in range(num_input, len(pred)):
    plt.plot(pred[i][0], pred[i][1], 'g*')

plt.grid(True)
plt.xlabel('x')
plt.ylabel('y')
red_patch = mpatches.Patch(color='red', label='original trajectory')
green_patch = mpatches.Patch(color='green', label='predicted trajectory')
blue_patch = mpatches.Patch(color='blue', label='input data')
plt.legend(handles=[red_patch, green_patch, blue_patch])
plt.show()
