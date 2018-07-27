import numpy as np
import tensorflow as tf
import pickle
import random
import matplotlib.pyplot as plt
from itertools import chain, repeat, islice

"""this file contains the structure of the network"""
def pad_infinite(iterable, padding=None):
   return chain(iterable, repeat(padding))

def pad(iterable, size, padding=None):
   return islice(pad_infinite(iterable, padding), size)

def length(sequence):
    used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
    length = tf.reduce_sum(used, 1)
    length = tf.cast(length, tf.int32)
    return length

num_hidden = 32
num_epoch = 3000
tr_example = 10000
num_input = 10
max_length = 80
frame_size = 2

# load in data
with open('Sequences4.pk1','rb') as input:
    _in_data = pickle.load(input)

# randomly shuffle it
random.shuffle(_in_data)

num_example = 0
in_data = []
in_target = []

# make examples
for i in range(0, len(_in_data)):
    co_list = _in_data[i]
    for j in range(0, len(co_list) - num_input):
        in_data.append(list(pad(co_list[0:j + num_input], max_length, [0, 0])))
        in_target.append(co_list[j + num_input])
        num_example += 1

    print(str(i + 1) + " trajectory saved")
print("there are " + str(num_example) + " examples")

tr_data = in_data[:tr_example]
tr_target = in_target[:tr_example]
te_data = in_data[tr_example:]
te_target = in_target[tr_example:]

batch_size = 1000
num_of_batches = int(len(tr_data)/batch_size)
# start the RNN
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
cost = tf.reduce_mean(tf.squared_difference(target, prediction))

optimizer = tf.train.AdamOptimizer(learning_rate=0.005).minimize(cost)

saver = tf.train.Saver()
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for epoch in range(num_epoch):
    ptr = 0
    avg_cost = 0
    for j in range(num_of_batches):
        inp, out = tr_data[ptr:ptr+batch_size], tr_target[ptr:ptr+batch_size]
        ptr += batch_size
        op, c = sess.run([optimizer,cost], {data:inp, target:out})
        avg_cost += c
    te_cost = sess.run(cost, {data:te_data, target: te_target})
    print("Epoch" + str(epoch) + ' - Cost' + str(avg_cost/num_of_batches))
    if epoch != 0:
       plt.plot(epoch, avg_cost/num_of_batches, 'r*')
       plt.plot(epoch, te_cost, 'g*')

    if epoch%10 == 0:
        for _ in range(10):
            i = np.random.randint(0, len(te_data))
            pred = sess.run(prediction, {data:np.reshape(te_data[i],
                                            [1, np.shape(te_data)[1], 2])})
            print("Final State =" + str(te_target[i]) + ", Prediction =" +
                   str(pred))
# save the model
saver.save(sess, "./RNN_tra")
plt.show()


