# coding: utf-8
# A linear regression learning algorithm for Bangi Data Center - SNMP temperature data using TensorFlow library.
# Devtar Singh <devtar.singh@petronas.com.my>

import tensorflow as tf
import numpy
import pandas as pd
import matplotlib.pyplot as plt
rng = numpy.random

# Parameters
learning_rate = 0.01
training_epochs = 1000
display_step = 50

#CSV logs from Sam
df=pd.read_csv('ILO_10.14.218.74.csv',usecols = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,33],skiprows = [0],header=None)
d = df.values
l = pd.read_csv('ILO_10.14.218.74.csv',usecols = [0] ,header=None)
datetime = l.values
data = numpy.float32(d)
datetime = numpy.array(l,'str')
#print datetime,data

# Training Data
train_X = data
train_Y = datetime
n_samples = train_X.shape[0]

# tf Graph Input
X = tf.placeholder("float")
Y = tf.placeholder("string")

# Set model weights
W = tf.Variable(rng.randn(), name="weight")
b = tf.Variable(rng.randn(), name="bias")

# Construct a linear model
pred = tf.add(tf.multiply(X, W), b)

# Mean squared error
cost = tf.reduce_sum(tf.pow(pred-Y, 2))/(2*n_samples)
# Gradient descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Fit all training data
    for epoch in range(training_epochs):
        for (x, y) in zip(train_X, train_Y):
            sess.run(optimizer, feed_dict={X: x, Y: y})

        #Display logs per epoch step
        if (epoch+1) % display_step == 0:
            c = sess.run(cost, feed_dict={X: train_X, Y:train_Y})
            print "Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c), \
                "W=", sess.run(W), "b=", sess.run(b)

    print "Optimization Finished!"
    training_cost = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
    print "Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n'

    # Graphic display
    plt.plot(train_X, train_Y, 'ro', label='BDC Temp data')
    plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Prediction line')
    plt.legend()
    # Linear Regression results
    plt.show()


