import input_data
import plotter
import tensorflow as tf

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


x = tf.placeholder("float", shape=[None, 784])
y = tf.placeholder("float", shape=[None, 10])

#quat
x_image = tf.reshape(x, [-1,28,28,1])

W = tf.Variable(tf.zeros([784,10]))


init = tf.initialize_all_variables()



#init weight
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

#init bias
def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

#convolution
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')

#1st layer
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)



#2st layer
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)



#3st layer
# W_fc1 = weight_variable([7 * 7 * 64, 1024])
# b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
# h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

#add dropout
# keep_prob = tf.placeholder("float")
# h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)



#4st layes
W_fc2 = weight_variable([7*7*64, 10])
b_fc2 = bias_variable([10])

y_hat = tf.nn.softmax(tf.matmul(h_pool2_flat, W_fc2) + b_fc2)



#optimization and evaluation
cross_entropy = -tf.reduce_sum(y*tf.log(y_hat))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

#vedere
sess = tf.InteractiveSession()

sess.run(tf.initialize_all_variables())

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_hat,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

for n in range(20000):
  batch = mnist.train.next_batch(50)
  if n %100 == 0:
    train_accuracy = accuracy.eval(feed_dict={x:batch[0], y: batch[1] }) #, keep_prob: 1.0})
    print("step %d, training accuracy %g" % (n, train_accuracy))
  sess.run(train_step,feed_dict={x: batch[0], y: batch[1] })#, keep_prob: 0.5})

print("test accuracy %g" % accuracy.eval(feed_dict={ x: mnist.test.images, y: mnist.test.labels }))# ,keep_prob: 1.0}))

