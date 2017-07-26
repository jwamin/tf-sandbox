import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#Sample data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#Placeholder, tf will insert mnist image at runtime, 'None' means tensor can be of any length
x = tf.placeholder(tf.float32, [None, 784])

#Tensors full os Zeros, one with 784
# We will learn W and b
#Weight
W = tf.Variable(tf.zeros([784, 10]))
#Bias.
b = tf.Variable(tf.zeros([10]))

#Model
y = tf.nn.softmax( #apply softmax
tf.matmul(x, W) #multiply x by W
 + b) #add b


#correct answer to test against, one_hot of 10, but empty
y_ = tf.placeholder(tf.float32, [None, 10])

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

train_step = tf.train.GradientDescentOptimizer(0.05).minimize(cross_entropy)

#Start interactive session, code can be written more flexibly
sess = tf.InteractiveSession()

#initialise variables
tf.global_variables_initializer().run()

for _ in range(1000):

  # get two random sets of length 100 for use in placeholders
  batch = mnist.train.next_batch(100)
  # prefrom training, use batches to populate placeholders
  sess.run(train_step, feed_dict={x: batch[0], y_: batch[1]})


#How well did the model do?
#check top prediction *tf.argmax(y,1)* against the answer tf.argmax(y_,1)
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
print(correct_prediction)

#do a mean on the result of above calculation, cast to float32s, false,true become 0,1
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#run output
print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
