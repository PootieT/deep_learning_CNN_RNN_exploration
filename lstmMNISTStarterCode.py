import tensorflow as tf 
from tensorflow.python.ops import rnn, rnn_cell
import numpy as np 
from tensorflow.examples.tutorials.mnist import input_data
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)#call mnist function

learningRate = 0.001
trainingIters = 50000
batchSize = 200
displayStep = 10
drop_rate = 0.6

nInput = 28 #we want the input to take the 28 pixels
nSteps = 28 #every 28
nHidden = 20 #number of neurons for the RNN
nClasses = 10 #this is MNIST so you know

result_dir = './results/'

x = tf.placeholder(tf.float32, [None, nSteps, nInput])
y = tf.placeholder(tf.float32, [None, nClasses])
# y = tf.placeholder(tf.float32)

weights = {'out': tf.Variable(tf.random_normal([nHidden, nClasses]))}
biases = {'out': tf.Variable(tf.random_normal([nClasses]))}
keep_prob = tf.placeholder(tf.float32)

sess = tf.InteractiveSession()

def RNN(x, weights, biases):
	x = tf.transpose(x, [1,0,2])
	x = tf.reshape(x, [-1, nInput])
	# x = tf.split(0, nSteps, x) #configuring so you can get it as needed for the 28 pixels
	x = tf.split(x,nSteps,0)

	# lstm = rnn_cell.BasicRNNCell(nHidden) #find which lstm to use in the documentation
	lstm = tf.contrib.rnn.GRUCell(nHidden)
	# lstm = tf.contrib.rnn.BasicLSTMCell(nHidden)

	# Initial state of the LSTM memory.
	outputs, states = rnn.static_rnn(lstm, x, dtype=tf.float32) #for the rnn where to get the output and hidden state 

	return tf.matmul(outputs[-1], weights['out'])+ biases['out']

pred = RNN(x, weights, biases)

#optimization
#create the cost, optimization, evaluation, and accuracy
#for the cost softmax_cross_entropy_with_logits seems really good
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred))
optimizer = tf.train.AdamOptimizer(learningRate).minimize(cost)
correctPred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))

#test summary
sum_test_accuracy = tf.summary.scalar("test_accuracy", accuracy)
sum_test_loss = tf.summary.scalar("test_loss", cost)
test_summary_op = tf.summary.merge([sum_test_accuracy,sum_test_loss])
#training summary
sum_train_accuracy = tf.summary.scalar("train_accuracy", accuracy)
sum_train_loss = tf.summary.scalar("train_loss", cost)
summary_op = tf.summary.merge([sum_train_accuracy,sum_train_loss])
#summary writer
summary_writer = tf.summary.FileWriter(result_dir + "/rnn/", sess.graph)
saver = tf.train.Saver()


init = tf.global_variables_initializer()

validation_X, validation_Y= mnist.validation.next_batch(5000)    # the batch size is 50
validation_X = validation_X.reshape((5000,nSteps,nInput))
test_set = mnist.test
summary_str = ''


sess.run(init)
step = 1
while step* batchSize < trainingIters:
	batchX, batchY = mnist.train.next_batch(batchSize) #mnist has a way to get the next batch
	batchX = batchX.reshape((batchSize, nSteps, nInput))
	sess.run(optimizer, feed_dict={x: batchX, y: batchY, keep_prob: drop_rate})
	if step % displayStep == 0:
		checkpoint_file = os.path.join(result_dir, 'checkpoint')
		saver.save(sess, checkpoint_file, global_step=step)

		train_acc = accuracy.eval(feed_dict={x: batchX, y: batchY, keep_prob: 1})
		test_acc = accuracy.eval(feed_dict={x: validation_X, y: validation_Y, keep_prob: 1})
		loss = cost.eval(feed_dict={x: batchX, y: batchY, keep_prob: 1})
		print("Iter " + str(step*batchSize) + ", Minibatch Loss= " + \
              str(loss) + ", Training Accuracy= " + str(train_acc) + \
              ", Testing Accuracy= " + str(test_acc))
		print "+================================="
		summary_str = sess.run(summary_op, feed_dict={x: batchX, y: batchY, keep_prob: 1})
    	summary_writer.add_summary(summary_str, step)
    	test_summ = sess.run(test_summary_op, feed_dict={x: validation_X, y: validation_Y, keep_prob: 1})
    	summary_writer.add_summary(test_summ, step)
    	summary_writer.flush()
	step +=1
print('Optimization finished')
print("Testing Accuracy:", \
    sess.run(accuracy, feed_dict={x: validation_X, y: validation_Y, keep_prob: 1}))