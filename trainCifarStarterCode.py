from scipy import misc
import numpy as np
import tensorflow as tf
import random
import matplotlib.pyplot as plt
import matplotlib as mp
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# --------------------------------------------------
# setup

def weight_variable(shape):
    '''
    Initialize weights
    :param shape: shape of weights, e.g. [w, h ,Cin, Cout] where
    w: width of the filters
    h: height of the filters
    Cin: the number of the channels of the filters
    Cout: the number of filters
    :return: a tensor variable for weights with initial values
    '''

    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    '''
    Initialize biases
    :param shape: shape of biases, e.g. [Cout] where
    Cout: the number of filters
    :return: a tensor variable for biases with initial values
    '''

    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    '''
    Perform 2-D convolution
    :param x: input tensor of size [N, W, H, Cin] where
    N: the number of images
    W: width of images
    H: height of images
    Cin: the number of channels of images
    :param W: weight tensor [w, h, Cin, Cout]
    w: width of the filters
    h: height of the filters
    Cin: the number of the channels of the filters = the number of channels of images
    Cout: the number of filters
    :return: a tensor of features extracted by the filters, a.k.a. the results after convolution
    '''

    h_conv = tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')
    return h_conv

def max_pool_2x2(x):
    '''
    Perform non-overlapping 2-D maxpooling on 2x2 regions in the input data
    :param x: input data
    :return: the results of maxpooling (max-marginalized + downsampling)
    '''

    h_max = tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    return h_max

def variable_summaries(var, name):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope(name):
    mean = tf.reduce_mean(var)
    s_mean = tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    s_stddev = tf.summary.scalar('stddev', stddev)
    s_max = tf.summary.scalar('max', tf.reduce_max(var))
    s_min = tf.summary.scalar('min', tf.reduce_min(var))
    s_hist = tf.summary.histogram('histogram', var)
    return tf.summary.merge([s_min,s_mean,s_max,s_hist,s_stddev])


nsamples = 10000 #total number of samples
ntrain = 1000 # per class
ntest = 100 # per class
nclass = 10 # number of classes
imsize = 28
nchannels = 1
batchsize = 500
max_step = 20000
drop_rate = 0.4
learning_rate = 0.0005

result_dir = './results/' # directory where the results from the training are saved

Train = np.zeros((ntrain*nclass,imsize,imsize,nchannels))
Test = np.zeros((ntest*nclass,imsize,imsize,nchannels))
LTrain = np.zeros((ntrain*nclass,nclass))
LTest = np.zeros((ntest*nclass,nclass))

itrain = -1
itest = -1
for iclass in range(0, nclass):
    for isample in range(0, ntrain):
        path = './CIFAR10/CIFAR10/Train/%d/Image%05d.png' % (iclass,isample)
        im = misc.imread(path); # 28 by 28
        im = im.astype(float)/255
        itrain += 1
        Train[itrain,:,:,0] = im
        LTrain[itrain,iclass] = 1 # 1-hot lable
    for isample in range(0, ntest):
        path = './CIFAR10/CIFAR10/Test/%d/Image%05d.png' % (iclass,isample)
        im = misc.imread(path); # 28 by 28
        im = im.astype(float)/255
        itest += 1
        Test[itest,:,:,0] = im
        LTest[itest,iclass] = 1 # 1-hot lable

sess = tf.InteractiveSession()

tf_data = tf.placeholder(tf.float32, shape = [None, imsize, imsize, nchannels]) #tf variable for the data, remember shape is [None, width, height, numberOfChannels] 
tf_labels = tf.placeholder(tf.float32, shape = [None, nclass])#tf variable for labels

# --------------------------------------------------
# model

# first convolutional layer
W_conv1 = weight_variable([5,5,1,32]) # wideth, height, input channel (thickness of filter), output channel
b_conv1 = bias_variable([32]) # 32 filters
z_conv1 = conv2d(tf_data, W_conv1) + b_conv1
h_conv1 = tf.nn.relu(z_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# second convolutional layer
W_conv2 = weight_variable([5,5,32,64]) # 64 filters
b_conv2 = bias_variable([64])
z_conv2 = conv2d(h_pool1,W_conv2) + b_conv2
h_conv2 = tf.nn.relu(z_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# densely connected layer
W_fc1 = weight_variable([7 * 7 * 64, 1024]) # (28/2/2 = 7)
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
z_fc1 = tf.matmul(h_pool2_flat, W_fc1) + b_fc1
h_fc1 = tf.nn.relu(z_fc1)

# dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# softmax
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# --------------------------------------------------
# loss
# set up the loss, optimization, evaluation, and accuracy
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_labels, logits=y_conv))
# train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
# train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
train_step = tf.train.RMSPropOptimizer(learning_rate).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(tf_labels,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# setup test summary
sum_acti = variable_summaries(h_conv1, 'conv1_activation')
sum_test_accuracy = tf.summary.scalar("test_accuracy", accuracy)
sum_test_loss = tf.summary.scalar("test_loss", cross_entropy)
test_summary_op = tf.summary.merge([sum_acti,sum_test_accuracy,sum_test_loss])

#setup train summary
sum_train_accuracy = tf.summary.scalar("train_accuracy", accuracy)
sum_train_loss = tf.summary.scalar("train_loss", cross_entropy)
vertical_padding = tf.zeros_like(tf.placeholder(tf.float32, shape=[1, 5, 1, 1]))
horizontal_padding = tf.zeros_like(tf.placeholder(tf.float32, shape=[1, 1, 35, 1]))
empty_layer_padding = tf.zeros_like(tf.placeholder(tf.float32, shape=[1, 5, 5, 1]))
for i in range(36):
    if i in range(32):
        layer_image = tf.reshape(W_conv1[:,:,:,i,None],[1,5,5,1])
    else:
        layer_image = empty_layer_padding

    if i == 0:                               # if first image in the canvas
        layer_image = tf.concat([layer_image, vertical_padding],2)
        weight_canvas_row = layer_image
        weight_canvas = layer_image
    elif i == 35:
        weight_canvas_row = tf.concat([weight_canvas_row, layer_image],2)
        weight_canvas = tf.concat([weight_canvas, weight_canvas_row],1)
    elif i % 6 == 5:                         # if last image in the row, no need for padding
        weight_canvas_row = tf.concat([weight_canvas_row, layer_image],2)
        weight_canvas_row = tf.concat([weight_canvas_row, horizontal_padding],1)
        if i/6 < 1:                          # if its first row of layer
            weight_canvas = weight_canvas_row
        else:                                # if its not first row
            weight_canvas = tf.concat([weight_canvas, weight_canvas_row],1)
    elif i % 6 == 0:                         # if first image in row, reinitiate row
        layer_image = tf.concat([layer_image, vertical_padding],2)
        weight_canvas_row = layer_image
    else:                                    # any other tile with vertical padding
        layer_image = tf.concat([layer_image, vertical_padding],2)
        weight_canvas_row = tf.concat([weight_canvas_row, layer_image],2)

    # tf.summary.image('weight_visualization_1', tf.reshape(W_conv1[:,:,:,i,None],[1,5,5,1]))
sum_weight = tf.summary.image('weight_canvas', weight_canvas)
summary_op = tf.summary.merge([sum_train_accuracy,sum_train_loss,sum_weight])

# Create a saver for writing training checkpoints.
saver = tf.train.Saver()

# Instantiate a SummaryWriter to output summaries and the Graph.
summary_writer = tf.summary.FileWriter(result_dir + "/train/", sess.graph)
test_writer = tf.summary.FileWriter(result_dir + "/test/", sess.graph)

# --------------------------------------------------
# optimization

sess.run(tf.initialize_all_variables())
batch_xs = np.zeros([batchsize,imsize,imsize,nchannels]) #setup as [batchsize, width, height, numberOfChannels] and use np.zeros()
batch_ys = np.zeros([batchsize,nclass])#setup as [batchsize, the how many classes] 
for i in range(max_step): # try a small iteration size once it works then continue
    perm = np.arange(nsamples)
    np.random.shuffle(perm)
    for j in range(batchsize):
        batch_xs[j,:,:,:] = Train[perm[j],:,:,:]
        batch_ys[j,:] = LTrain[perm[j],:]
    if i%100 == 0:
        #calculate train accuracy and print it
        train_accuracy = accuracy.eval(feed_dict={tf_data: batch_xs, tf_labels: batch_ys, keep_prob: drop_rate})
        print("step %d, training accuracy %g"%(i, train_accuracy))

        # Update the events file which is used to monitor the training (in this case,
        # only the training loss is monitored)
        summary_str = sess.run(summary_op, feed_dict={tf_data: batch_xs, tf_labels: batch_ys, keep_prob: drop_rate})
        summary_writer.add_summary(summary_str, i)
        summary_writer.flush()
    # save the checkpoints every 1100 iterations
    if i % 250 == 0 or i == max_step:
        checkpoint_file = os.path.join(result_dir, 'checkpoint')
        saver.save(sess, checkpoint_file, global_step=i)
        # updating testing summary
        test_accuracy = accuracy.eval(feed_dict={tf_data: Test, tf_labels: LTest, keep_prob: 1})
        print("step %d, testing accuracy %g"%(i, test_accuracy))

        test_summ = sess.run(test_summary_op, feed_dict={tf_data: Test, tf_labels: LTest, keep_prob: 1})
        summary_writer.add_summary(test_summ, i)
        summary_writer.flush()
        # test_writer.add_summary(test_summ, i)
        # test_writer.flush()
    train_step.run(feed_dict={tf_data: batch_xs, tf_labels: batch_ys, keep_prob: drop_rate}) # run one train_step # dropout only during training

# --------------------------------------------------
# test




print("test accuracy %g"%accuracy.eval(feed_dict={tf_data: Test, tf_labels: LTest, keep_prob: 1.0}))


sess.close()