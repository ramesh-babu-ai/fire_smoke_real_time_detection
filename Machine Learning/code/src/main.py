import os
import numpy as np
import tensorflow as tf

#pre deal the lable with the dataset
#two type of the dataset, positive and negative datasets

#dealing label and pass the dataset path
# 0:positive, 1:negative
def LabelDataset(training_positive_dir, training_negative_dir):
    positive_list = []
    negative_list = []
    label_positive = []
    label_negative = []
    for image in os.listdir(training_positive_dir):
        positive_list.append(training_positive_dir + image)
        label_positive.append(0)
    for image in os.listdir(training_negative_dir):
        negative_list.append(training_negative_dir + image)
        label_negative.append(1)

    #this step is merge two data list as one data list
    image_list = np.hstack((positive_list,negative_list))
    #this step is merge two label list as one label list
    label_list = np.hstack((label_positive,label_negative))

    temp_array = np.array([image_list, label_list])
    temp_array = temp_array.transpose()

    image_list = list(temp_array[:, 0])
    label_list = list(temp_array[:, 1])
    label_list = [int(i) for i in label_list]

    return image_list, label_list


def SetBatch(image_list, label_list, image_W, image_H, batch_size, capacity):
    image_list = tf.cast(image_list, tf.string)
    label_list = tf.cast(label_list, tf.int32)
    input_queue = tf.train.slice_input_producer([image_list,label_list])
    label_list = input_queue[1]
    image_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image_contents, channels=3)
    image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)
    image = tf.image.per_image_standardization(image)
    image_batch,label_batch = tf.train.batch([image,label_list],batch_size = batch_size,num_threads=16,capacity = capacity)
    label_batch = tf.reshape(label_batch, [batch_size])
    image_batch = tf.cast(image_batch, tf.float32)
    return image_batch, label_batch


#prepare the CNN model
def weight_variable(shape, n):
    initial = tf.truncated_normal(shape, stddev=n, dtype=tf.float32)
    return initial

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape, dtype=tf.float32)
    return initial

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x, name):
    return tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)



def DeepCNN(images, batch_size, n_classes):
    #first layer
    with tf.variable_scope('conv1') as scope:
        w_conv1 = tf.Variable(weight_variable([3, 3, 3, 64], 1.0), name='weights', dtype=tf.float32)
        b_conv1 = tf.Variable(bias_variable([64]), name='biases', dtype=tf.float32)
        h_conv1 = tf.nn.relu(conv2d(images, w_conv1)+b_conv1, name='conv1')
    #first pooling
    with tf.variable_scope('pooling1_lrn') as scope:
        pool1 = max_pool_2x2(h_conv1, 'pooling1')
        norm1 = tf.nn.lrn(pool1, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')

    #second layer
    with tf.variable_scope('conv2') as scope:
        w_conv2 = tf.Variable(weight_variable([3, 3, 64, 32], 0.1), name='weights', dtype=tf.float32)
        b_conv2 = tf.Variable(bias_variable([32]), name='biases', dtype=tf.float32)
        h_conv2 = tf.nn.relu(conv2d(norm1, w_conv2)+b_conv2, name='conv2')
    #second pooling
    with tf.variable_scope('pooling2_lrn') as scope:
        pool2 = max_pool_2x2(h_conv2, 'pooling2')
        norm2 = tf.nn.lrn(pool2, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')

    # third layer
    with tf.variable_scope('conv3') as scope:
        w_conv3 = tf.Variable(weight_variable([3, 3, 32, 16], 0.1), name='weights', dtype=tf.float32)
        b_conv3 = tf.Variable(bias_variable([16]), name='biases', dtype=tf.float32)
        h_conv3 = tf.nn.relu(conv2d(norm2, w_conv3)+b_conv3, name='conv3')

    # third pooling
    with tf.variable_scope('pooling3_lrn') as scope:
        pool3 = max_pool_2x2(h_conv3, 'pooling3')
        norm3 = tf.nn.lrn(pool3, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm3')

    # fourth fully connected layer
    with tf.variable_scope('local3') as scope:
        reshape = tf.reshape(norm3, shape=[batch_size, -1])
        dim = reshape.get_shape()[1].value
        w_fc1 = tf.Variable(weight_variable([dim, 128], 0.005),  name='weights', dtype=tf.float32)
        b_fc1 = tf.Variable(bias_variable([128]), name='biases', dtype=tf.float32)
        h_fc1 = tf.nn.relu(tf.matmul(reshape, w_fc1) + b_fc1, name=scope.name)

    # fifth fully connected layer
    with tf.variable_scope('local4') as scope:
        w_fc2 = tf.Variable(weight_variable([128 ,128], 0.005),name='weights', dtype=tf.float32)
        b_fc2 = tf.Variable(bias_variable([128]), name='biases', dtype=tf.float32)
        h_fc2 = tf.nn.relu(tf.matmul(h_fc1, w_fc2) + b_fc1, name=scope.name)
    h_fc2_dropout = tf.nn.dropout(h_fc2, 0.5)

    # Softmax regression layer
    with tf.variable_scope('softmax_linear') as scope:
        weights = tf.Variable(weight_variable([128, n_classes], 0.005), name='softmax_linear', dtype=tf.float32)
        biases = tf.Variable(bias_variable([n_classes]), name='biases', dtype=tf.float32)
        softmax_linear = tf.add(tf.matmul(h_fc2_dropout, weights), biases, name='softmax_linear')
    return softmax_linear


#define the loss
def losses(logits, labels):
    with tf.variable_scope('loss') as scope:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels, name='xentropy_per_example')
        loss = tf.reduce_mean(cross_entropy, name='loss')
        tf.summary.scalar(scope.name + '/loss', loss)
    return loss


def trainning(loss, learning_rate):
    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op


def evaluation(logits, labels):
    with tf.variable_scope('accuracy') as scope:
        correct = tf.nn.in_top_k(logits, labels, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float16))
        tf.summary.scalar(scope.name + '/accuracy', accuracy)
    return accuracy



#start main
N_CLASSES = 2
IMG_W = 32
IMG_H = 24
BATCH_SIZE = 20
CAPACITY = 200
MAX_STEP = 10000
learning_rate = 0.0001

train_dir_positive = 'F:/training/positive/positive_training/'
train_dir_negative = 'F:/training/negative/negative_training/'
train_dir_log = 'F:/training/log/'

train, train_label = LabelDataset(train_dir_positive,train_dir_negative)

train_batch, train_label_batch = SetBatch(train, train_label, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)

train_logits = DeepCNN(train_batch, BATCH_SIZE, N_CLASSES)
train_loss = losses(train_logits, train_label_batch)
train_op = trainning(train_loss, learning_rate)
train_acc = evaluation(train_logits, train_label_batch)
summary_op = tf.summary.merge_all()
sess = tf.Session()
train_writer = tf.summary.FileWriter(train_dir_log, sess.graph)
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)


try:
    for step in np.arange(MAX_STEP):
        if coord.should_stop():
            break
        _, tra_loss, tra_acc = sess.run([train_op, train_loss, train_acc])
        if step % 100 == 0:
            print('Step %d, train loss = %.2f, train accuracy = %.2f%%' % (step, tra_loss, tra_acc * 100.0))
            summary_str = sess.run(summary_op)
            train_writer.add_summary(summary_str, step)
        checkpoint_path = os.path.join(train_dir_log, 'thing.ckpt')
        saver.save(sess, checkpoint_path)

except tf.errors.OutOfRangeError:
    print('Done training -- epoch limit reached')

finally:
    coord.request_stop()
coord.join(threads)
sess.close()

