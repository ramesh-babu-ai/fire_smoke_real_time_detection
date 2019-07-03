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


