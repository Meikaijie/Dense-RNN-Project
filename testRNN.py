from __future__ import print_function, division
import os
import tensorflow as tf
import numpy as np
from models._all import *

# PARAMETERS
num_epochs = 100
total_series_length = 50000 * 100
truncated_backprop_length = 32
state_size = 256
layers = 4
batch_size = 1
regularizers = [2, 4, 8]
cell_type = 'RNN'
classifier = drnn_classification
num_features = 123
real_dilations = [1, 2, 4, 8, 16, 32, 64, 128, 256]
non_dilations = [1] * layers
dilations = real_dilations
num_examples = 4620
num_batches = num_examples #//batch_size//truncated_backprop_length
num_classes = 61
device = 'CPU'  # change to 'GPU' to run on GPU

with tf.device('/{}:0'.format(device)):
    batchX_placeholder = \
        tf.placeholder(tf.float32,
                       [None, truncated_backprop_length, num_features])
    batchY_placeholder = \
        tf.placeholder(tf.int32, [None, truncated_backprop_length])

    x_input = batchX_placeholder

    labels_series = tf.unstack(batchY_placeholder, axis=1)

    logits_series = \
        classifier(x_input, [state_size] * layers, dilations[0:layers],
                   truncated_backprop_length, num_classes,
                   num_features, cell_type, regularizers)
    predictions_series = [tf.argmax(logits, 1) for logits in logits_series]

    losses = [
        tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                       labels=labels)
        for logits, labels in zip(logits_series, labels_series)
    ]
    total_loss = tf.reduce_mean(losses)

    train_step = tf.train.AdagradOptimizer(0.3).minimize(total_loss)

def test(validation=True):
  print("Starting test")

  all_test_feature_files = os.listdir('feature_files/TEST')
  all_test_label_files = os.listdir('feature_labels/TEST')
  total_samples = len(all_test_feature_files)
  test_feature_files = \
    all_test_feature_files[:int(total_samples/2)] if validation \
    else all_test_feature_files[int(total_samples/2):]
  test_label_files = \
    all_test_label_files[:int(total_samples/2)] if validation \
    else all_test_label_files[int(total_samples/2):]
  for sample_idx in range(len(test_feature_files)):

      example = np.load(os.path.join('feature_files/TEST',
                                      test_feature_files[sample_idx]))
      label = np.load(os.path.join('feature_labels/TEST',
                                      test_label_files[sample_idx]))

      stop_index = len(example) - len(example) % truncated_backprop_length

      batchX = [[] for _ in range(len(example)//truncated_backprop_length)]
      batchY = [[] for _ in range(len(example)//truncated_backprop_length)]

      for batch_pos in range(stop_index):
          batchX[batch_pos//truncated_backprop_length].append(example[batch_pos])
          batchY[batch_pos//truncated_backprop_length].append(label[batch_pos])
      batchX = np.array(batchX)
      batchY = np.array(batchY)

      _predictions_series = sess.run(
          [predictions_series],
          feed_dict={
              batchX_placeholder:batchX,
          })

      preds = np.array(_predictions_series).T
      accuracy = np.sum(preds == batchY)/float(preds.size)
      if sample_idx % 100 == 0:
        print("Testing, Step: {}, Accuracy: {}"
              .format(sample_idx, accuracy))

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    loss_list = []

    for epoch_idx in range(num_epochs):
        print("Starting epoch", epoch_idx)

        train_feature_files = os.listdir('feature_files/TRAIN')
        train_label_files = os.listdir('feature_labels/TRAIN')
        total_samples = len(train_feature_files)
        for sample_idx in range(total_samples):

            example = np.load(os.path.join('feature_files/TRAIN',
                                           train_feature_files[sample_idx]))
            label = np.load(os.path.join('feature_labels/TRAIN',
                                         train_label_files[sample_idx]))

            stop_index = len(example) - len(example) % truncated_backprop_length

            batchX = [[] for _ in range(len(example)//truncated_backprop_length)]
            batchY = [[] for _ in range(len(example)//truncated_backprop_length)]

            for batch_pos in range(stop_index):
                batchX[batch_pos//truncated_backprop_length].append(example[batch_pos])
                batchY[batch_pos//truncated_backprop_length].append(label[batch_pos])
            batchX = np.array(batchX)
            batchY = np.array(batchY)

            _total_loss, _train_step, _predictions_series = sess.run(
                [total_loss, train_step, predictions_series],
                feed_dict={
                    batchX_placeholder:batchX,
                    batchY_placeholder:batchY,
                })

            preds = np.array(_predictions_series).T
            accuracy = np.sum(preds == batchY)/float(preds.size)
            loss_list.append(_total_loss)

            if sample_idx % 100 == 0:
                print("Epoch {}, Step: {}, Loss: {}, Accuracy: {}"
                      .format(epoch_idx, sample_idx, _total_loss, accuracy))
                test()
