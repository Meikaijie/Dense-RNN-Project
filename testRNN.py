from __future__ import print_function, division
import os
import tensorflow as tf
import numpy as np
from models._all import *
import sys
import pickle
import time

# when using reg_rnn_classification, layers should equal 1, also I think we should make truncated_backprop_length at least twice the size of the longest
# dilation we are using or longest regularizer. Also we should have the option of only training losses on predictions where we have info for all skip connections.

# PARAMETERS
num_states = int(sys.argv[1])                # state_size
num_layers = int(sys.argv[2])                # layers
step_size = int(sys.argv[3])
# reg_steps = int(sys.argv[3])                 # length of regularizer array
# reg_step_size = int(sys.argv[4])             # multiplier for regularizer array (reg_step_size^i)
# dil_steps = int(sys.argv[5])                 # length of dilations array
# dil_step_size = int(sys.argv[6])             # multiplier for dilations array (dil_step_size^i)
epochs = int(sys.argv[4])                    #num_epochs
cell = sys.argv[5]
classifier_string = sys.argv[6]


classifier_dict = {'bireg_rnn_classification':bireg_rnn_classification,
                   'reg_rnn_classification':reg_rnn_classification,
                   'bdrnn_classification':bdrnn_classification, 
                   'drnn_classification':drnn_classification,
                   'brnn_classification':bdrnn_classification,
                   'rnn_classification':drnn_classification}

num_epochs = epochs #100                #** make this user input
state_size = num_states #128            #** make this user input
layers = num_layers #1                  #** make this user input, but follow rules above
# if classifier_string != 'bdrnn_classification' or classifier_string != 'drnn_classification':
#   layers = 1
regularizers = [step_size**i for i in range(1,num_layers+1)]     #** make this user input
cell_type = cell                                                 #** make this user input
classifier = classifier_dict[classifier_string]                  #** make this user input
num_features = 123
real_dilations = [step_size**i for i in range(num_layers)]    #** make this user input
truncated_backprop_length = 1          #** make this twice the max dilation/regularizer
if classifier_string == 'bireg_rnn_classification' or classifier_string == 'reg_rnn_classification':
  truncated_backprop_length = 2*regularizers[-1]
elif classifier_string == 'bdrnn_classification' or classifier_string == 'drnn_classification':
  truncated_backprop_length = 2*real_dilations[-1]
truncated_backprop_length = max(truncated_backprop_length,32)
num_classes = 61
device = 'CPU'  # change to 'GPU' to run on GPU

model_name = '_'.join(sys.argv[1:])
if not os.path.exists(model_name):
  os.makedirs(model_name)

with tf.device('/{}:0'.format(device)):
    batchX_placeholder = \
        tf.placeholder(tf.float32,
                       [None, truncated_backprop_length, num_features])
    batchY_placeholder = \
        tf.placeholder(tf.int32, [None, truncated_backprop_length])

    x_input = batchX_placeholder

    labels_series = tf.unstack(batchY_placeholder, axis=1)

    logits_series = \
        classifier(x_input, [state_size] * layers, real_dilations,
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

  all_test_feature_files = sorted(os.listdir('feature_files/TEST'))
  all_test_label_files = sorted(os.listdir('feature_labels/TEST'))
  total_samples = len(all_test_feature_files)
  test_feature_files = \
    all_test_feature_files[:int(total_samples/2)] if validation \
    else all_test_feature_files[int(total_samples/2):]
  test_label_files = \
    all_test_label_files[:int(total_samples/2)] if validation \
    else all_test_label_files[int(total_samples/2):]
  total_correct = 0
  total_phonemes = 0
  cumulative_loss = 0
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

      _loss, _predictions_series = sess.run(
          [total_loss, predictions_series],
          feed_dict={
              batchX_placeholder:batchX,
              batchY_placeholder:batchY
          })

      cumulative_loss += _loss
      preds = np.array(_predictions_series).T
      num_correct = np.sum(preds == batchY)
      total_correct += num_correct
      total_phonemes += preds.size
      accuracy = num_correct/float(preds.size)
      if sample_idx % 100 == 0:
        print("Testing, Step: {}, Loss: {}, Accuracy: {}"
              .format(sample_idx, _loss, accuracy))
  return cumulative_loss, total_correct/float(total_phonemes)

def train_epoch(train_model=True):
  train_feature_files = sorted(os.listdir('feature_files/TRAIN'))
  train_label_files = sorted(os.listdir('feature_labels/TRAIN'))
  total_samples = len(train_feature_files)
  total_correct = 0
  total_phonemes = 0
  cumulative_loss = 0
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

      if train_model:
        _loss, _train_step, _predictions_series = sess.run(
            [total_loss, train_step, predictions_series],
            feed_dict={
                batchX_placeholder:batchX,
                batchY_placeholder:batchY,
            }
        )
      else:
        _loss, _predictions_series = sess.run(
            [total_loss, predictions_series],
            feed_dict={
                batchX_placeholder:batchX,
                batchY_placeholder:batchY,
            }
        )

      cumulative_loss += _loss
      preds = np.array(_predictions_series).T
      num_correct = np.sum(preds == batchY)
      total_correct += num_correct
      total_phonemes += preds.size
      accuracy = num_correct/float(preds.size)

      if sample_idx % 100 == 0:
          print("Step: {}, Loss: {}, Accuracy: {}"
                .format(sample_idx, _loss, accuracy))
  return cumulative_loss, total_correct/float(total_phonemes)

start_time = time.clock()

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    model_saver = tf.train.Saver()
    training_loss_list = []
    training_accuracy_list = []
    validation_loss_list = []
    validation_accuracy_list = []

    for epoch_idx in range(num_epochs):
        print("Starting epoch", epoch_idx)
        train_epoch(train_model=True)
        training_loss, training_accuracy = train_epoch(train_model=False)
        print("Epoch {} finished, training loss: {}, training accuracy: {}"
              .format(epoch_idx, training_loss, training_accuracy))
        training_loss_list.append(training_loss)
        training_accuracy_list.append(training_accuracy)
        validation_loss, validation_accuracy = test(validation=True)
        print("Epoch {} finished, validation loss: {}, validation accuracy: {}"
              .format(epoch_idx, validation_loss, validation_accuracy))
        validation_loss_list.append(validation_loss)
        validation_accuracy_list.append(validation_accuracy)
    #** write loss_list to file
    f = open(model_name+'/training_loss','wb')
    pickle.dump(training_loss_list, f)
    f.close()
    f = open(model_name+'/validation_loss','wb')
    pickle.dump(validation_loss_list, f)
    f.close()
    #** write accuracy to file
    f = open(model_name+'/training_acc','wb')
    pickle.dump(training_accuracy_list, f)
    f.close()
    f = open(model_name+'/validation_acc','wb')
    pickle.dump(validation_accuracy_list, f)
    f.close()
    #** save model
    model_saver.save(sess, model_name+'/model.ckpt')
    #** track training time
    total_time = time.clock()-start_time
    f = open(model_name+'/training_time.txt','wb')
    f.write(str(total_time))
    f.close()