from __future__ import print_function, division
import tensorflow as tf
import copy
import itertools
import numpy as np
#import matplotlib.pyplot as plt
from . import _rnn_reformat, _contruct_cells, dRNN, multi_dRNN_with_dilations

num_epochs = 100
total_series_length = 50000 * 100
regularizers = [2, 4, 8]
truncated_backprop_length = regularizers[-1] * 2
state_size = 4
num_classes = 2
layers = 2
echo_step = 3
batch_size = 5
num_batches = total_series_length//batch_size//truncated_backprop_length

def bireg_rnn_classification(x,
                           hidden_structs,
                           dilations,
                           n_steps,
                           n_classes,
                           input_dims=1,
                           cell_type="RNN",
                           regularizers=None):
    """
    This function construct a multilayer dilated RNN for classifiction.
    Inputs:
        x -- a tensor of shape (batch_size, n_steps, input_dims).
        hidden_structs -- a list, each element indicates the hidden node dimension of each layer.
        dilations -- a list, each element indicates the dilation of each layer.
        regularizers -- a list, indicating which connections back you should use to make predictions
        n_steps -- the length of the sequence.
        n_classes -- the number of classes for the classification.
        input_dims -- the input dimension.
        cell_type -- the type of the RNN cell, should be in ["RNN", "LSTM", "GRU"].

    Outputs:
        pred -- the prediction logits at the last timestamp and the last layer of the RNN.
                'pred' does not pass any output activation functions.
    """
    # error checking
    assert (len(hidden_structs) == len(dilations))

    # reshape inputs
    x_reformat = _rnn_reformat(x, input_dims, n_steps)

    # construct a list of cells
    cells = _contruct_cells(hidden_structs, cell_type)
    cells2 = _contruct_cells(hidden_structs, cell_type)

    # define dRNN structures
    with tf.variable_scope('forward'):
        layer_outputs = multi_dRNN_with_dilations(cells, x_reformat, dilations)
    x_reformat.reverse()
    with tf.variable_scope('backward'):
        layer_outputs_backwards = multi_dRNN_with_dilations(cells2, x_reformat, dilations)
    layer_outputs_backwards.reverse()
    final_outputs = [tf.concat([layer_outputs[i], layer_outputs_backwards[i]], 1) for i in range(n_steps)]

    if dilations[0] == 1:
        # dilation starts at 1, no data dependency lost
        # define the output layer
        weights = tf.Variable(tf.random_normal(shape=[hidden_structs[-1] * (len(regularizers) + 1) * 2,
                                                      n_classes]))
        bias = tf.Variable(tf.random_normal(shape=[n_classes]))

        preds = []
        for i in range(len(final_outputs)):
            to_concat = [final_outputs[i]]
            for j in regularizers:
                if i-j >= 0:
                    to_concat.append(final_outputs[i-j])
                else:
                    to_concat.append(tf.matmul(to_concat[-1], tf.constant(0.0, shape=[hidden_structs[-1] * 2, hidden_structs[-1] * 2])))
            preds.append(tf.matmul(tf.concat(to_concat, 1), weights) + bias)

    return preds

# def generateData():
#     x = np.array(np.random.choice(2, total_series_length, p=[0.9, 0.1]))
#     y = np.roll(x, echo_step)
#     y[0:echo_step] = 0
#
#     x = x.reshape((batch_size, -1))  # The first index changing slowest, subseries as rows
#     y = y.reshape((batch_size, -1))
#
#     return (x, y)
#
# batchX_placeholder = tf.placeholder(tf.float32, [batch_size, truncated_backprop_length])
# batchY_placeholder = tf.placeholder(tf.int32, [batch_size, truncated_backprop_length])
#
# labels_series = tf.unstack(batchY_placeholder, axis=1)
# labels_series = labels_series[truncated_backprop_length//2:]
#
# x_input = tf.expand_dims(batchX_placeholder, 2)
#
# logits_series = reg_rnn_classification(x_input, [state_size], [1], regularizers, truncated_backprop_length, num_classes, 1)
# logits_series = logits_series[truncated_backprop_length//2:]
# predictions_series = [tf.nn.softmax(logits) for logits in logits_series]
#
# losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels) for logits, labels in zip(logits_series,labels_series)]
# total_loss = tf.reduce_mean(losses)
#
# train_step = tf.train.AdagradOptimizer(0.3).minimize(total_loss)
#
# with tf.Session() as sess:
#     sess.run(tf.initialize_all_variables())
#     plt.ion()
#     plt.figure()
#     plt.show()
#     loss_list = []
#
#     for epoch_idx in range(num_epochs):
#         x,y = generateData()
#
#         print("New data, epoch", epoch_idx)
#
#         for batch_idx in range(num_batches):
#             start_idx = batch_idx * truncated_backprop_length
#             end_idx = start_idx + truncated_backprop_length
#
#             batchX = x[:,start_idx:end_idx]
#             batchY = y[:,start_idx:end_idx]
#
#             _total_loss, _train_step, _predictions_series = sess.run(
#                 [total_loss, train_step, predictions_series],
#                 feed_dict={
#                     batchX_placeholder:batchX,
#                     batchY_placeholder:batchY,
#                 })
#
#             loss_list.append(_total_loss)
#
#             if batch_idx%100 == 0:
#                 print("Step",batch_idx, "Loss", _total_loss)
#
# plt.ioff()
# plt.show()
