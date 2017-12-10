from __future__ import print_function, division
import os
import tensorflow as tf
import numpy as np
import sys
import scipy.io.wavfile as wav
import python_speech_features as psf
from recorder import Recorder
from models._all import *

model = sys.argv[1]
model_parts = model.split('_')
num_states = int(model_parts[0])              
num_layers = int(model_parts[1])               
step_size = int(model_parts[2])
classifier_string = model_parts[5]

regularizers = [step_size**i for i in range(1,num_layers+1)]     #** make this user input
real_dilations = [step_size**i for i in range(num_layers)]    #** make this user input

truncated_backprop_length = 1          #** make this twice the max dilation/regularizer
if classifier_string == 'bireg_rnn_classification' or classifier_string == 'reg_rnn_classification':
  truncated_backprop_length = 2*regularizers[-1]
elif classifier_string == 'bdrnn_classification' or classifier_string == 'drnn_classification':
  truncated_backprop_length = 2*real_dilations[-1]
truncated_backprop_length = max(truncated_backprop_length,32)

sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

saver = tf.train.import_meta_graph('{}/model.ckpt.meta'.format(model))
saver.restore(sess, tf.train.latest_checkpoint('{}/'.format(model)))

graph = tf.get_default_graph()
batchX_placeholder = graph.get_tensor_by_name("Placeholder:0")

ops = []
for i in range(truncated_backprop_length):
  suffix = '_{}'.format(i) if i != 0 else ''
  ops.append(graph.get_tensor_by_name("ArgMax{}:0".format(suffix)))

print('Model restored.')

wav_file = 'demo_voice.wav'
# wav_file2 = 'data/lisa/data/timit/raw/TIMIT/TRAIN/DR1/FCJF0/SA1.WAV.wav'

dictionary = ["b", "bcl", "d", "dcl", "g", "gcl", "p", "pcl", "t", "tcl", "k", "kcl", "dx", "q", "jh", "ch", "s", "sh", "z", "zh", 
    "f", "th", "v", "dh", "m", "n", "ng", "em", "en", "eng", "nx", "l", "r", "w", "y", 
    "hh", "hv", "el", "iy", "ih", "eh", "ey", "ae", "aa", "aw", "ay", "ah", "ao", "oy",
    "ow", "uh", "uw", "ux", "er", "ax", "ix", "axr", "ax-h", "pau", "epi", "h#"]

def compress(preds, threshold):
    counter = 1
    current = None
    result = []
    for p in preds:
        if p == current:
            counter += 1
        elif current is not None:
            result.append((counter, current))
            counter = 1
        current = p
    result.append((counter, current))
    final = []
    for (c, p) in result:
        if c >= threshold:
            final.append(p)
    return final

while True:
    print('Press enter to record...')
    raw_input()
    rec = Recorder(channels=1, rate=16000)
    with rec.open(wav_file, 'wb') as recfile2:
        recfile2.start_recording()
        print('Press to stop recording...')
        raw_input()
        recfile2.stop_recording()

    print('Processing...')

    (rate,sig) = wav.read(wav_file)
    mfcc_feat = psf.mfcc(sig,rate,numcep=41,nfilt=41, appendEnergy=True)
    dmfcc_feat = psf.base.delta(mfcc_feat,2)
    ddmfcc_feat = psf.base.delta(dmfcc_feat,2)
    mfcc_feat = np.append(mfcc_feat,dmfcc_feat,axis=1)
    example = np.append(mfcc_feat,ddmfcc_feat,axis=1)

    stop_index = len(example) - len(example) % truncated_backprop_length

    batchX = [[] for _ in range(len(example)//truncated_backprop_length)]

    for batch_pos in range(stop_index):
        batchX[batch_pos//truncated_backprop_length].append(example[batch_pos])
    batchX = np.array(batchX)

    feed_dict = {batchX_placeholder: batchX}

    preds = list(np.array(sess.run(ops, feed_dict)).T.reshape(-1))

    final = compress([dictionary[p] for p in preds], 3)
    # print([dictionary[p] for p in preds])
    print(' '.join(final))
    # exit()
