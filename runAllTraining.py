import os

# COMMAND FORMAT: testRNN.py num_states num_layers reg_steps reg_step_size dil_steps dil_step_size num_epochs classifier
# Test Command List
command_list = ['python testRNN.py 5 1 2 2 0 0 1 bireg_rnn_classification', 'python testRNN.py 5 8 2 2 0 0 1 reg_rnn_classification', 'python testRNN.py 5 8 0 0 2 2 1 bdrnn_classification', 'python testRNN.py 5 8 0 0 2 2 1 drnn_classification', 'python testRNN.py 5 8 0 0 2 2 1 brnn_classification', 'python testRNN.py 5 8 0 0 2 2 1 rnn_classification']

# Add to command list here


# execute command list
for command in command_list:
	os.system(command)