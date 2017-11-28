import os

# COMMAND FORMAT: testRNN.py num_states num_layers step_size num_epochs cell_type classifier
# Test Command List
command_list = []#, 'python testRNN.py 5 8 2 2 0 0 1 RNN reg_rnn_classification', 'python testRNN.py 5 8 0 0 2 2 1 RNN bdrnn_classification', 'python testRNN.py 5 8 0 0 2 2 1 RNN drnn_classification', 'python testRNN.py 5 8 0 0 2 2 1 RNN brnn_classification', 'python testRNN.py 5 8 0 0 2 2 1 RNN rnn_classification']

# Add to command list here
layers = [3,5]
states = [250,500]
step_sizes = [1,2,3]
cell_types = ['RNN','LSTM']
model_types = ['bdrnn_classification','bireg_rnn_classification','drnn_classification','reg_rnn_classification']

for layer in layers:
	for state in states:
		for step_size in step_sizes:
			for cell_type in cell_types:
				for model_type in model_types:
					command = 'python testRNN.py ' + str(state) + ' ' + str(layer) + ' ' + str(step_size) + ' 10 ' + str(cell_type) + ' ' + str(model_type)
					command_list.append(command)

# baseline RNN
command_list.append('python testRNN.py 500 1 1 10 RNN drnn_classification')

# baseline LSTM
command_list.append('python testRNN.py 500 1 1 10 LSTM drnn_classification')

# execute command list
for command in command_list:
	os.system(command)