import os
import matplotlib.pyplot as plt
import pickle
import numpy as np

# Get overall best models
allrnns = [f for f in os.listdir('.') if 'classification' in f]
allvalaccs = []
for model in allrnns:
	with open(model+'/validation_acc','rb') as f:
		val_acc_array = pickle.load(f)
		val_acc = max(val_acc_array)
	allvalaccs.append(val_acc)
print(np.array(allrnns)[np.argsort(allvalaccs)[-10:]])

# Get different models into separate lists
bdrnns = [f for f in os.listdir('.') if 'bdrnn' in f]
drnns = [f for f in os.listdir('.') if '_drnn' in f]
biregs = [f for f in os.listdir('.') if 'bireg' in f]
regs = [f for f in os.listdir('.') if '_reg' in f]

model_lists = []
model_lists.append(bdrnns)
model_lists.append(drnns)
model_lists.append(biregs)
model_lists.append(regs)

# Get best model names
best_models = []

for model_list in model_lists:
	highest_acc = 0
	best_model = None
	for model in model_list:
		f = open(model+'/validation_acc','rb')
		val_acc_array = pickle.load(f)
		val_acc = max(val_acc_array)
		if val_acc > highest_acc:
			best_model = model
			highest_acc = val_acc
		f.close()
	print(highest_acc)
	best_models.append(best_model)

# Get best t_loss, v_loss, t_acc, v_acc
best_t_loss = {}
best_v_loss = {}
best_t_acc = {}
best_v_acc = {}

for model in best_models:
	with open(model+'/training_loss','rb') as f:
		t_loss = pickle.load(f)
	best_t_loss[model] = t_loss
	with open(model+'/validation_loss','rb') as f:
		v_loss = pickle.load(f)
	best_v_loss[model] = v_loss
	with open(model+'/training_acc','rb') as f:
		t_acc = pickle.load(f)
	best_t_acc[model] = t_acc
	with open(model+'/validation_acc','rb') as f:
		v_acc = pickle.load(f)
	best_v_acc[model] = v_acc

model_names = ['bidirectional dilated', 'dilated', 'bidirectional regularized', 'regularized']

# Get to plotting curves
fig,ax = plt.subplots(1,1)
for ind, model in enumerate(best_models):
	t_loss = best_t_loss[model]
	x = range(len(t_loss))
	ax.plot(x, t_loss, label = model)
plt.title('Best Training Loss Curves by Architecture')
plt.xlabel('Epochs')
plt.ylabel('Training Loss')
plt.legend(loc='best')
plt.show()

fig,ax = plt.subplots(1,1)
for ind, model in enumerate(best_models):
	v_loss = best_v_loss[model]
	x = range(len(v_loss))
	ax.plot(x, v_loss, label = model)
plt.title('Best Validation Loss Curves by Architecture')
plt.xlabel('Epochs')
plt.ylabel('Validation Loss')
plt.legend(loc='best')
plt.show()

fig,ax = plt.subplots(1,1)
for ind, model in enumerate(best_models):
	t_acc = best_t_acc[model]
	x = range(len(t_acc))
	ax.plot(x, t_acc, label = model)
plt.title('Best Training Accuracy Curves by Architecture')
plt.xlabel('Epochs')
plt.ylabel('Training Accuracy')
plt.legend(loc='best')
plt.show()

fig,ax = plt.subplots(1,1)
for ind, model in enumerate(best_models):
	v_acc = best_v_acc[model]
	x = range(len(v_acc))
	ax.plot(x, v_acc, label = model)
plt.title('Best Validation Accuracy Curves by Architecture')
plt.xlabel('Epochs')
plt.ylabel('Validation Accuracy')
plt.legend(loc='best')
plt.show()

three_layer_accs = [] #
five_layer_accs = [] #

states_250_accs = [] #
states_500_accs = [] #

RNN_cell_accs = [] #
LSTM_cell_accs = [] #

one_step_accs = [] #
two_step_accs = [] #
three_step_accs = [] #

for model_list in model_lists:
	for model in model_list:
		model_params = model.split('_')
		if model_params[:-2] == ['250','3','2','10','RNN'] or model_params[:-3] == ['250','3','2','10','RNN']:
			with open(model+'/validation_acc', 'rb') as f:
				val_acc = max(pickle.load(f))
			three_layer_accs.append(val_acc)
			states_250_accs.append(val_acc)
			two_step_accs.append(val_acc)
			RNN_cell_accs.append(val_acc)
		elif model_params[:-2] == ['250','5','2','10','RNN'] or model_params[:-3] == ['250','5','2','10','RNN']:
			with open(model+'/validation_acc', 'rb') as f:
				val_acc = max(pickle.load(f))
			five_layer_accs.append(val_acc)
		elif model_params[:-2] == ['500','3','2','10','RNN'] or model_params[:-3] == ['500','3','2','10','RNN']:
			with open(model+'/validation_acc', 'rb') as f:
				val_acc = max(pickle.load(f))
			states_500_accs.append(val_acc)
		elif model_params[:-2] == ['250','3','3','10','RNN'] or model_params[:-3] == ['250','3','3','10','RNN']:
			with open(model+'/validation_acc', 'rb') as f:
				val_acc = max(pickle.load(f))
			three_step_accs.append(val_acc)
		elif model_params[:-2] == ['250','3','1','10','RNN'] or model_params[:-3] == ['250','3','1','10','RNN']:
			with open(model+'/validation_acc', 'rb') as f:
				val_acc = max(pickle.load(f))
			one_step_accs.append(val_acc)
		elif model_params[:-2] == ['250','3','2','10','LSTM'] or model_params[:-3] == ['250','3','2','10','LSTM']:
			with open(model+'/validation_acc', 'rb') as f:
				val_acc = max(pickle.load(f))
			LSTM_cell_accs.append(val_acc)
	# print(five_layer_accs)

print(states_250_accs)
print(states_500_accs)

# layer_tuples = zip(three_layer_accs,five_layer_accs)
state_tuples = zip(states_250_accs,states_500_accs)
cell_tuples = zip(RNN_cell_accs,LSTM_cell_accs)
step_tuples = zip(one_step_accs,two_step_accs,three_step_accs)

# Generate bar graph for layer parameter
# fig, ax = plt.subplots()
# index = np.arange(2)
# bar_width = 0.1
# opacity = 0.8

# rects1 = plt.bar(index, layer_tuples[0], bar_width, alpha=opacity, color='r', label='bi-dilated')
# rects2 = plt.bar(index+bar_width, layer_tuples[1], bar_width, alpha=opacity, color='g', label='dilated')
# rects3 = plt.bar(index+2*bar_width, layer_tuples[2], bar_width, alpha=opacity, color='b', label='bireg')
# rects4 = plt.bar(index+3*bar_width, layer_tuples[3], bar_width, alpha=opacity, color='y', label='reg')

# plt.xlabel('Number of Layers')
# plt.ylabel('Accuracy')
# plt.title('Accuracy vs. Layers')
# plt.xticks(index+2*bar_width, ('3','5'))
# plt.ylim([0,1.0])
# plt.legend(loc='best')
# plt.show()

# Generate bar graph for state number parameter
fig, ax = plt.subplots()
index = np.arange(2)
bar_width = 0.1
opacity = 0.8

rects1 = plt.bar(index, state_tuples[0], bar_width, alpha=opacity, color='r', label='bi-dilated')
rects2 = plt.bar(index+bar_width, state_tuples[1], bar_width, alpha=opacity, color='g', label='dilated')
rects3 = plt.bar(index+2*bar_width, state_tuples[2], bar_width, alpha=opacity, color='b', label='bireg')
rects4 = plt.bar(index+3*bar_width, state_tuples[3], bar_width, alpha=opacity, color='y', label='reg')

plt.xlabel('Number of States')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. States')
plt.xticks(index+2*bar_width, ('250','500'))
plt.ylim([0,1.0])
plt.legend(loc='best')
plt.show()

# Generate bar graph for cell type parameter
fig, ax = plt.subplots()
index = np.arange(2)
bar_width = 0.1
opacity = 0.8

rects1 = plt.bar(index, cell_tuples[0], bar_width, alpha=opacity, color='r', label='bi-dilated')
rects2 = plt.bar(index+bar_width, cell_tuples[1], bar_width, alpha=opacity, color='g', label='dilated')
rects3 = plt.bar(index+2*bar_width, cell_tuples[2], bar_width, alpha=opacity, color='b', label='bireg')
rects4 = plt.bar(index+3*bar_width, cell_tuples[3], bar_width, alpha=opacity, color='y', label='reg')

plt.xlabel('Cell Type')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. Cell Type')
plt.xticks(index+2*bar_width, ('RNN','LSTM'))
plt.ylim([0,1.0])
plt.legend(loc='best')
plt.show()

# Generate bar graph for step size parameter
fig, ax = plt.subplots()
index = np.arange(3)
bar_width = 0.1
opacity = 0.8

rects1 = plt.bar(index, step_tuples[0], bar_width, alpha=opacity, color='r', label='bi-dilated')
rects2 = plt.bar(index+bar_width, step_tuples[1], bar_width, alpha=opacity, color='g', label='dilated')
rects3 = plt.bar(index+2*bar_width, step_tuples[2], bar_width, alpha=opacity, color='b', label='bireg')
rects4 = plt.bar(index+3*bar_width, step_tuples[3], bar_width, alpha=opacity, color='y', label='reg')

plt.xlabel('Step Size')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. Step Size')
plt.xticks(index+2*bar_width, ('1','2','3'))
plt.ylim([0,1.0])
plt.legend(loc='best')
plt.show()

# # Get different num_layers into separate lists
# three_layers = [f for f in os.listdir('.') if f.split('_')[1]=='3']
# five_layers = [f for f in os.listdir('.') if f.split('_')[1]=='5']

# layer_lists = []
# layer_lists.append(three_layers)
# layer_lists.append(five_layers)

# # Get different num_states into separate lists
# states_250 = [f for f in os.listdir('.') if f.split('_')[0]=='250']
# states_500 = [f for f in os.listdir('.') if f.split('_')[0]=='500']

# state_lists = []
# state_lists.append(states_250)
# state_lists.append(states_500)

# # Get different cell type into separate lists
# RNN_cells = [f for f in os.listdir('.') if 'RNN' in f]
# LSTM_cells = [f for f in os.listdir('.') if 'LSTM' in f]

# cell_lists = []
# cell_lists.append(RNN_cells)
# cell_lists.append(LSTM_cells)

# # Get different step_size into separate lists
# one_step = [f for f in os.listdir('.') if f.split('_')[2]=='1']
# two_step = [f for f in os.listdir('.') if f.split('_')[2]=='2']
# three_step = [f for f in os.listdir('.') if f.split('_')[2]=='3']

# step_lists = []
# step_lists.append(one_step)
# step_lists.append(two_step)
# step_lists.append(three_step)



