import numpy as np
import os
import sys


def GetLabels(directory_path, windowsize, windowstep, t, dictionary=None):
	if dictionary == None:
		dictionary = ["b", "bcl", "d", "dcl", "g", "gcl", "p", "pcl", "t", "tcl", "k", "kcl", "dx", "q", "jh", "ch", "s", "sh", "z", "zh", 
	"f", "th", "v", "dh", "m", "n", "ng", "em", "en", "eng", "nx", "l", "r", "w", "y", 
	"hh", "hv", "el", "iy", "ih", "eh", "ey", "ae", "aa", "aw", "ay", "ah", "ao", "oy",
	"ow", "uh", "uw", "ux", "er", "ax", "ix", "axr", "ax-h", "pau", "epi", "h#"]
	# labelToPos = {}
	# counter = 0
	# #get the number of labels observed in training data
	# for DR in os.listdir(directory_path):
	# 	if DR != '.DS_Store':
	# 		for data_folder in os.listdir(os.path.join(directory_path,DR)):
	# 			if data_folder != '.DS_Store':
	# 				for labelfile in os.listdir(os.path.join(directory_path,DR,data_folder)):
	# 					if t.upper() == 'PHN' and labelfile.endswith(t.upper()):
	# 						f = open(os.path.join(directory_path,DR,data_folder,labelfile))
	# 						for line in f:
	# 							phoneme = line.split()[2]
	# 							if not phoneme in labelToPos:
	# 								labelToPos[phoneme] = counter
	# 								counter += 1
	# 						f.close()
	# 					# elif t.upper() == 'TXT' and labelfile.endswith(t.upper()):
	# 					# 	f = open(os.path.join(directory_path,DR,data_folder,labelfile))
	# 					# 	for line in f:
	# 					# 		words = line.split()[2:]
	# 					# 		for word in words:
	# 					# 			if not word in labelToPos:
	# 					# 				labelToPos[word] = counter
	# 					# 				counter += 1
	# 					# 	f.close()
	# 					else:
	# 						print("Invalid file type. Valid label file types include: PHN and TXT")
	# 						return

	#Get one-hot label vectors
	for DR in os.listdir(directory_path):
		if DR != '.DS_Store':
			for data_folder in os.listdir(os.path.join(directory_path,DR)):
				if data_folder != '.DS_Store':
					for labelfile in os.listdir(os.path.join(directory_path,DR,data_folder)):
						if t.upper() == 'PHN' and labelfile.endswith(t.upper()):

							filename = DR+'_'+data_folder+'_'+labelfile[:-4]
							features = None
							if directory_path.endswith('TRAIN'):
								features = np.load(os.path.join('feature_files/TRAIN','TRAIN_'+filename+'.WAV.npy'))
							elif directory_path.endswith('TEST'):
								features = np.load(os.path.join('feature_files/TEST','TEST_'+filename+'.WAV.npy'))
							else:
								print("Invalid path or file organization")
								return

							labels = np.zeros((features.shape[0],len(dictionary)))
							f = open(os.path.join(directory_path,DR,data_folder,labelfile))
							windowstart = 0
							labelIndex = 0
							label = [0]*len(dictionary)
							for line in f:
								info = line.split()
								start = float(info[0])/16000
								end = float(info[1])/16000
								phoneme = info[2]
								while windowstart + windowsize <= end + 0.5*windowsize and labelIndex<len(labels):
									label[dictionary.index(phoneme)] = 1
									labels[labelIndex] = label
									windowstart += windowstep
									labelIndex += 1
									label[dictionary.index(phoneme)] = 0
							# print(labels[50])

							if directory_path.endswith('TRAIN'):
								filename = 'TRAIN_'+filename
								np.save(os.path.join('feature_labels','TRAIN',filename), labels)
							else:
								filename = 'TEST_'+filename
								np.save(os.path.join('feature_labels','TEST',filename), labels)

if __name__ == "__main__":
	try:
		directory_path = sys.argv[1]
		filetype = sys.argv[2]
	except:
		print("Invalid input: Need to enter a directory path then a file extension")
	GetLabels(directory_path,0.025,0.01,filetype)

	print("I'm done")





