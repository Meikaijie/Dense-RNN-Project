import numpy as np
import os
import scipy.io.wavfile as wav
import sys
import python_speech_features as psf

if __name__ == "__main__":
	try:
		directory_path = sys.argv[1]
	except:
		print("Need to enter a directory path")

	for DR in os.listdir(directory_path):
		if DR != '.DS_Store':
			for data_folder in os.listdir(os.path.join(directory_path,DR)):
				if data_folder != '.DS_Store':
					for wavfile in os.listdir(os.path.join(directory_path,DR,data_folder)):
						if wavfile.endswith(".wav"):
							(rate,sig) = wav.read(os.path.join(directory_path,DR,data_folder,wavfile))
							mfcc_feat = psf.mfcc(sig,rate,numcep=40)
							featurefile = DR+'_'+data_folder+'_'+wavfile[:-4]
							if directory_path.endswith('TRAIN'):
								featurefile = 'TRAIN_'+featurefile
								np.save(os.path.join('feature_files','TRAIN', featurefile), mfcc_feat)
							else:
								featurefile = 'TEST_'+featurefile
								np.save(os.path.join('feature_files','TEST', featurefile), mfcc_feat)

	print("I'm done")
