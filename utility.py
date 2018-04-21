import numpy as np

def create_dataset():
	data = np.load('cb513+profile_split1.npy')
	sequence = 0
	aa = 0
	data = data.reshape(514, 700, 57)
	#print data[sequence][aa][0:22].tolist()
	#print data[sequence][aa][22:31].tolist()
	dataset = []
	for sequence in xrange(514):
		x_array = []
		label_array = []
		for aa in xrange(700):
			x_array.append(data[sequence][aa][0:22].tolist())
			label_array.append(data[sequence][aa][22:31].tolist())
		dataset.append([x_array, label_array])
	#a,b = dataset[0]
	#print a[0]
	#print b[0]
	return dataset
