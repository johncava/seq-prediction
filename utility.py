import numpy as np

table_hot = { 0: [0.0,1.8,67.0,-1.0],
             15: [1.0,-4.5,148.0,1.0],
             11: [0.0,-3.5,96.0,1.0],
              3: [-1.0,-3.5,91.0,1.0],
              1: [0.0,2.5,86.0,-1.0],
              2: [-1.0,-3.5,109.0,1.0],
             12: [0.0,-3.5,114.0,1.0],
              4: [0.0,-0.4,48.0,-1.0],
              7: [1.0,-3.2,118.0,1.0],
              6: [0.0,4.5,124.0,-1.0],
             10: [0.0,3.8,124.0,-1.0],
              8: [1.0,-3.9,135.0,1.0],
              9: [0.0,1.9,124.0,-1.0],
              5: [0.0,2.8,135.0,-1.0],
             13: [0.0,-1.6,90.0,-1.0],
             14: [0.0,-0.8,73.0,1.0],
             16: [0.0,-0.7,93.0,1.0],
             17: [0.0,-0.9,163.0,-1.0],
             19: [0.0,-1.3,141.0,1.0],
             18: [0.0,4.2,105.0,-1.0],
             20: [1.0,1.0,1.0,1.0],
             21: [0.0,0.0,0.0,0.0]
             }
        
def create_dataset(switch):
    data = np.load('cb513+profile_split1.npy')
    sequence = 0
    aa = 0
    data = data.reshape(514, 700, 57)
    dataset = []
    if switch == 0:
        for sequence in xrange(514):
            x_array = []
            label_array = []
            for aa in xrange(700):
                x_array.append(data[sequence][aa][0:22].tolist())
                label_array.append(data[sequence][aa][22:31].tolist())
            dataset.append([x_array, label_array])	
    elif switch == 1:
        for sequence in xrange(514):
            x_array = []
            label_array = []
            for aa in xrange(700):
                feature = data[sequence][aa][0:22].tolist()
                th = table_hot[feature.index(max(feature))]
                x_array.append(th)
                label_array.append(data[sequence][aa][22:31].tolist())
            dataset.append([x_array, label_array])
    return dataset
