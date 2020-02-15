import numpy as np
import pdb

data_path = ''
# for pose data format -->
# https://github.com/kevinlin311tw/keras-openpose-reproduce/blob/master/inference/prediction.py

data = np.load(data_path, mmap_mode='r')
N, C, T, V, M = data.shape
# in the first version --> N=0 [first data idex], M=0 [first person id], C=0,1 [x and y, not confidence]
demo_item = data[0,:2,:,:,0]

