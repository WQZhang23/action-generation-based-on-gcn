import numpy as np
import matplotlib.pyplot as plt
import cv2
import pdb

#data_path = '/home/wuqiang/Workspace/2_generative_model/3_DA_Gesture/2_ST_GCN/st-gcn-master/data/NTU-RGB-D/xsub/val_data.npy'
data_path = '/home/wuqiang/Workspace/2_generative_model/3_DA_Gesture/2_ST_GCN/st-gcn-master/data/Kinetics/kinetics-skeleton/val_data.npy'
# for pose data format -->
# https://github.com/kevinlin311tw/keras-openpose-reproduce/blob/master/inference/prediction.py

# find connection in the specified sequence, center 29 is in the position 15
limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], \
           [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], \
           [1, 16], [16, 18], [3, 17], [6, 18]]

# visualize
colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0],
          [0, 255, 0], \
          [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255],
          [85, 0, 255], \
          [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

data = np.load(data_path, mmap_mode='r')
#data = np.load(data_path)
N, C, T, V, M = data.shape
# in the first version --> N=0 [first data idex], M=0 [first person id], C=0,1 [x and y, not confidence]
demo_item = data[17,:2,:,:,0]
C, T, V = demo_item.shape

# frame visu
height = 256
width = 340

def frame_visu(frame_item):
    out = np.zeros((height,width,3), np.uint8)
    for k in range(16):
        A_idx = limbSeq[k][0] - 1
        B_idx = limbSeq[k][1] - 1
        A_cor = frame_item[:, A_idx]
        B_cor = frame_item[:, B_idx]
        A_x = int((frame_item[0, A_idx] + 0.5) * 340)
        A_y = int((frame_item[1, A_idx] + 0.5) * 256)
        B_x = int((frame_item[0, B_idx] + 0.5) * 340)
        B_y = int((frame_item[1, B_idx] + 0.5) * 256)
        A = (int(A_x), int(A_y))
        B = (int(B_x), int(B_y))

        out = cv2.line(out,A,B,colors[k],1)
    return out

folder_path = '/home/wuqiang/Workspace/2_generative_model/3_DA_Gesture/3_Aug/generation/tools/videos/'

# process the video and save as gif
frame_list = []
for t in range(T):
    frame_item = demo_item[:, t, :]
    frame_out = frame_visu(frame_item)
    frame_list.append(frame_out)
    # pdb.set_trace()
    # plt.imshow(frame_out)
    # plt.show()
    cv2.imwrite(folder_path+str(t)+'.png', frame_out)






