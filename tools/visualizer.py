import numpy as np
import matplotlib.pyplot as plt
import cv2
import pdb
import argparse
import os
import shutil

# finish the ntu
def getArgs():
    parse = argparse.ArgumentParser()
    parse.add_argument('--data_path', type=str, help='the input data', default='//home/wuqiang/Workspace/2_generative_model/3_DA_Gesture/2_ST_GCN/st-gcn-master/data/NTU-RGB-D/xsub/val_data.npy')
    parse.add_argument('--data_layout', type=str, help='openpose, ntu-rgb+d, ntu_edge', default='openpose')
    parse.add_argument('--out_path', type=str, help='the path where the result will be saved', default='./videos/')
    args = parse.parse_args()
    return vars(args)

'''
basic setting
including 1) the connection of open pose --> /net/utils/graph.py
          2) color for visualization
          3) image size
'''


''' data loader '''
def data_loader(data_path, data_id):
    data = np.load(data_path, mmap_mode='r')

    # N=0 [data index], M=0 [person id], C=0,1 [x and y, not confidence]
    # N, C, T, V, M = data.shape
    demo_item = data[data_id,:,:,:,:]
    return demo_item


'''visualization for each frame'''
def data_visu(data_item, frame_id, out_path):
    C, T, V, M = data_item.shape
    #print(data_item.shape)
    connecting_joint = np.array(
        [2, 1, 21, 3, 21, 5, 6, 7, 21, 9, 10, 11, 1, 13, 14, 15, 1, 17, 18, 19, 2, 8, 8, 12, 12]) - 1
    location = data_item[:, frame_id,:,:]

    plt.figure()
    plt.cla()
    plt.xlim(-1000, 2000)
    plt.ylim(-2000, 2000)
    for m in range(M):
        x = data_item[0,frame_id,:,m] * 1080
        y = (data_item[1,frame_id,:,m] * 1080)

        for v in range(V):
            k = connecting_joint[v]
            plt.plot([x[v],x[k]], [y[v],y[k]], '-o', c=(0.1,0.1,0.1), linewidth=0.5, markersize=0)
        plt.scatter(x, y, marker='o', s=16)
    #plt.show()
    plt.savefig(out_path + str(t) + '.png')
    plt.close()



if __name__ == '__main__':
    args = getArgs()
    data_path = args['data_path']
    out_path = args['out_path']
    data_id = 3
    data_item = data_loader(data_path, data_id)
    C, T, V, M = data_item.shape

    # process the data_item in each frame
    # rm and mkdir
    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    os.makedirs(out_path)

    for t in range(100):
        data_visu(data_item, t, out_path)


