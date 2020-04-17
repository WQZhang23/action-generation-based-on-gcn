import numpy as np
import argparse

# input_data[0] --> single_item
# input_data_item [0:3]-->gt, [3]-->mask, [4:]-->generated data
def getArgs():
    parse = argparse.ArgumentParser()
    parse.add_argument('--data_path', type=str, help='the input data', default='/workdir/generation/ntu-xsub/hinge_loss_w_100_sgd/inference_data.npy')
    args = parse.parse_args()
    return vars(args)

class OKS_pre_item(object):
    def __init__(self, input_data_item):
        self.gt = input_data_item[0:3]
        self.mask = input_data_item[3]
        self.pre = input_data_item[4:]
        self.len_incom = self.mask[:,0,0].tolist().count(1)

    def get_l2_loss(self):
        gt_mask = self.gt * self.mask
        pre_mask = self.pre * self.mask
        l2_loss = np.sum(np.square(gt_mask - pre_mask))
        l2_loss_norm = l2_loss/self.len_incom
        return l2_loss_norm


if __name__ == '__main__':
    args = getArgs()
    data_path = args['data_path']
    data = np.load(data_path)
    loss_list = []
    for item in data:
        itemer = OKS_pre_item(item)
        item_loss = itemer.get_l2_loss()
        loss_list.append(item_loss)

    loss_mean = np.mean(loss_list)
    print(loss_mean, itemer.len_incom)