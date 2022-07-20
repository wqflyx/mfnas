import sys
import torch
import torch.backends.cudnn as cudnn
import time
import numpy as np

from config import DefaultConfig
from data import DataEncap
from child import Child
from utils.genotypes import Genotype


def f_enas2darts(enas_module: list, enas2darts: list) -> Genotype:
    """
    This function changes enas representation to darts representation for future training and testing.
    :param enas_module: enas module
    :param enas2darts: transform matrix
    :return: darts genotype
    """

    normal_list = []
    reduce_list = []
    i = 0
    if len(enas_module) == 32:
        while i < len(enas_module):
            if i < int(len(enas_module)) / 2:
                node = [enas2darts[enas_module[i + 1]], int(enas_module[i])]
                normal_list.append(node)
                i += 2
            else:
                node = [enas2darts[enas_module[i + 1]], int(enas_module[i])]
                reduce_list.append(node)
                i += 2

    return Genotype(
        normal=normal_list, normal_concat=[2, 3, 4, 5],
        reduce=reduce_list, reduce_concat=[2, 3, 4, 5])


def main():
    start_time = time.time()
    opt = DefaultConfig()
    new_config = {"dataset_name": "CIFAR10", "picture_size": 32, "input_channel": 3,
                  "class_nums": 10}  # support: MNIST, CIFAR10, SVHN, STL10, FashionMNIST, ImageFolder
    if not new_config["dataset_name"] == "ImageFolder":
        opt.base_model_dir = opt.base2_model_dir + new_config["dataset_name"]
        opt.genotype_dir = opt.base2_genotype_dir + new_config["dataset_name"]
    else:
        if opt.dataset_root.split("/")[-1]:
            opt.base_model_dir = opt.base2_model_dir + opt.dataset_root.split("/")[-1]
            opt.genotype_dir = opt.base2_genotype_dir + opt.dataset_root.split("/")[-1]
        else:
            opt.base_model_dir = opt.base2_model_dir + opt.dataset_root.split("/")[-2]
            opt.genotype_dir = opt.base2_genotype_dir + opt.dataset_root.split("/")[-2]
    opt.parse(new_config)

    if opt.use_gpu:
        if not torch.cuda.is_available():
            print('no gpu device available')
            sys.exit(1)

    if not opt.seed == None:
        np.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)
    if opt.use_gpu:
        torch.cuda.set_device(opt.gpu)
    cudnn.benchmark = True
    cudnn.enabled = True

    child = Child(opt)

    top_idx = []
    top_results = []
    for episode in range(len(opt.tasks)):

        nums_cand, nums_epoch = opt.tasks[episode]

        if episode < len(opt.tasks) - 1:
            data = DataEncap(opt)
            train_queue, test_valid_queue = data.data_generate(is_search=True)
        else:
            data = DataEncap(opt)
            train_queue, test_valid_queue = data.data_generate(is_search=False)

        rewards = child.run_models([1], nums_epoch, train_queue, test_valid_queue)
        sorted_idx = (-np.array(rewards)).argsort()
        print("reward:", rewards)
        print("sorted_idx", sorted_idx)

        top_idx.append(list(sorted_idx))
        top_results.append(rewards)

        rewards = [np.exp(-1 * (1 - reward_i) * (1 - reward_i)) for reward_i in rewards]

        end_time = time.time()
        print("episode %d takes %f hours with %f rewards: " % (episode, (end_time - start_time) / 3600.0, rewards))


if __name__ == "__main__":
    main()
