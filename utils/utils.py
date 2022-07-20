import numpy as np
import torch
from torch.autograd import Variable
from thop import profile


def count_parameters_in_MB(model):
    # return np.sum(np.prod(v.size()) for v in model.parameters()) / 1e6
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6

def count_flops_in_MB(model, inputs):
    flops, params = profile(model, inputs=inputs)
    return flops / 1000 ** 2


class AvgrageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def drop_path(x, drop_prob, use_gpu=True):
    if drop_prob > 0.:
        keep_prob = 1. - drop_prob
        if use_gpu:
            mask = Variable(torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))
        else:
            mask = Variable(torch.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))
        x.div_(keep_prob)
        x.mul_(mask)
    return x


def link2mat(link_id, node_id):  # link_id:0,1,2,...,5; node_id:2,3,4,5
    accm_node = 0
    for i in range(2, node_id):
        accm_node += i
    return accm_node + link_id


def mat2link(mat_id):  # mat_id: 0, 1, 2, ..., 13; return(node_id, link_id)
    if mat_id in range(0, 2):
        return (2, mat_id)
    elif mat_id in range(2, 5):
        return (3, mat_id - 2)
    elif mat_id in range(5, 9):
        return (4, mat_id - 5)
    elif mat_id in range(9, 14):
        return (4, mat_id - 9)


def save(model, model_path):
    torch.save(model.state_dict(), model_path)


def load(model, model_path):
    model.load_state_dict(torch.load(model_path))
