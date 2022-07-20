import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import json
import csv

import torch.nn as nn
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
from model import NetworkCIFAR as Network
from utils import *
from utils import genotypes as genotypes
from data.dataset import DataEncap

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--mode', type=str, default='eval', help='location of the data corpus')
parser.add_argument('--data', type=str, default='/public/data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=96, help='batch size')
parser.add_argument('--picture_size', type=int, default=32, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=600, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')
parser.add_argument('--layers', type=int, default=20, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--auxiliary', action='store_true', default=True, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--cutout', action='store_true', default=True, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
parser.add_argument('--save', type=str, default='tmp', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='DARTS', help='which architecture to use')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
args = parser.parse_args()

# args.save = 'eval-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
# utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')

CIFAR_CLASSES = 10


def infer(valid_queue, model, criterion):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()

    with torch.no_grad():
        for step, (input, target) in enumerate(valid_queue):
            input = Variable(input, volatile=True).cuda()
            # target = Variable(target, volatile=True).cuda(async=True)
            target = Variable(target, volatile=True).cuda()

            logits, _ = model(input)
            loss = criterion(logits, target)

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

            if step % args.report_freq == 0:
                logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


def eval_arch(genotype_file, ckpt_path):
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)

    # train_transform, valid_transform = utils._data_transforms_cifar10(args)
    train_transform, valid_transform = DataEncap._data_transforms_cifar10(args)
    valid_data = dset.CIFAR10(root=args.data, train=False, download=True, transform=valid_transform)

    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=2)

    tmp_dict = json.load(open(genotype_file, 'r'))
    genotype = genotypes.Genotype(**tmp_dict)

    model = Network(args.init_channels, CIFAR_CLASSES, args.layers, args.auxiliary, genotype)
    model.drop_path_prob = 0.
    model = model.cuda()
    # clear unused gpu memory
    torch.cuda.empty_cache()
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))
    model.load_state_dict(torch.load(ckpt_path), strict=False)
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    valid_acc, valid_obj = infer(valid_queue, model, criterion)
    logging.info('valid_acc %f', valid_acc)


if __name__ == '__main__':
    base_dir = './'
    genotype_path = base_dir
    genotype_names = ['architecture']
    genotype_file = os.path.join(base_dir, '%s.txt' % (genotype_names[0]))
    ckpt_path = os.path.join(base_dir, 'weights_97_520.pt')
    eval_arch(genotype_file, ckpt_path)
