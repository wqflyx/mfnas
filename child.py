from collections import namedtuple
import os
from collections import OrderedDict
import time

from utils import *
from utils import genotypes

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')


class Cell(nn.Module):
    def __init__(self, genotype, C_prev_prev, C_prev, C, reduction, reduction_prev):
        super(Cell, self).__init__()

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, 0, 1)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0)

        if reduction:
            op_names, indices = zip(*genotype.reduce)
            concat = genotype.reduce_concat
        else:
            op_names, indices = zip(*genotype.normal)
            concat = genotype.normal_concat
        self._compile(C, op_names, indices, concat, reduction)

    def _compile(self, C, op_names, indices, concat, reduction):
        assert len(op_names) == len(indices)
        self._steps = len(op_names) // 2
        self._concat = concat
        self.multiplier = len(concat)

        self._ops = nn.ModuleList()
        ii = 0
        for name, index in zip(op_names, indices):
            stride = 2 if reduction and index < 2 else 1
            op = OPS[name](C, stride, True, index, 2 + int(ii / 2))
            self._ops += [op]
            ii += 1
        self._indices = indices

    def forward(self, s0, s1, drop_prob, use_gpu):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        for i in range(self._steps):
            h1 = states[self._indices[2 * i]]
            h2 = states[self._indices[2 * i + 1]]
            op1 = self._ops[2 * i]
            op2 = self._ops[2 * i + 1]
            h1 = op1(h1)
            h2 = op2(h2)
            if self.training and drop_prob > 0.:
                if not isinstance(op1, Identity):
                    h1 = drop_path(h1, drop_prob, use_gpu)
                if not isinstance(op2, Identity):
                    h2 = drop_path(h2, drop_prob, use_gpu)
            s = h1 + h2
            states += [s]
        return torch.cat([states[i] for i in self._concat], dim=1)


class AuxiliaryHead(nn.Module):
    def __init__(self, C, num_classes, picture_size, sss_ps):
        # assuming input size 8x8 for picture_size 32
        # assuming input size 14x14 for picture_size 224
        super(AuxiliaryHead, self).__init__()

        if picture_size <= sss_ps:
            self.features = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.AvgPool2d(5, stride=3, padding=0, count_include_pad=False),  # image size = 2 x 2
                nn.Conv2d(C, 128, 1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 768, 2, bias=False),
                nn.BatchNorm2d(768),
                nn.ReLU(inplace=True)
            )
        else:
            self.features = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.AvgPool2d(5, stride=2, padding=0, count_include_pad=False),
                nn.Conv2d(C, 128, 1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 768, 2, bias=False),
                # nn.BatchNorm2d(768),
                nn.ReLU(inplace=True)
            )
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.view(x.size(0), -1))
        return x


class MicroNetwork(nn.Module):
    def __init__(self, C, num_classes, layers, auxiliary, genotype, input_data_channel, picture_size, sss_ps, use_gpu,
                 pool_position):
        super(MicroNetwork, self).__init__()
        self._layers = layers
        self._auxiliary = auxiliary
        self._input_data_channel = input_data_channel
        self._picture_size = picture_size
        self._sss_ps = sss_ps
        self._use_gpu = use_gpu
        self._pool_position = pool_position

        if picture_size <= sss_ps:
            stem_multiplier = 3
            C_curr = stem_multiplier * C
            self.stem = nn.Sequential(
                nn.Conv2d(self._input_data_channel, C_curr, 3, padding=1, bias=False),
                nn.BatchNorm2d(C_curr)
            )
            C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
            reduction_prev = False
        else:  # using large datasets searching space
            self.stem0 = nn.Sequential(
                nn.Conv2d(self._input_data_channel, C // 2, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(C // 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(C // 2, C, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(C),
            )

            self.stem1 = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Conv2d(C, C, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(C),
            )
            C_prev_prev, C_prev, C_curr = C, C, C
            reduction_prev = True

        self.cells = nn.ModuleList()

        for i in range(layers):
            if i in self._pool_position:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr
            if i == 2 * layers // 3:
                C_to_auxiliary = C_prev

        if auxiliary:
            self.auxiliary_head = AuxiliaryHead(C_to_auxiliary, num_classes, picture_size, sss_ps)

        if picture_size <= sss_ps:
            self.global_pooling = nn.AdaptiveAvgPool2d(1)
        else:
            self.global_pooling = nn.AvgPool2d(7)
        self.classifier = nn.Linear(C_prev, num_classes)

    def forward(self, input):
        logits_aux = None
        if self._picture_size <= self._sss_ps:
            s0 = s1 = self.stem(input)
        else:
            s0 = self.stem0(input)
            s1 = self.stem1(s0)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, self.drop_path_prob, self._use_gpu)
            if i == 2 * self._layers // 3:
                if self._auxiliary and self.training:
                    logits_aux = self.auxiliary_head(s1)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits, logits_aux


class Child:
    def __init__(self, opt):
        self.primitive_mapping = {
            3: 'max_pool_3x3',
            2: 'avg_pool_3x3',
            4: 'skip_connect',
            0: 'sep_conv_3x3',
            1: 'sep_conv_5x5',
            5: 'dil_conv_3x3',
            6: 'dil_conv_5x5'
        }

        self.primitive_mapping2 = {
            3: 1,
            2: 2,
            4: 3,
            0: 4,
            1: 5,
            5: 6,
            6: 7
        }

        self.opt = opt
        self.md = OrderedDict()

    def arc_decoding(self, arc):
        normal_list = []
        reduce_list = []
        i = 0
        if len(arc) == 32:
            while i < len(arc):
                if i < int(len(arc) / 2):
                    node = (self.primitive_mapping[arc[i + 1]], arc[i])
                    normal_list.append(node)
                    i += 2
                else:
                    node = (self.primitive_mapping[arc[i + 1]], arc[i])
                    reduce_list.append(node)
                    i += 2

        return Genotype(normal=normal_list, normal_concat=[2, 3, 4, 5],
                        reduce=reduce_list, reduce_concat=[2, 3, 4, 5])

    def arc2alpha(self, arc):
        # self.ind2nums = {1: 2, 3: 3, 5: 4, 7: 5}
        # self.ind_start = {1: 0, 3: 2, 5: 5, 7: 9}
        index_start = [0, 2, 5, 9]

        k = sum(1 for i in range(self.opt.module_nodes) for n in range(2 + i))
        num_ops = len(self.opt.primitive)

        normal = np.zeros((k, num_ops))
        reduce = np.zeros((k, num_ops))

        arc_normal = arc[0:len(arc) // 2]
        arc_reduce = arc[len(arc) // 2:]

        for i in range(len(arc_normal)):
            if i % self.opt.module_nodes == 3:  # i = 3, 7, 11, 15
                j = int(i / self.opt.module_nodes)
                link_index = arc_normal[i - 3] + index_start[j]
                ops_index = self.primitive_mapping2[arc_normal[i - 2]]
                normal[link_index, ops_index] = 1

                link_index = arc_normal[i - 1] + index_start[j]
                ops_index = self.primitive_mapping2[arc_normal[i]]
                normal[link_index, ops_index] = 1

        for i in range(len(arc_reduce)):
            if i % self.opt.module_nodes == 3:  # i = 3, 7, 11, 15
                j = int(i / self.opt.module_nodes)
                link_index = arc_reduce[i - 3] + index_start[j]
                ops_index = self.primitive_mapping2[arc_reduce[i - 2]]
                reduce[link_index, ops_index] = 1

                link_index = arc_reduce[i - 1] + index_start[j]
                ops_index = self.primitive_mapping2[arc_reduce[i]]
                reduce[link_index, ops_index] = 1

        for i in range(np.size(normal, 0)):
            if not np.any(normal[i, :]):
                normal[i, 0] = 1
            if not np.any(reduce[i, :]):
                reduce[i, 0] = 1

        return normal, reduce

    def _train(self, train_queue, model, criterion, optimizer):
        objs = AvgrageMeter()
        top1 = AvgrageMeter()
        top5 = AvgrageMeter()
        model.train()

        for step, (input, target) in enumerate(train_queue):  # every batch
            if self.opt.use_gpu:
                input = Variable(input).cuda()
                target = Variable(target).cuda()
            else:
                input = Variable(input)
                target = Variable(target)

            optimizer.zero_grad()
            logits, logits_aux = model(input)
            loss = criterion(logits, target)
            if self.opt.auxiliary:
                loss_aux = criterion(logits_aux, target)
                loss += self.opt.auxiliary_weight * loss_aux
            loss.backward()
            nn.utils.clip_grad_norm(model.parameters(), self.opt.grad_clip)
            optimizer.step()

            prec1, prec5 = accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

        return top1.avg

    def _infer(self, valid_queue, model, criterion):
        objs = AvgrageMeter()
        top1 = AvgrageMeter()
        top5 = AvgrageMeter()
        model.eval()

        for step, (input, target) in enumerate(valid_queue):

            with torch.no_grad():
                if self.opt.use_gpu:
                    input = Variable(input).cuda()
                    target = Variable(target).cuda()
                else:
                    input = Variable(input, volatile=True)
                    target = Variable(target, volatile=True)

            logits, _ = model(input)
            loss = criterion(logits, target)

            prec1, prec5 = accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

        return top1.avg

    def run_models(self, candidate_models, epochs, train_queue, test_valid_queue):
        self.opt.init_channels = 36
        self.opt.layers = 20
        self.opt.drop_path_prob = 0.2
        self.opt.auxiliary = True
        self.opt.weight_decay = 3e-4
        self.opt.pool_position = [self.opt.layers // 3, 2 * self.opt.layers // 3]

        reward = [0] * len(candidate_models)
        model_size = [0] * len(candidate_models)

        # controller will be opensourced in our nearly opensourced AutoML system project. We will add the link here.
        fix_model = eval("genotypes.MFNAS")

        model = MicroNetwork(self.opt.init_channels, self.opt.class_nums, self.opt.layers, self.opt.auxiliary,
                             fix_model, self.opt.input_channel,
                             self.opt.picture_size,
                             self.opt.sss_ps, self.opt.use_gpu, self.opt.pool_position)

        model_size[0] = utils.count_parameters_in_MB(model)

        input1 = torch.randn(1, self.opt.input_channel, self.opt.picture_size, self.opt.picture_size)

        model.drop_path_prob = self.opt.drop_path_prob

        print("flops is = %fMB" % utils.count_flops_in_MB(model, inputs=(input1,)))

        print("param size1 = %fMB" % model_size[0])

        criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.SGD(
            model.parameters(),
            self.opt.learning_rate,
            momentum=self.opt.momentum,
            weight_decay=self.opt.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.opt.max_search_epoch)

        if self.opt.use_gpu:
            model = model.cuda()
            criterion = criterion.cuda()

        for group in optimizer.param_groups:
            print("lr:", group["lr"])

        for epoch in range(epochs):
            scheduler.step()
            print('epoch %d lr %e' % (epoch, scheduler.get_last_lr()[0]))
            model.drop_path_prob = self.opt.drop_path_prob * epoch / epochs
            train_acc = self._train(train_queue, model, criterion, optimizer)
            valid_acc = self._infer(test_valid_queue, model, criterion)
            if epoch == epochs - 1:
                print('train_acc %f' % train_acc)
                print('val_acc %f' % valid_acc)
            valid_acc /= 100.0

        return reward
