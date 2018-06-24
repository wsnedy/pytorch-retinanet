from __future__ import print_function

import os
import sys

sys.path.append(os.path.join('..'))
import argparse
import torch
import torch.optim as optim
import torch.utils.data
from torch.nn.parallel import DataParallel
from lib.network.retinanet import RetinaNet
from lib.datasets.coco_dataset import COCODataset, collate_minibatch, MinibatchSampler
from lib.datasets.roidb import combined_roidb_for_training
from torch.autograd import Variable
import logging.handlers

log_file = 'log.txt'
handler = logging.handlers.RotatingFileHandler(log_file, maxBytes=1024 * 1024, backupCount=5)
fmt = '%(asctime)s - %(filename)s: %(lineno)s - %(name)s - %(message)s'
formatter = logging.Formatter(fmt)
handler.setFormatter(formatter)
logger = logging.getLogger('log')
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

parser = argparse.ArgumentParser(description='PyTorch RetinaNet Training')
parser.add_argument('--lr', default=1e-2, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

assert torch.cuda.is_available(), 'Error: CUDA not found'
best_loss = float('inf')  # best test loss
start_epoch = -1  # start from epoch 0 or last epoch

# Data
print('==> Preparing data...')
root = '/mnt/xfs1/home/chenqiang/data/coco'
dataset_names = ['train2014', 'valminusminival2014']
cache_dir = '../cache_dir'
roidb, ratio_list, ratio_index = combined_roidb_for_training(root, dataset_names, cache_dir)
sampler = MinibatchSampler(ratio_list, ratio_index)
dataset = COCODataset(roidb)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=24, sampler=sampler, num_workers=8,
                                         collate_fn=collate_minibatch)

# Model
net = RetinaNet()
net.load_state_dict(torch.load('../pretrained_model/net.pth'))
if args.resume:
    print('==> Resuming from checkpoint..')
    checkpoint = torch.load('../checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_loss = checkpoint['loss']
    start_epoch = checkpoint['epoch']

net = DataParallel(net, device_ids=range(torch.cuda.device_count()))
net.cuda()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)


# training
def train(epoch):
    print('\nEpoch: {}'.format(epoch))
    logger.debug('\nEpoch: {}'.format(epoch))
    net.train()
    net.module.freeze_bn()
    train_loss = 0
    for batch_idx, blobs in enumerate(dataloader):
        inputs, loc_targets, cls_targets = blobs['data'], blobs['loc_target'], blobs['cls_target']
        inputs = list(map(Variable, inputs))
        loc_targets = list(map(Variable, loc_targets))
        cls_targets = list(map(Variable, cls_targets))

        optimizer.zero_grad()
        prediction, num_pos, loc_loss, cls_loss = net(inputs, loc_targets, cls_targets)
        sum_loc_loss = loc_loss.sum()
        sum_cls_loss = cls_loss.sum()
        sum_num_pos = num_pos.data.sum()
        mean_loc_loss = sum_loc_loss / sum_num_pos
        mean_cls_loss = sum_cls_loss / sum_num_pos
        loss = mean_loc_loss + mean_cls_loss
        loss.backward()
        optimizer.step()

        train_loss += loss.data[0]
        print('loc_loss: {:.3f} | cls_loss: {:.3f} | train_loss: {:.3f} | avg_loss: {:.3f}\n'.format(
            mean_loc_loss.data[0], mean_cls_loss.data[0], loss.data[0], train_loss / (batch_idx + 1)))
        logger.debug(('loc_loss: {:.3f} | cls_loss: {:.3f} | train_loss: {:.3f} | avg_loss: {:.3f}\n'.format(
            mean_loc_loss.data[0], mean_cls_loss.data[0], loss.data[0], train_loss / (batch_idx + 1))))
    # save checkpoint
    global best_loss
    train_loss /= len(dataloader)
    if train_loss < best_loss:
        print('Saving...')
        state = {
            'net': net.module.state_dict(),
            'loss': train_loss,
            'epoch': epoch
        }
        if not os.path.isdir('../checkpoint'):
            os.mkdir('../checkpoint')
        torch.save(state, '../checkpoint/ckpt.pth')
        best_loss = train_loss


def adjust_learning_rate_manual(optimizer, iterations):
    if iterations == 60000:
        for param_group in optimizer.param_groups:
            param_group['lr'] /= 10.
    elif iterations == 80000:
        for param_group in optimizer.param_groups:
            param_group['lr'] /= 10.


iteration_per_epoch = int(len(dataloader) / 24.)
iterations = iteration_per_epoch * (start_epoch + 1)
for epoch in range(start_epoch + 1, start_epoch + 200):
    iterations += iteration_per_epoch * epoch
    if iterations <= 90000:
        adjust_learning_rate_manual(optimizer, iterations)
        train(epoch)
