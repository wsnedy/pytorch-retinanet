import os, sys

lib_path_datasets = os.path.abspath(os.path.join('..', 'datasets'))
lib_path_network = os.path.abspath(os.path.join('..', 'network'))
lib_path_parallel = os.path.abspath(os.path.join('..', 'parallel'))
sys.path.append(lib_path_datasets)
sys.path.append(lib_path_network)
sys.path.append(lib_path_parallel)
import argparse
import torch
import torch.optim as optim
import torch.utils.data
from data_parallel import DataParallel
from retinanet import RetinaNet
from coco_json_dataset import COCOJsonDataset
from coco_dataset import COCODataset, collate_minibatch, MinibatchSampler, BatchSampler
from roidb import combined_roidb_for_training
from torch.autograd import Variable
import time
from eval_coco import evaluate_coco
import logging.handlers

log_file = 'log_rerun_xavier.txt'
handler = logging.handlers.RotatingFileHandler(log_file, maxBytes=1024 * 1024 * 20, backupCount=5)
fmt = '%(asctime)s - %(filename)s: %(lineno)s - %(name)s - %(message)s'
formatter = logging.Formatter(fmt)
handler.setFormatter(formatter)
logger = logging.getLogger('log')
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

parser = argparse.ArgumentParser(description='PyTorch RetinaNet Training')
parser.add_argument('--lr', default=1e-2, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--epoch', default=0, type=int, help='epoch number for checkpoint')
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
batchSampler = BatchSampler(
        sampler=MinibatchSampler(ratio_list, ratio_index),
        batch_size=16,
        drop_last=True
    )
dataset = COCODataset(roidb)
# there has a bug in dataloader, when use num_worker > 0,  finish a epoch, the code will stuck in dataloader
# perhaps it's the memory problem in dataloader
# so change the num_worker to 0
dataloader = torch.utils.data.DataLoader(dataset, batch_sampler=batchSampler, num_workers=4,
                                         collate_fn=collate_minibatch)
# for evaluation
eval_dataset_name = 'minival2014'
eval_dataset = COCOJsonDataset(root=root, annFile=eval_dataset_name, cache_dir=cache_dir)

# Model
net = RetinaNet()
net.load_state_dict(torch.load('../pretrained_model/net.pth'))
if args.resume:
    print('==> Resuming from checkpoint..')
    checkpoint = torch.load('../checkpoint/ckpt_xavier_{}.pth'.format(args.epoch))
    net.load_state_dict(checkpoint['net'])
    best_loss = checkpoint['loss']
    start_epoch = checkpoint['epoch']

net = DataParallel(net, device_ids=range(torch.cuda.device_count()), minibatch=True)
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
        inputs, loc_targets, cls_targets = blobs['data'], blobs['loc_targets'], blobs['cls_targets']
        inputs = list(map(Variable, inputs))
        loc_targets = list(map(Variable, loc_targets))
        cls_targets = list(map(Variable, cls_targets))

        optimizer.zero_grad()
        loc_loss, cls_loss = net(inputs, loc_targets, cls_targets)
        mean_loc_loss = loc_loss.mean()
        mean_cls_loss = cls_loss.mean()
        loss = mean_loc_loss + mean_cls_loss
        loss.backward()
        # torch.nn.utils.clip_grad_norm(net.parameters(), 0.1)
        optimizer.step()

        train_loss += loss.data[0]
        print('loc_loss: {:.3f} | cls_loss: {:.3f} | train_loss: {:.3f} | avg_loss: {:.3f}\n'.format(
            mean_loc_loss.data[0], mean_cls_loss.data[0], loss.data[0], train_loss / (batch_idx + 1)))
        logger.debug(('loc_loss: {:.3f} | cls_loss: {:.3f} | train_loss: {:.3f} | avg_loss: {:.3f}\n'.format(
            mean_loc_loss.data[0], mean_cls_loss.data[0], loss.data[0], train_loss / (batch_idx + 1))))
    # save checkpoint
    net.eval()
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
        torch.save(state, '../checkpoint/ckpt_xavier_{}.pth'.format(epoch))
        best_loss = train_loss


def adjust_learning_rate_manual(optimizer, iterations):
    if iterations == 65961:
        for param_group in optimizer.param_groups:
            param_group['lr'] /= 10.
            print(param_group['lr'], 'lr')
    elif iterations == 87948:
        for param_group in optimizer.param_groups:
            param_group['lr'] /= 10.
            print(param_group['lr'], 'lr')


iteration_per_epoch = int(len(dataloader))
print(iteration_per_epoch, 'iteration_per_epoch')  # 7330
logger.debug('iteration_per_epoch: {:.3f}'.format(iteration_per_epoch))
iterations = iteration_per_epoch * (start_epoch + 1)
print(iterations, 'iterations begin')
for epoch in range(start_epoch + 1, start_epoch + 200):
    if iterations <= 100000:
        adjust_learning_rate_manual(optimizer, iterations)
        train(epoch)
        # begin_time = time.time()
        # logger.debug(evaluate_coco(eval_dataset, net))
        # end_time = time.time()
        # print('time cost for evaluation: {}'.format(end_time - begin_time))
        # logger.debug('time cost for evaluation: {}'.format(end_time - begin_time))
        iterations += iteration_per_epoch
