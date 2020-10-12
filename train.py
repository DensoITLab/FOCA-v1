#!/usr/bin/env python3

import torch, os, sys, math, argparse, time, datetime, pickle, random, threading, scipy.io
import torch.nn as nn
import torch.optim as Optim
import torch.nn.functional as F
import torch.nn.init as init
import torch.distributed as dist
import torch.utils.data.distributed
import torch.multiprocessing as mp
import numpy as np
from wide_resnet import Wide_ResNet_fet, Wide_ResNet_clf
from pyramid_resnet import Pyramid_ResNet_fet, Pyramid_ResNet_clf
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

def set_args(idxLoop):
    parser = argparse.ArgumentParser()
    parser.add_argument('--flgContinue', type=bool, default=0)
    parser.add_argument('--flgMultiProcessing', type=bool, default=0)
    parser.add_argument('--flgFixSeed', type=bool, default=1)
    parser.add_argument('--flgNesterov', type=bool, default=1)
    parser.add_argument('--flgDataAug', type=bool, default=1)
    parser.add_argument('--flgCutout', type=bool, default=1)
    parser.add_argument('--flgRandomErasing', type=bool, default=0)
    parser.add_argument('--flgCosineSchedule', type=bool, default=0)
    if idxLoop == 0:
        parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--lr_step_schedule', type=float, default=[300, 400, 500], help='schedule of learn rate')
    parser.add_argument('--lr_drop_rate', type=float, default=0.1, help='drop rate of learn rate')
    parser.add_argument('--init_lr_fet', type=float, default=0.03, help='learn rate of 1st train of feature extractor')
    parser.add_argument('--lr_clf', type=float, default=0.03, help='learn rate of weak classifier')
    parser.add_argument('--lr_clf_2nd', type=float, default=0.003, help='learn rate of 2nd train')
    parser.add_argument('--wdr_fet', type=float, default=5e-4, help='weight decay rate')
    parser.add_argument('--wdr_clf', type=float, default=1e-2, help='weight decay rate')
    parser.add_argument('--numTrainClf', type=int, default=64)
    parser.add_argument('--numTrainFet', type=int, default=32)
    parser.add_argument('--numEpoch_1st', type=int, default=600)
    parser.add_argument('--numNeuron', type=int, default=128)
    parser.add_argument('--numEpoch_2nd', type=int, default=5)
    parser.add_argument('--mr', type=float, default=0.9, help='momentum rate')
    parser.add_argument('--init_mode', type=str, default='fan_out', choices=('fan_in', 'fan_out'))
    parser.add_argument('--init_std', type=float, default=0.1)
    parser.add_argument('--sizeBatch', type=int, default=128)
    parser.add_argument('--sizeBatch_valid', type=int, default=100)
    parser.add_argument('--freqLog', type=int, default=300, help='iteration')
    parser.add_argument('--freqValid', type=int, default=10, help='epoch')
    parser.add_argument('--optimizer', type=str, default='momentumSGD', choices=('sgd', 'momentumSGD', 'adam'))
    parser.add_argument('--depth', type=int, default=28, help='depth of net')
    parser.add_argument('--factor', type=int, default=10, help='factor of net')
    parser.add_argument('--network', type=str, default='wide_resnet', choices=('wide_resnet', 'pyramid_resnet'))
    parser.add_argument('--dataset', type=str, default='cifar10', choices=('cifar10', 'cifar100'))
    parser.add_argument('--idGPU', type=int, default=[0])
    parser.add_argument('--dirSave', type=str, default='./result_continue/')
    parser.add_argument('--epochWeight', type=int, default='500')

    args = parser.parse_args()
    args.sizeWorld = len(args.idGPU)
    args.sizeBatch = int(args.sizeBatch / args.sizeWorld)
    args.sizeBatch_valid = int(args.sizeBatch_valid / args.sizeWorld)
    if args.dataset == 'cifar100':
        args.numClass = 100
    elif args.dataset == 'cifar10':
        args.numClass = 10
    return args

def print_config(args, data, numParam):
    print("==============================================================")
    if args.flgContinue:
        print("Continue Train from {:d} epoch".format(args.epochWeight))
    else:
        print("New Train")
    print('GPU id: {}'.format(args.idGPU))
    print('Cosine Schedule flag: {}'.format(args.flgCosineSchedule))
    print('Step Lr schedule: {}'.format(args.lr_step_schedule))
    print('Num Train Epoch: {}'.format(args.numEpoch_1st))
    print('Feature Extractor Learn rate: {}'.format(args.init_lr_fet))
    print('Classifier Learn rate: {}'.format(args.lr_clf))
    print('2nd Train Learn rate: {}\tNum 2nd Train Epoch: {}'.format(args.lr_clf_2nd, args.numEpoch_2nd))
    print('Feature Extractor L2: {}\tClassifier L2: {}'.format(args.wdr_fet, args.wdr_clf))
    print('Num Classifier Batch-size: {}\tNum Classifier Train: {}'.format(int(data['sizeBatch_clf']*args.sizeWorld), args.numTrainClf))
    print('Num Minibatch-size: {}\t\tNum Feature Extractor Train: {}'.format(args.sizeBatch*args.sizeWorld, args.numTrainFet))
    print('Num Neuron: {}'.format(args.numNeuron))
    print('Dataset: {}\tNetwork: {}'.format(args.dataset, args.network))
    print('Network Depth: {}\tFactor: {}'.format(args.depth, args.factor))
    print('Init Mode: {}\tInit Std: {}'.format(args.init_mode, args.init_std))
    print('Optimizer: {}\tMomentum rate: {:.2f}'.format(args.optimizer, args.mr))
    print('Nesterov flag: {}'.format(args.flgNesterov))
    print('Standard Data Augmentation flag: {}'.format(args.flgDataAug))
    print('Cutout flag: {}'.format(args.flgCutout))
    print('Random Erasing flag: {}'.format(args.flgRandomErasing))
    print('Multi Processing flag: {}'.format(args.flgMultiProcessing))
    print('Num Train Data: {}'.format(int(data['train']['num']*args.sizeWorld)))
    print('Number of params: {}'.format(numParam))
    print('==============================================================')
    print("train start")
    f = open("{}config.txt".format(args.dirSave),"a")
    f.write("==============================================================\n")
    if args.flgContinue:
        f.write("Continue Train from {:d} epoch\n".format(args.epochWeight))
    else:
        f.write("New Train\n")
    f.write('GPU id: {}\n'.format(args.idGPU))
    f.write('Cosine Schedule flag: {}\n'.format(args.flgCosineSchedule))
    f.write('Step Lr schedule: {}\n'.format(args.lr_step_schedule))
    f.write('Num Train Epoch: {}\n'.format(args.numEpoch_1st))
    f.write('Feature Extractor Learn rate: {}\n'.format(args.init_lr_fet))
    f.write('Classifier Learn rate: {}\n'.format(args.lr_clf))
    f.write('2nd Train Learn rate: {}\t\tNum 2nd Train Epoch: {}\n'.format(args.lr_clf_2nd, args.numEpoch_2nd))
    f.write('Feature Extractor L2: {}\tClassifier L2: {}\n'.format(args.wdr_fet, args.wdr_clf))
    f.write('Num Classifier Batch-size: {}\tNum Classifier Train: {}\n'.format(int(data['sizeBatch_clf']*args.sizeWorld), args.numTrainClf))
    f.write('Num Minibatch-size: {}\t\tNum Feature Extractor Train: {}\n'.format(args.sizeBatch*args.sizeWorld, args.numTrainFet))
    f.write('Num Neuron: {}\n'.format(args.numNeuron))
    f.write('Dataset: {}\tNetwork: {}\n'.format(args.dataset, args.network))
    f.write('Network Depth: {}\tFactor: {}\n'.format(args.depth, args.factor))
    f.write('Init Mode: {}\tInit Std: {}\n'.format(args.init_mode, args.init_std))
    f.write('Optimizer: {}\tMomentum rate: {:.2f}\n'.format(args.optimizer, args.mr))
    f.write('Nesterov flag: {}\n'.format(args.flgNesterov))
    f.write('Standard Data Augmentation flag: {}\n'.format(args.flgDataAug))
    f.write('Cutout flag: {}\n'.format(args.flgCutout))
    f.write('Random Erasing flag: {}\n'.format(args.flgRandomErasing))
    f.write('Multi Processing flag: {}\n'.format(args.flgMultiProcessing))
    f.write('Num Train Data: {}\n'.format(int(data['train']['num']*args.sizeWorld)))
    f.write('Number of params: {}\n'.format(numParam))
    f.write("==============================================================\n")
    f.close()

    # set save
    result = dict()
    result['trainF'] = open(os.path.join(args.dirSave, 'train.csv'), 'a')
    result['validF'] = open(os.path.join(args.dirSave, 'valid.csv'), 'a')
    result['2ndTrainF'] = open(os.path.join(args.dirSave, 'transfer.csv'), 'a')
    result['tranValidF'] = open(os.path.join(args.dirSave, 'transfer_valid.csv'), 'a')
    return result

def unpickle(file):
    fp = open(file, 'rb')
    if sys.version_info.major == 2:
        data = pickle.load(fp)
    elif sys.version_info.major == 3:
        data = pickle.load(fp, encoding='latin-1')
    fp.close()
    return data

def get_dataset(args, idPS):
    data = dict()
    train = dict()
    valid = dict()
    if args.dataset == 'cifar10':
        d_train = []
        l_train = []
        for i in range(1,6):
            data_dic = unpickle("./../dataset/cifar10/cifar-10-batches-py/data_batch_{}".format(i))
            d_train.extend(data_dic['data'])
            l_train.extend(data_dic['labels'])
        train['input'] = torch.tensor(d_train, dtype=torch.float32).reshape(50000, 3, 32, 32)
        train['label'] = torch.tensor(l_train, dtype=torch.int64)
        del d_train, l_train
        data_dic = unpickle("./../dataset/cifar10/cifar-10-batches-py/test_batch")
        valid['input'] = torch.tensor(data_dic['data'], dtype=torch.float32).reshape(10000, 3, 32, 32)
        valid['label'] = torch.tensor(data_dic['labels'], dtype=torch.int64)
        data['numClass'] = 10
        data['sizeBatch_clf'] = 100
        data['co_length'] = 8
        data['mean'] = [0.49139968, 0.48215827, 0.44653124]
        data['std'] = [0.24703233, 0.24348505, 0.26158768]
    elif args.dataset == 'cifar100':
        data_dic = unpickle("./../dataset/cifar100/cifar-100-python/train")
        train['input'] = torch.tensor(data_dic['data'], dtype=torch.float32).reshape(50000, 3, 32, 32)
        train['label'] = torch.tensor(data_dic['fine_labels'], dtype=torch.int64)
        data_dic = unpickle("./../dataset/cifar100/cifar-100-python/test")
        valid['input'] = torch.tensor(data_dic['data'], dtype=torch.float32).reshape(10000, 3, 32, 32)
        valid['label'] = torch.tensor(data_dic['fine_labels'], dtype=torch.int64)
        data['numClass'] = 100
        data['sizeBatch_clf'] = 1000
        data['co_length'] = 4
        data['mean'] = [0.5071, 0.4867, 0.4408]
        data['std'] = [0.2675, 0.2565, 0.2761]

    train['input'] /= 255
    valid['input'] /= 255
    if args.flgDataAug:
        if train['input'].shape[2] == 32:
            temp = torch.zeros((len(train['input']), 3, 40, 40), dtype=torch.float)
            temp[:, :, 4:-4, 4:-4] = train['input']
        else:
            temp = torch.zeros((len(train['input']), 3, 80, 80), dtype=torch.float)
            temp[:, :, 8:-8, 8:-8] = train['input']
        train['input'] = temp
    for i in range(3):
        valid['input'][:,i,:,:] = (valid['input'][:,i,:,:] - torch.mean(train['input'][:,i,:,:])) / torch.std(train['input'][:,i,:,:])
        train['input'][:,i,:,:] = (train['input'][:,i,:,:] - torch.mean(train['input'][:,i,:,:])) / torch.std(train['input'][:,i,:,:])

    if args.flgMultiProcessing:
        train['num'] = int(len(train['input'])/args.sizeWorld)
        valid['num'] = int(len(valid['input'])/args.sizeWorld)
        train['input'] = train['input'][idPS*train['num'] + np.arange(train['num'], dtype=np.int32)]
        valid['input'] = valid['input'][idPS*valid['num'] + np.arange(valid['num'], dtype=np.int32)]
        train['label'] = train['label'][idPS*train['num'] + np.arange(train['num'], dtype=np.int32)]
        valid['label'] = valid['label'][idPS*valid['num'] + np.arange(valid['num'], dtype=np.int32)]
        data['sizeBatch_clf'] = int(data['sizeBatch_clf'] / args.sizeWorld)
    else:
        train['num'] = len(train['input'])
        valid['num'] = len(valid['input'])
    data['train'] = train
    data['valid'] = valid
    data['gather'] = [torch.zeros(2, device=idPS) for i in range(args.sizeWorld)]
    data['sizeEpoch'] = math.ceil(data['train']['num']/args.sizeBatch)
    del train, valid

    return data

def set_network(args, numClass, idPS):
    net = dict()
    if args.network == 'wide_resnet':
        net['fet'] = Wide_ResNet_fet(args)
        net['clf'] = Wide_ResNet_clf(args, numClass)
    elif args.network == 'pyramid_resnet':
        net['fet'] = Pyramid_ResNet_fet(args)
        net['clf'] = Pyramid_ResNet_clf(args, numClass)

    optimizer = dict()
    optimizer['fet'] = Optim.SGD(net['fet'].parameters(), lr=args.init_lr_fet, momentum=args.mr, weight_decay=args.wdr_fet, nesterov=args.flgNesterov)
    optimizer['clf'] = Optim.SGD(net['clf'].parameters(), lr=args.lr_clf, momentum=args.mr, weight_decay=args.wdr_clf, nesterov=args.flgNesterov)

    if args.flgMultiProcessing:
        print('Multi Processing Distributed Data Parallel')
        dist.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:23458', world_size=args.sizeWorld, rank=idPS)
        net['device'] = idPS
        torch.cuda.set_device(net['device'])
        net['fet'] = net['fet'].cuda(net['device'])
        net['clf'] = net['clf'].cuda(net['device'])
        net['fet'] = nn.parallel.DistributedDataParallel(net['fet'], device_ids=[net['device']])
        net['clf'] = nn.parallel.DistributedDataParallel(net['clf'], device_ids=[net['device']])
    else:
        net['device'] = idPS
        net['fet'] = net['fet'].cuda(net['device'])
        net['clf'] = net['clf'].cuda(net['device'])

    if args.flgContinue:
        state = torch.load("{}fet.pth".format(args.dirSave))
        net['fet'].load_state_dict(state['param'])
        optimizer['fet'].load_state_dict(state['optim'])
    else:
        args.epochWeight = 0
        if idPS == 0:
            args.dirSave = "./result_" + "{0:%Y%m%d_%H%M%S}/".format(datetime.datetime.now())
            os.mkdir(args.dirSave)

    net['numParam'] = sum([p.data.nelement() for p in net['fet'].parameters()]) + sum([p.data.nelement() for p in net['clf'].parameters()])
    return net, optimizer

def lr_schedule(args, optimizer, epoch):
    # reference https://github.com/facebookresearch/moco/blob/master/main_moco.py
    lr = args.init_lr_fet
    numEpoch = args.numEpoch_1st
    if args.flgCosineSchedule:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / numEpoch))
    else:  # stepwise lr schedule
        for milestone in args.lr_step_schedule:
            lr *= args.lr_drop_rate if epoch >= milestone else 1.
    optimizer['fet'].param_groups[0]['lr'] = lr

def valid(args, data, net):
    net['fet'].eval()
    net['clf'].eval()
    loss_valid = 0
    incorrect = 0
    with torch.no_grad():
        for idxBatch in range(int(data['valid']['num']/args.sizeBatch_valid)):
            idxData = idxBatch*args.sizeBatch_valid + torch.arange(args.sizeBatch_valid, dtype=torch.long)
            input, label = data['valid']['input'][idxData], data['valid']['label'][idxData]
            input, label = input.cuda(net['device'], non_blocking=True), label.cuda(net['device'], non_blocking=True)
            feature = net['fet'](input)
            output = net['clf'](feature)
            loss_valid += F.cross_entropy(output, label, reduction='sum').data.item()
            pred = output.data.max(1)[1]
            incorrect += pred.ne(label.data).cpu().sum()
    loss_valid /= data['valid']['num']
    err_valid = 100.*float(incorrect) / data['valid']['num']

    if args.flgMultiProcessing:
        dist.all_gather(data['gather'], torch.tensor([loss_valid, err_valid], device=net['device']))
        if net['device'] == 0:
            for i in range(1, args.sizeWorld):
                loss_valid += data['gather'][i][0] 
                err_valid += data['gather'][i][1] 
            loss_valid /= args.sizeWorld
            err_valid /= args.sizeWorld

    net['fet'].train()
    net['clf'].train()
    return loss_valid, err_valid

def init_weight(args, net, optimizer):
    for m in net.modules():
        if isinstance(m, nn.Linear):
            init.normal_(m.weight.data, mean=0, std=args.init_std)
            m.bias.data.zero_()
    for m in optimizer.state.values():
        m['momentum_buffer'].zero_()

def train_classifier(args, data, net, optimizer):
    init_weight(args, net['clf'], optimizer['clf'])
    idxData = torch.tensor(np.random.randint(0, data['train']['num'], data['sizeBatch_clf']), dtype=torch.long)
    input, label = data['train']['input'][idxData], data['train']['label'][idxData]
    if args.flgDataAug:
        input = data_augment(args, input, data['co_length'])
    input, label = input.cuda(net['device'], non_blocking=True), label.cuda(net['device'], non_blocking=True)
    with torch.no_grad():
        feature = net['fet'](input)

    for k in range(args.numTrainClf):
        optimizer['clf'].zero_grad()
        output = net['clf'](feature)
        loss = F.cross_entropy(output, label)
        loss.backward()
        optimizer['clf'].step()

def data_augment(args, image_org, length):
    if image_org.shape[2] == 40:
        sizeInput, padding = 32, 8
    else:
        sizeInput, padding = 64, 16
    image_aug = torch.zeros(image_org.shape[0], 3, sizeInput, sizeInput, dtype=torch.float, device=image_org.device.index)
    for i in range(image_org.shape[0]):
        xy = np.random.randint(0, padding, 2)
        image = image_org[i, :, xy[0]:xy[0] + sizeInput, xy[1]:xy[1] + sizeInput]
        if random.random() < 0.5:
            image = torch.flip(image, [2])

        if args.flgCutout:
            xy = torch.tensor(np.random.randint(0, 32, 2), dtype=torch.long)
            x1 = torch.clamp(xy[0] - length, 0, sizeInput)
            x2 = torch.clamp(xy[0] + length, 0, sizeInput)
            y1 = torch.clamp(xy[1] - length, 0, sizeInput)
            y2 = torch.clamp(xy[1] + length, 0, sizeInput)
            image[:, x1:x2, y1:y2] = 0.
        elif args.flgRandomErasing:
            if random.random() < 0.5:
                target_area = random.uniform(0.02, 0.4) * 1024
                aspect_ratio = random.uniform(0.3, 1 / 0.3)
                h = int(round(math.sqrt(target_area * aspect_ratio)))
                w = int(round(math.sqrt(target_area / aspect_ratio)))
                if h < image.shape[1] and w < image.shape[2]:
                    x1 = random.randint(0, image.shape[1] - h)
                    y1 = random.randint(0, image.shape[2] - w)
                    image[:, x1:x1+h, y1:y1+w] = 0.

        image_aug[i] = image
    return image_aug

def train(idPS, idxLoop, args):
    data = get_dataset(args, idPS)

    net, optimizer = set_network(args, data['numClass'], idPS)

    if idPS == 0:
        result = print_config(args, data, net['numParam'])
        stime = time.perf_counter()

    # train
    for idxEpoch in range(args.epochWeight+1, args.numEpoch_1st+1):
        idxShuffle = torch.randperm(data['train']['num'])
        lr_schedule(args, optimizer, idxEpoch-1)

        for idxBatch in range(data['sizeEpoch']):
            iter = (idxEpoch-1)*data['sizeEpoch'] + idxBatch + 1
            # train classifier
            if iter % args.numTrainFet == 1:
                train_classifier(args, data, net, optimizer)

            # train featurer
            idxData = idxShuffle[idxBatch*args.sizeBatch:(idxBatch+1)*args.sizeBatch]
            input, label = data['train']['input'][idxData], data['train']['label'][idxData]
            if args.flgDataAug:
                input = data_augment(args, input, data['co_length'])
            input, label = input.cuda(net['device'], non_blocking=True), label.cuda(net['device'], non_blocking=True)
            feature = net['fet'](input)
            output = net['clf'](feature)
            loss = F.cross_entropy(output, label)

            optimizer['fet'].zero_grad()
            loss.backward()
            optimizer['fet'].step()

            # log
            if iter % args.freqLog == 0:
                pEpoch = iter/data['sizeEpoch']
                pred = output.data.max(1)[1]
                incorrect = pred.ne(label.data).sum().float()
                err = 100.*incorrect/args.sizeBatch
                if args.flgMultiProcessing:
                    dist.all_gather(data['gather'], torch.tensor([loss.data, err], device=net['device']))
                    if net['device'] == 0:
                        for i in range(1, args.sizeWorld):
                            err += data['gather'][i][1]
                        err /= args.sizeWorld

                if idPS == 0:
                    etime = time.perf_counter()-stime
                    stime = time.perf_counter()
                    print('Epoch: {:.2f}\tLoss: {:.4f}\tError: {:.2f}%\tlr_fet: {:.0e}\tlr_clf: {:.0e}\t{:.4f}s/iter'.
                        format(pEpoch, loss.data.item(), err, optimizer['fet'].param_groups[0]['lr'], optimizer['clf'].param_groups[0]['lr'], etime/args.freqLog))
                    result['trainF'].write('{},{},{}\n'.format(pEpoch, loss.data.item(), err))
                    result['trainF'].flush()

        # validation
        if idxEpoch % args.freqValid == 0:
            loss_valid, err_valid = valid(args, data, net)
            if idPS == 0:
                print('Valid Epoch: {:.2f} \tloss: {:.4f} \tError: {:.2f}%\n'.format(idxEpoch, loss_valid, err_valid))
                result['validF'].write('{},{},{}\n'.format(idxEpoch, loss_valid, err_valid))
                result['validF'].flush()

    if idPS == 0:
        torch.save({'param': net['fet'].state_dict(), 'optim': optimizer['fet'].state_dict()}, os.path.join(args.dirSave, 'fet.pth'))

    # 2nd train
    init_weight(args, net['clf'], optimizer['clf'])
    optimizer['clf'].param_groups[0]['lr'] = args.lr_clf_2nd
    loss_tran = torch.zeros(args.numEpoch_2nd)
    err_tran = torch.zeros(args.numEpoch_2nd)
    for epoch_2nd in range(args.numEpoch_2nd):
        idxShuffle = torch.randperm(data['train']['num'])
        for idxBatch in range(data['sizeEpoch']):
            idxData = idxShuffle[idxBatch*args.sizeBatch:(idxBatch+1)*args.sizeBatch]
            input, label = data['train']['input'][idxData], data['train']['label'][idxData]
            if args.flgDataAug:
                input = data_augment(args, input, data['co_length'])
            input, label = input.cuda(net['device'], non_blocking=True), label.cuda(net['device'], non_blocking=True)
            with torch.no_grad():
                feature = net['fet'](input)
            output = net['clf'](feature)
            loss = F.cross_entropy(output, label)
            optimizer['clf'].zero_grad()
            loss.backward()
            optimizer['clf'].step()

        loss_valid, err_valid = valid(args, data, net)
        loss_tran[epoch_2nd] = loss_valid
        err_tran[epoch_2nd] = err_valid
        if idPS == 0:
            print('Valid Epoch of Transfer: {:.1f} \tloss: {:.4f} \tError: {:.2f}%'.format(epoch_2nd+1, loss_valid, err_valid))
            err_valid, ie = torch.min(err_tran, 0)

    if idPS == 0:
        print('2nd Train Epoch: {:.2f} \tloss: {:.4f} \tError: {:.2f}% at {:d} epoch\n'.format(idxEpoch, loss_tran[ie], err_valid, ie+1))
        result['2ndTrainF'].write('{},{},{}\n'.format(idxEpoch, loss_tran[ie], err_valid))
        result['2ndTrainF'].flush()
        result['tranValidF'].write('{},{}\n'.format(idxEpoch, loss_tran.numpy()))
        result['tranValidF'].write('{},{}\n\n'.format(idxEpoch, err_tran.numpy()))
        result['tranValidF'].flush()

    if idPS == 0:
        result['trainF'].close()
        result['validF'].close()
        result['2ndTrainF'].close()
        result['tranValidF'].close()

if __name__=='__main__':
    for idxLoop in range(10):
        args = set_args(idxLoop)
        if args.flgFixSeed:
            random.seed(args.seed)
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
        if args.flgMultiProcessing:
            if args.sizeWorld == 2:
                os.environ["CUDA_VISIBLE_DEVICES"] = "{},{}".format(args.idGPU[0],args.idGPU[1])
            mp.spawn(train, nprocs=args.sizeWorld, args=(idxLoop, args))
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(args.idGPU[0])
            train(0, idxLoop, args)
