import os
import json
import time
import datetime
import argparse

import numpy as np

import torch
import torch.optim
import torchvision
from torchvision import datasets, transforms

from optim.adam import Adam
from optim.adamax import Adamax
from models.nets import Model
from models.utils import preprocess, postprocess


def main(args):
    if args.mode == 'train':
        device = torch.device(args.device)
        save_dir = get_save_dir(args)
        # file_name = os.path.join(save_dir, args.dataset + '.json')
        # with open(file_name, 'w') as f_obj:
        #    json.dump(args.__dict__, f_obj)
        model = get_model(args)
        # model = torch.nn.DataParallel(model)
        model.to(device)
        optimizer = get_optimizer(args, model)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.step_size, args.lr_decay)
        train_data, test_data = get_dataset(args)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size,
                                                   num_workers=args.workers, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size,
                                                  num_workers=args.workers, shuffle=False)
        init_data = get_init_data(args, train_data)
        # param_num = sum(p.numel() for p in model.parameters())
        # print('parameter number  ', param_num)
        train(args, device, save_dir, model, optimizer, scheduler, train_loader, test_loader, init_data)

    elif args.mode == 'test':
        device = torch.device(args.device)
        save_dir = get_save_dir(args)
        model = get_model(args)
        model.to(device)
        # model = torch.nn.DataParallel(model)
        optimizer = get_optimizer(args, model)
        _, test_data = get_dataset(args)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size,
                                                  num_workers=args.workers, shuffle=False)
        model_dir = os.path.join(save_dir, 'models')
        file_name = 'epoch_{}.pth'.format(args.test_epoch)
        state_file = os.path.join(model_dir, file_name)
        if not os.path.isfile(state_file):
            raise RuntimeError('file {} is not found'.format(state_file))
        print('load checkpoint {}'.format(state_file))
        checkpoint = torch.load(state_file, device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        optimizer.swap()
        test(args, device, model, test_loader, args.test_epoch)
        optimizer.swap()

    elif args.mode == 'sample':
        device = torch.device(args.device)
        save_dir = get_save_dir(args)
        model = get_model(args)
        model.to(device)
        model = torch.nn.DataParallel(model)
        optimizer = get_optimizer(args, model)
        model_dir = os.path.join(save_dir, 'models')
        file_name = 'epoch_{}.pth'.format(args.sample_epoch)
        state_file = os.path.join(model_dir, file_name)
        if not os.path.isfile(state_file):
            raise RuntimeError('file {} is not found'.format(state_file))
        print('load checkpoint {}'.format(state_file))
        checkpoint = torch.load(state_file, device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        optimizer.swap()
        sample(args, device, save_dir, model, args.sample_epoch)
        optimizer.swap()

    else:
        raise ValueError('wrong mode')


def train(args, device, save_dir, model, optimizer, scheduler, train_loader, test_loader, init_data):
    train_log = {'train_loss': [], 'epoch_loss': [], 'epoch_time': [], 'test_loss': []}
    start_epoch = 1
    best_loss = 1e8
    lr_lambda = lambda step: (step + 1) / (len(train_loader) * args.warmup_epoch)
    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    if args.resume_epoch is not None:
        state_file = os.path.join(save_dir, 'models', 'epoch_' + str(args.resume_epoch) + '.pth')
        if not os.path.isfile(state_file):
            raise RuntimeError('file {} is not found'.format(state_file))
        print('load checkpoint {}'.format(state_file))
        checkpoint = torch.load(state_file, device)
        start_epoch = checkpoint['epoch'] + 1
        train_log = checkpoint['train_log']
        model.module.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
    else:
        model.eval()
        with torch.no_grad():
            init_data = init_data.to(device)
            z = torch.rand_like(init_data)
            init_data = preprocess(init_data, args.bits, z)
            _, _ = model(init_data, init=True)

    print('start training')
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        total_loss = 0.
        number = 0
        t0 = time.time()

        for data, _ in train_loader:
            data = data.to(device)
            z = torch.rand_like(data)
            data = preprocess(data, args.bits, z)
            output, log_det = model(data)
            loss = compute_loss(args, output, log_det)
            train_log['train_loss'].append(loss.item() / (np.log(2) * args.dimension))
            total_loss += loss.item() * data.size(0)
            number += data.size(0)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if epoch <= args.warmup_epoch:
                warmup_scheduler.step()

        bits_per_dim = total_loss / number / (np.log(2) * args.dimension)
        train_log['epoch_loss'].append((epoch, bits_per_dim))
        t1 = time.time()
        train_log['epoch_time'].append((epoch, t1 - t0))
        print('[train:epoch {}]. loss: {:.8f},time:{:.1f}s '.format(epoch, bits_per_dim, t1 - t0))
        scheduler.step()
        if not (epoch % args.test_interval):
            optimizer.swap()
            test_loss = test(args, device, model, test_loader, epoch)
            optimizer.swap()
            test_loss1 = test(args, device, model, test_loader, epoch)
            train_log['test_loss'].append((epoch, test_loss, test_loss1))
            if test_loss < best_loss:
                best_loss = test_loss
                save(save_dir, epoch, train_log, model, optimizer, scheduler, is_best=True)
        if not (epoch % args.save_interval):
            save(save_dir, epoch, train_log, model, optimizer, scheduler)
    return


def test(args, device, model, test_loader, epoch):
    total_loss = 0.
    number = 0
    model.eval()
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            z = torch.rand_like(data)
            data = preprocess(data, args.bits, z)
            output, log_det = model(data)
            loss = compute_loss(args, output, log_det)
            total_loss += loss.item() * data.size(0)
            number += data.size(0)
    bits_per_dim = total_loss / number / (np.log(2) * args.dimension)
    print('[test:epoch {}]. loss: {:.8f} '.format(epoch, bits_per_dim))
    return bits_per_dim


def sample(args, device, save_dir, model, epoch):
    z = torch.randn(args.sample_size, 3, args.image_size, args.image_size).to(device)
    model.eval()
    with torch.no_grad():
        output, _ = model(z, reverse=True)
        output = postprocess(output, args.bits)
    sample_dir = os.path.join(save_dir, 'samples')
    if not os.path.isdir(sample_dir):
        os.makedirs(sample_dir)
    torchvision.utils.save_image(output, os.path.join(sample_dir, 'epoch_{}.png'.format(epoch)),
                                 nrow=int(args.sample_size ** 0.5), pad_value=1)
    return


def get_save_dir(args):
    if args.save_dir:
        save_dir = args.save_dir
    else:
        name = args.dataset + '_' + str(datetime.datetime.now())[:-7].replace(' ', '-').replace(':', '-')
        save_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'save', name)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    return save_dir


def get_model(args):
    model = Model(args.levels, args.num_flows, args.conv_type, args.flow_type, args.num_blocks, args.hidden_channels,
                  args.image_size)
    return model


def get_dataset(args):
    if args.dataset == 'cifar10':
        train_data = datasets.CIFAR10(args.dataset_dir, train=True,
                                      transform=transforms.Compose([
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor()
                                      ]))
        test_data = datasets.CIFAR10(args.dataset_dir, train=False,
                                     transform=transforms.Compose([
                                         transforms.ToTensor()
                                     ]))
        assert args.image_size == 32
        assert args.dimension == 3072
    elif args.dataset == 'imagenet32':
        train_data = datasets.ImageFolder(os.path.join(args.dataset_dir, 'train_32x32'),
                                          transform=transforms.Compose([
                                              transforms.ToTensor()
                                          ]))
        test_data = datasets.ImageFolder(os.path.join(args.dataset_dir, 'valid_32x32'),
                                         transform=transforms.Compose([
                                             transforms.ToTensor()
                                         ]))
        assert args.image_size == 32
        assert args.dimension == 3072
    elif args.dataset == 'imagenet64':
        train_data = datasets.ImageFolder(os.path.join(args.dataset_dir, 'train_64x64'),
                                          transform=transforms.Compose([
                                              transforms.ToTensor()
                                          ]))
        test_data = datasets.ImageFolder(os.path.join(args.dataset_dir, 'valid_64x64'),
                                         transform=transforms.Compose([
                                             transforms.ToTensor()
                                         ]))
        assert args.image_size == 64
        assert args.dimension == 12288
    else:
        raise ValueError('wrong dataset')
    return train_data, test_data


def get_init_data(args, train_data):
    train_index = np.arange(len(train_data))
    np.random.shuffle(train_index)
    init_index = np.random.choice(train_index, args.init_batch_size, replace=False)
    images = []
    for index in init_index:
        image, _ = train_data[index]
        images.append(image)
    return torch.stack(images, dim=0)


def get_optimizer(args, model):
    if args.optimizer == 'adam':
        optimizer = Adam([{'params': model.parameters()}], lr=args.lr,
                         weight_decay=args.weight_decay, polyak=args.polyak)
    elif args.optimizer == 'adamax':
        optimizer = Adamax([{'params': model.parameters()}], lr=args.lr,
                           weight_decay=args.weight_decay, polyak=args.polyak)
    else:
        raise ValueError('wrong optimizer')
    return optimizer


def compute_loss(args, output, log_det):
    log_p = torch.distributions.Normal(torch.zeros_like(output), torch.ones_like(output)).log_prob(output).view(
        output.size(0), -1).sum(-1)
    loss = -(log_p + log_det - np.log((2.0 ** args.bits) / 2.0) * args.dimension).mean()
    return loss


def save(save_dir, epoch, train_log, model, optimizer, scheduler, is_best=False):
    model_dir = os.path.join(save_dir, 'models')
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    file_name = 'epoch_best.pth' if is_best else 'epoch_{}.pth'.format(epoch)
    state_path = os.path.join(model_dir, file_name)
    state = {
        'epoch': epoch,
        'train_log': train_log,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
    }
    torch.save(state, state_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MEF')
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'test', 'sample'],
                        help='mode')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='device')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'imagenet32', 'imagenet64'],
                        help='dataset')
    parser.add_argument('--dataset_dir', type=str, default='/datasets',
                        help='dataset directory.')
    parser.add_argument('--save_dir', type=str, default='',
                        help='save directory')
    parser.add_argument('--image_size', type=int, default=32,
                        choices=[32, 64],
                        help='image size')
    parser.add_argument('--dimension', type=int, default=3072,
                        choices=[3072, 12288],
                        help='image dimension')
    parser.add_argument('--bits', type=int, default=8,
                        help='number of bits per pixel.')
    parser.add_argument('--workers', type=int, default=8,
                        help='number of data loader workers')
    parser.add_argument('--save_interval', type=int, default=1,
                        help='number of batches to save')
    parser.add_argument('--test_interval', type=int, default=1,
                        help='number of batches to test')
    parser.add_argument('--resume_epoch', type=int, default=None,
                        help='resume training epoch')
    parser.add_argument('--test_epoch', type=int, default=None,
                        help='test epoch')
    parser.add_argument('--sample_epoch', type=int, default=None,
                        help='sample epoch')
    parser.add_argument('--sample_size', type=int, default=64,
                        help='sample size')

    parser.add_argument('--levels', type=int, default=3,
                        help='number of flow levels')
    parser.add_argument('--num_flows', type=list, default=[8, 4, 2],
                        help='number of flows per level')
    parser.add_argument('--conv_type', type=str, default='matrixexp',
                        choices=['standard', 'decomposition', 'matrixexp'],
                        help='invertible 1x1 convolution')
    parser.add_argument('--flow_type', type=str, default='matrixexp',
                        choices=['additive', 'affine', 'matrixexp'],
                        help='flow type')
    parser.add_argument('--num_blocks', type=int, default=8,
                        help='number of blocks of coupling layers')
    parser.add_argument('--hidden_channels', type=int, default=128,
                        help='hidden channels of coupling layers')

    parser.add_argument('--epochs', type=int, default=50,
                        help='number of epochs to train ')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size')
    parser.add_argument('--init_batch_size', type=int, default=512,
                        help='batch size for initialization')
    parser.add_argument('--optimizer', type=str, default='adamax',
                        choices=['adam', 'adamax'],
                        help='optimizer')
    parser.add_argument('--warmup_epoch', type=int, default=1,
                        help='warmup epoch')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--step_size', type=list, default=[40],
                        help='multi step learning rate decay')
    parser.add_argument('--lr_decay', type=float, default=0.5,
                        help='factor of learning rate decay')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='weight decay')
    parser.add_argument('--polyak', type=float, default=0.999,
                        help='polyak average')
    parser.add_argument('--seed', type=int, default=0,
                        help='seed')
    parse_args = parser.parse_args()

    """
    param_dir = ''
    if param_dir:
        with open(param_dir) as f_obj:
            parse_args.__dict__ = json.load(f_obj)
    """

    np.random.seed(parse_args.seed)
    torch.manual_seed(parse_args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(parse_args.seed)
    main(parse_args)
