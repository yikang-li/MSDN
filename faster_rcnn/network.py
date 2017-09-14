import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import math
import pdb

class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, relu=True, same_padding=False, bn=False):
        super(Conv2d, self).__init__()
        padding = int((kernel_size - 1) / 2) if same_padding else 0
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class FC(nn.Module):
    def __init__(self, in_features, out_features, relu=True):
        super(FC, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.fc(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


def save_net(fname, net):
    import h5py
    h5f = h5py.File(fname, mode='w')
    for k, v in net.state_dict().items():
        h5f.create_dataset(k, data=v.cpu().numpy())
        #print '[Saved]: {}'.format(k)


def load_net(fname, net):
    import h5py
    h5f = h5py.File(fname, mode='r')
    try:
        for k, v in net.state_dict().items():
            if k in h5f:
                param = torch.from_numpy(np.asarray(h5f[k]))
                v.copy_(param)
                print '[Copied]: {}'.format(k)
            else:
                print '[Missed]: {}'.format(k)
    except Exception as e:
        pdb.set_trace()
        print '[Loaded net not complete] Parameter[{}] Size Mismatch...'.format(k)
        


def np_to_variable(x, is_cuda=True, dtype=torch.FloatTensor):
    v = Variable(torch.from_numpy(x).type(dtype))
    if is_cuda:
        v = v.cuda()
    return v


def set_trainable(model, requires_grad):
    set_trainable_param(model.parameters(), requires_grad)

def set_trainable_param(parameters, requires_grad):
    for param in parameters:
        param.requires_grad = requires_grad


def weights_normal_init(model, dev=0.01):
    if isinstance(model, list):
        for m in model:
            weights_normal_init(m, dev)
    else:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, dev)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, dev)


def weights_MSRA_init(model):
    if isinstance(model, list):
        for m in model:
            weights_MSRA_init(m)
    else:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2./n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.001)
                if m.bias is not None:
                    m.bias.data.zero_()


def clip_gradient(model, clip_norm):
    """Computes a gradient clipping coefficient based on gradient norm."""
    totalnorm = 0
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            modulenorm = p.grad.data.norm()
            totalnorm += modulenorm ** 2
    totalnorm = np.sqrt(totalnorm)

    norm = clip_norm / max(totalnorm, clip_norm)
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            p.grad.mul_(norm)

def get_optimizer(lr, mode, args, vgg_features_var, rpn_features, hdn_features, language_features):
    """ To get the optimizer
    mode 0: training from scratch
    mode 1: training with RPN
    mode 2: resume training
    mode 3: finetune language model"""
    if mode == 0:
        set_trainable_param(rpn_features, True)
        set_trainable_param(hdn_features, True)
        set_trainable_param(language_features, True)
        if args.optimizer == 0:
            optimizer = torch.optim.SGD([
                {'params': rpn_features},
                {'params': hdn_features}, 
                {'params': language_features, 'weight_decay':0.0}
                ], lr=lr, momentum=args.momentum, weight_decay=0.0005, nesterov=args.nesterov)
        elif args.optimizer == 1:
            optimizer = torch.optim.Adam([
                {'params': rpn_features},
                {'params': hdn_features}, 
                {'params': language_features, 'weight_decay':0.0}
                ], lr=lr, weight_decay=0.0005)
        elif args.optimizer == 2:    
            optimizer = torch.optim.Adagrad([
                {'params': rpn_features},
                {'params': hdn_features}, 
                {'params': language_features, 'weight_decay':0.0}
                ], lr=lr, weight_decay=0.0005)
        else:
            raise Exception('Unrecognized optimization algorithm specified!')

    elif mode == 1:
        set_trainable_param(hdn_features, True)
        set_trainable_param(language_features, True)
        if args.optimizer == 0:
            optimizer = torch.optim.SGD([
                {'params': hdn_features},
                {'params': language_features, 'weight_decay': 0.0}
                ], lr=lr, momentum=args.momentum, weight_decay=0.0005, nesterov=args.nesterov)
        elif args.optimizer == 1:
            optimizer = torch.optim.Adam([
                {'params': hdn_features},
                {'params': language_features, 'weight_decay': 0.0}
                ], lr=lr, weight_decay=0.0005)
        elif args.optimizer == 2:    
            optimizer = torch.optim.Adagrad([
                {'params': hdn_features},
                {'params': language_features, 'weight_decay': 0.0}
                ], lr=lr, weight_decay=0.0005)
        else:
            raise Exception('Unrecognized optimization algorithm specified!')
        

    elif mode == 2:
        set_trainable_param(rpn_features, True)
        set_trainable_param(vgg_features_var, True)
        set_trainable_param(hdn_features, True)
        set_trainable_param(language_features, True)
        if args.optimizer == 0:
            optimizer = torch.optim.SGD([
                {'params': rpn_features},
                {'params': vgg_features_var, 'lr': lr * 0.1},
                {'params': hdn_features}, 
                {'params': language_features, 'weight_decay': 0.0}
                ], lr=lr, momentum=args.momentum, weight_decay=0.0005, nesterov=args.nesterov)
        elif args.optimizer == 1:
            optimizer = torch.optim.Adam([
                {'params': rpn_features},
                {'params': vgg_features_var, 'lr': lr * 0.1},
                {'params': hdn_features}, 
                {'params': language_features, 'weight_decay': 0.0}
                ], lr=lr, weight_decay=0.0005)
        elif args.optimizer == 2:    
            optimizer = torch.optim.Adagrad([
                {'params': rpn_features},
                {'params': vgg_features_var, 'lr': lr * 0.1},
                {'params': hdn_features}, 
                {'params': language_features, 'weight_decay': 0.0}
                ], lr=lr, weight_decay=0.0005)
        else:
            raise Exception('Unrecognized optimization algorithm specified!')

        

    elif mode == 3:
        set_trainable_param(rpn_features, True)
        set_trainable_param(vgg_features_var, True)
        set_trainable_param(hdn_features, True)
        set_trainable_param(language_features, True)
        if args.optimizer == 0:
            optimizer = torch.optim.SGD([
                {'params': rpn_features},
                {'params': vgg_features_var, 'lr': lr * 0.01},
                {'params': hdn_features[:-4]}, 
                {'params': hdn_features[-4:], 'lr': lr}, 
                {'params': language_features, 'weight_decay': 0.0, 'lr': lr}
                ], lr=lr * 0.1, momentum=args.momentum, weight_decay=0.0005, nesterov=args.nesterov)
        elif args.optimizer == 1:
            optimizer = torch.optim.Adam([
                {'params': rpn_features},
                {'params': vgg_features_var, 'lr': lr * 0.01},
                {'params': hdn_features[:-4]}, 
                {'params': hdn_features[-4:], 'lr': lr}, 
                {'params': language_features, 'weight_decay': 0.0, 'lr': lr}
                ], lr=lr * 0.1, weight_decay=0.0005)
        elif args.optimizer == 2:    
            optimizer = torch.optim.Adagrad([
                {'params': rpn_features},
                {'params': vgg_features_var, 'lr': lr * 0.01},
                {'params': hdn_features[:-4]}, 
                {'params': hdn_features[-4:], 'lr': lr}, 
                {'params': language_features, 'weight_decay': 0.0, 'lr': lr}
                ], lr=lr * 0.1, weight_decay=0.0005)
        else:
            raise Exception('Unrecognized optimization algorithm specified!')
        
    
    return optimizer



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class AccuracyMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.tp = 0.
        self.tf = 0.
        self.fg = 0.
        self.bg = 0.
        self.count = 0

    def update(self, tp, tf, fg, bg, count=1):
        self.tp += tp
        self.tf += tf
        self.fg += fg
        self.bg += bg
        self.count += 1

    @property
    def ture_pos(self):
        return self.tp / self.fg

    @property
    def true_neg(self):
        return self.tf / self.bg

    @property
    def foreground(self):
        return self.fg / self.count

    @property
    def background(self):
        return self.bg / self.count

