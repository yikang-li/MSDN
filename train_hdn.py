import os
import shutil
import time
import random
import numpy as np
import numpy.random as npr
import argparse


import torch

from faster_rcnn import network
from faster_rcnn.MSDN import Hierarchical_Descriptive_Model
from faster_rcnn.utils.timer import Timer
from faster_rcnn.fast_rcnn.config import cfg
from faster_rcnn.datasets.visual_genome_loader import visual_genome
from faster_rcnn.utils.HDN_utils import get_model_name, group_features

import pdb

# To log the training process
from tensorboard_logger import configure, log_value

TIME_IT = cfg.TIME_IT
parser = argparse.ArgumentParser('Options for training Hierarchical Descriptive Model in pytorch')

# Training parameters
parser.add_argument('--lr', type=float, default=0.01, metavar='LR', help='base learning rate for training')
parser.add_argument('--max_epoch', type=int, default=10, metavar='N', help='max iterations for training')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='percentage of past parameters to store')
parser.add_argument('--log_interval', type=int, default=1000, help='Interval for Logging')
parser.add_argument('--step_size', type=int, default = 2, help='Step size for reduce learning rate')
parser.add_argument('--resume_training', action='store_true', help='Resume training from the model [resume_model]')
parser.add_argument('--resume_model', type=str, default='', help='The model we resume')
parser.add_argument('--load_RPN', action='store_true', help='To end-to-end train from the scratch')
parser.add_argument('--enable_clip_gradient', action='store_true', help='Whether to clip the gradient')
parser.add_argument('--use_normal_anchors', action='store_true', help='Whether to use kmeans anchors')


# structure settings
parser.add_argument('--disable_language_model', action='store_true', help='To disable the Lanuage Model ')
parser.add_argument('--mps_feature_len', type=int, default=1024, help='The expected feature length of message passing')
parser.add_argument('--dropout', action='store_true', help='To enables the dropout')
parser.add_argument('--MPS_iter', type=int, default=1, help='Iterations for Message Passing')
parser.add_argument('--gate_width', type=int, default=128, help='The number filters for gate functions in GRU')
parser.add_argument('--nhidden_caption', type=int, default=512, help='The size of hidden feature in language model')
parser.add_argument('--nembedding', type=int, default=256, help='The size of word embedding')
parser.add_argument('--rnn_type', type=str, default='LSTM_baseline', help='Select the architecture of RNN in caption model[LSTM_im | LSTM_normal]')
parser.add_argument('--caption_use_bias', action='store_true', help='Use the flap to enable the bias term to caption model')
parser.add_argument('--caption_use_dropout', action='store_const', const=0.5, default=0., help='Set to use dropout in caption model')
parser.add_argument('--enable_bbox_reg', dest='region_bbox_reg', action='store_true')
parser.add_argument('--disable_bbox_reg', dest='region_bbox_reg', action='store_false')
parser.set_defaults(region_bbox_reg=True)
parser.add_argument('--use_kernel_function', action='store_true')
# Environment Settings
parser.add_argument('--seed', type=int, default=1, help='set seed to some constant value to reproduce experiments')
parser.add_argument('--saved_model_path', type=str, default = 'model/pretrained_models/VGG_imagenet.npy', help='The Model used for initialize')
parser.add_argument('--dataset_option', type=str, default='small', help='The dataset to use (small | normal | fat)')
parser.add_argument('--output_dir', type=str, default='./output/HDN', help='Location to output the model')
parser.add_argument('--model_name', type=str, default='HDN', help='The name for saving model.')
parser.add_argument('--nesterov', action='store_true', help='Set to use the nesterov for SGD')
parser.add_argument('--finetune_language_model', action='store_true', help='Set to disable the update of other parameters')
parser.add_argument('--optimizer', type=int, default=0, help='which optimizer used for optimize language model [0: SGD | 1: Adam | 2: Adagrad]')


parser.add_argument('--evaluate', action='store_true', help='Only use the testing mode')
args = parser.parse_args()
# Overall loss logger
overall_train_loss = network.AverageMeter()
overall_train_rpn_loss = network.AverageMeter()
overall_train_region_caption_loss = network.AverageMeter()

optimizer_select = 0



def main():
    global args, optimizer_select
    # To set the model name automatically
    print args
    lr = args.lr
    args = get_model_name(args)
    print 'Model name: {}'.format(args.model_name)

    # To set the random seed
    random.seed(args.seed)
    torch.manual_seed(args.seed + 1)
    torch.cuda.manual_seed(args.seed + 2)

    print("Loading training set and testing set..."),
    train_set = visual_genome(args.dataset_option, 'train')
    test_set = visual_genome('small', 'test')
    print("Done.")

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True, num_workers=8, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)

    # Model declaration
    net = Hierarchical_Descriptive_Model(nhidden=args.mps_feature_len,
                 n_object_cats=train_set.num_object_classes, 
                 n_predicate_cats=train_set.num_predicate_classes, 
                 n_vocab=train_set.voc_size,
                 voc_sign=train_set.voc_sign,
                 max_word_length=train_set.max_size, 
                 MPS_iter=args.MPS_iter, 
                 use_language_loss=not args.disable_language_model,
                 object_loss_weight=train_set.inverse_weight_object, 
                 predicate_loss_weight=train_set.inverse_weight_predicate,
                 dropout=args.dropout, 
                 use_kmeans_anchors=not args.use_normal_anchors, 
                 gate_width = args.gate_width, 
                 nhidden_caption = args.nhidden_caption, 
                 nembedding = args.nembedding,
                 rnn_type=args.rnn_type, 
                 rnn_droptout=args.caption_use_dropout, rnn_bias=args.caption_use_bias, 
                 use_region_reg = args.region_bbox_reg, 
                 use_kernel = args.use_kernel_function)

    params = list(net.parameters())
    for param in params:
        print param.size()
    print net 

    # To group up the features
    vgg_features_fix, vgg_features_var, rpn_features, hdn_features, language_features = group_features(net)

    # Setting the state of the training model
    net.cuda()
    net.train()
    logger_path = "log/logger/{}".format(args.model_name)
    if os.path.exists(logger_path):
        shutil.rmtree(logger_path)
    configure(logger_path, flush_secs=5) # setting up the logger


    network.set_trainable(net, False)
    #  network.weights_normal_init(net, dev=0.01)
    if args.finetune_language_model:
        print 'Only finetuning the language model from: {}'.format(args.resume_model)
        args.train_all = False
        if len(args.resume_model) == 0:
            raise Exception('[resume_model] not specified')
        network.load_net(args.resume_model, net)
        optimizer_select = 3
        

    elif args.load_RPN:
        print 'Loading pretrained RPN: {}'.format(args.saved_model_path)
        args.train_all = False
        network.load_net(args.saved_model_path, net.rpn)
        net.reinitialize_fc_layers()
        optimizer_select = 1       


    elif args.resume_training:
        print 'Resume training from: {}'.format(args.resume_model)
        if len(args.resume_model) == 0:
            raise Exception('[resume_model] not specified')
        network.load_net(args.resume_model, net)
        args.train_all = True
        optimizer_select = 2

    else:
        print 'Training from scratch.'
        net.rpn.initialize_parameters()
        net.reinitialize_fc_layers()
        optimizer_select = 0
        args.train_all = True

    optimizer = network.get_optimizer(lr,optimizer_select, args, 
                vgg_features_var, rpn_features, hdn_features, language_features)

    target_net = net
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)


    top_Ns = [50, 100]
    best_recall = np.zeros(len(top_Ns))


    if args.evaluate:
        recall = test(test_loader, net, top_Ns)
        print('======= Testing Result =======') 
        for idx, top_N in enumerate(top_Ns):
            print('[Recall@{top_N:d}] {recall:2.3f}%% (best: {best_recall:2.3f}%%)'.format(
                top_N=top_N, recall=recall[idx] * 100, best_recall=best_recall[idx] * 100))

        print('==============================')
    else:
        for epoch in range(0, args.max_epoch):
            # Training
            train(train_loader, target_net, optimizer, epoch)
            # snapshot the state
            save_name = os.path.join(args.output_dir, '{}_epoch_{}.h5'.format(args.model_name, epoch))
            network.save_net(save_name, net)
            print('save model: {}'.format(save_name))


            # Testing
            # network.set_trainable(net, False) # Without backward(), requires_grad takes no effect

            recall = test(test_loader, net, top_Ns)

            if np.all(recall > best_recall):
                best_recall = recall
                save_name = os.path.join(args.output_dir, '{}_best.h5'.format(args.model_name))
                network.save_net(save_name, net)
                print('\nsave model: {}'.format(save_name))

            print('Epoch[{epoch:d}]:'.format(epoch = epoch)), 
            for idx, top_N in enumerate(top_Ns):
                print('\t[Recall@{top_N:d}] {recall:2.3f}%% (best: {best_recall:2.3f}%%)'.format(
                    top_N=top_N, recall=recall[idx] * 100, best_recall=best_recall[idx] * 100)),

            # updating learning policy
            if epoch % args.step_size == 0 and epoch > 0:
                lr /= 10
                args.lr = lr
                print '[learning rate: {}]'.format(lr)
            
                args.enable_clip_gradient = False
                if not args.finetune_language_model:
                    args.train_all = True
                    optimizer_select = 2
                # update optimizer and correponding requires_grad state   
                optimizer = network.get_optimizer(lr, optimizer_select, args, 
                            vgg_features_var, rpn_features, hdn_features, language_features)

        



def train(train_loader, target_net, optimizer, epoch):
    global args
    # Overall loss logger
    global overall_train_loss
    global overall_train_rpn_loss
    global overall_train_region_caption_loss

    batch_time = network.AverageMeter()
    data_time = network.AverageMeter()
    # Total loss
    train_loss = network.AverageMeter()
    # object related loss
    train_obj_cls_loss = network.AverageMeter()
    train_obj_box_loss = network.AverageMeter()
    # relationship cls loss
    train_pred_cls_loss = network.AverageMeter()
    # region captioning related loss
    train_region_caption_loss = network.AverageMeter()
    train_region_box_loss = network.AverageMeter()
    train_region_objectiveness_loss = network.AverageMeter()
    # RPN loss
    train_rpn_loss = network.AverageMeter()
    # object
    accuracy_obj = network.AccuracyMeter()
    accuracy_pred = network.AccuracyMeter()
    accuracy_reg = network.AccuracyMeter()

    target_net.train()
    end = time.time()
    for i, (im_data, im_info, gt_objects, gt_relationships, gt_regions) in enumerate(train_loader):
        # measure the data loading time
        data_time.update(time.time() - end)
        target_net(im_data, im_info, gt_objects.numpy()[0], gt_relationships.numpy()[0], gt_regions.numpy()[0])

        # Determine the loss function
        if args.train_all:
            loss = target_net.loss + target_net.rpn.loss
        elif args.finetune_language_model:
            loss = target_net.loss_region_box + target_net.region_caption_loss
        else:
            loss = target_net.loss

        loss += target_net.objectiveness_loss

        train_loss.update(target_net.loss.data.cpu().numpy()[0], im_data.size(0))
        train_obj_cls_loss.update(target_net.cross_entropy_object.data.cpu().numpy()[0], im_data.size(0))
        train_obj_box_loss.update(target_net.loss_obj_box.data.cpu().numpy()[0], im_data.size(0))
        train_pred_cls_loss.update(target_net.cross_entropy_predicate.data.cpu().numpy()[0], im_data.size(0))
        train_region_caption_loss.update(target_net.region_caption_loss.data.cpu().numpy()[0], im_data.size(0))
        train_rpn_loss.update(target_net.rpn.loss.data.cpu().numpy()[0], im_data.size(0))
        overall_train_loss.update(target_net.loss.data.cpu().numpy()[0], im_data.size(0))
        overall_train_rpn_loss.update(target_net.rpn.loss.data.cpu().numpy()[0], im_data.size(0))
        overall_train_region_caption_loss.update(target_net.region_caption_loss.data.cpu().numpy()[0], im_data.size(0))
        accuracy_obj.update(target_net.tp, target_net.tf, target_net.fg_cnt, target_net.bg_cnt)
        accuracy_pred.update(target_net.tp_pred, target_net.tf_pred, target_net.fg_cnt_pred, target_net.bg_cnt_pred)
        accuracy_reg.update(target_net.tp_reg, target_net.tf_reg, target_net.fg_cnt_reg, target_net.bg_cnt_reg)

        if args.region_bbox_reg:
            train_region_box_loss.update(target_net.loss_region_box.data.cpu().numpy()[0], im_data.size(0))

        train_region_objectiveness_loss.update(target_net.objectiveness_loss.data.cpu().numpy()[0], im_data.size(0))


        optimizer.zero_grad()
        loss.backward()
        if args.enable_clip_gradient:
            network.clip_gradient(target_net, 10.)
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # Logging the training loss
        if  (i + 1) % args.log_interval == 0:
            print('Epoch: [{0}][{1}/{2}] [lr: {lr}] [Solver: {solver}]\n'
                  '\tBatch_Time: {batch_time.avg: .3f}s\t'
                  'FRCNN Loss: {loss.avg: .4f}\t'
                  'RPN Loss: {rpn_loss.avg: .4f}'.format(
                   epoch, i + 1, len(train_loader), batch_time=batch_time,lr=args.lr, 
                   loss=train_loss, rpn_loss=train_rpn_loss, solver=args.solver))


            print('\t[Loss]\tobj_cls_loss: %.4f\tobj_box_loss: %.4f' %
                  (train_obj_cls_loss.avg, train_obj_box_loss.avg)),
            print('\tpred_cls_loss: %.4f,' % (train_pred_cls_loss.avg)),

            if not args.disable_language_model:
                print ('\tcaption_loss: %.4f,' % (train_region_caption_loss.avg)),
            if args.region_bbox_reg:
                print('\tregion_box_loss: %.4f, ' % (train_region_box_loss.avg)),
            print('\tregion_objectness_loss: %.4f' % (train_region_objectiveness_loss.avg)),
            
            print('\n\t[object]\ttp: %.2f, \ttf: %.2f, \tfg/bg=(%d/%d)' %
                  (accuracy_obj.ture_pos*100., accuracy_obj.true_neg*100., accuracy_obj.foreground, accuracy_obj.background))
            print('\t[predicate]\ttp: %.2f, \ttf: %.2f, \tfg/bg=(%d/%d)' %
                  (accuracy_pred.ture_pos*100., accuracy_pred.true_neg*100., accuracy_pred.foreground, accuracy_pred.background))
            print('\t[region]\ttp: %.2f, \ttf: %.2f, \tfg/bg=(%d/%d)' %
                  (accuracy_reg.ture_pos*100., accuracy_reg.true_neg*100., accuracy_reg.foreground, accuracy_reg.background))

            # logging to tensor board
            log_value('FRCNN loss', overall_train_loss.avg, overall_train_loss.count)
            log_value('RPN_loss loss', overall_train_rpn_loss.avg, overall_train_rpn_loss.count)
            log_value('caption loss', overall_train_region_caption_loss.avg, overall_train_region_caption_loss.count)


    



def test(test_loader, net, top_Ns):

    global args

    print '========== Testing ======='
    net.eval()
    # For efficiency inference
    languge_state = net.use_language_loss
    region_reg_state = net.use_region_reg
    net.use_language_loss = False
    net.use_region_reg = False

    
    rel_cnt = 0.
    rel_cnt_correct = np.zeros(len(top_Ns))

    batch_time = network.AverageMeter()
    end = time.time()
    for i, (im_data, im_info, gt_objects, gt_relationships, gt_regions) in enumerate(test_loader):
        # Forward pass
        total_cnt_t, rel_cnt_correct_t = net.evaluate(
            im_data, im_info, gt_objects.numpy()[0], gt_relationships.numpy()[0], gt_regions.numpy()[0], 
            top_Ns = top_Ns, nms=True)
        rel_cnt += total_cnt_t
        rel_cnt_correct += rel_cnt_correct_t
        batch_time.update(time.time() - end)
        end = time.time()
        if (i + 1) % 500 == 0 and i > 0:
            for idx, top_N in enumerate(top_Ns):
                print '[%d/%d][Evaluation] Top-%d Recall: %2.3f%%' % (
                    i+1, len(test_loader), top_N, rel_cnt_correct[idx] / float(rel_cnt) * 100)

    recall = rel_cnt_correct / rel_cnt
    print '====== Done Testing ===='
    # Restore the related states
    net.use_language_loss = languge_state
    net.use_region_reg = region_reg_state

    return recall


if __name__ == '__main__':
    main()
