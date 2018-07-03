import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import pdb
from .cython_bbox import bbox_overlaps, bbox_intersections


def get_model_name(arguments):


    if arguments.nesterov:
        arguments.model_name += '_nesterov'

    if arguments.MPS_iter < 0:
        print 'Using random MPS iterations to training'
        arguments.model_name += '_rand_iters'
    else:
        arguments.model_name += '_{}_iters'.format(arguments.MPS_iter)


    if arguments.use_kernel_function:
        arguments.model_name += '_with_kernel'
    if arguments.load_RPN or arguments.resume_training:
        arguments.model_name += '_alt'
    else:
        arguments.model_name += '_end2end'
    if arguments.dropout:
        arguments.model_name += '_dropout'
    arguments.model_name += '_{}'.format(arguments.dataset_option)
    if arguments.disable_language_model:
        arguments.model_name += '_no_caption'
    else:
        if arguments.rnn_type == 'LSTM_im':
            arguments.model_name += '_H_LSTM'
        elif arguments.rnn_type == 'LSTM_normal':
            arguments.model_name += '_I_LSTM'
        elif arguments.rnn_type == 'LSTM_baseline':
            arguments.model_name += '_B_LSTM'
        else:
            raise Exception('Error in RNN type')
        if arguments.caption_use_bias:
            arguments.model_name += '_with_bias'
        else:
            arguments.model_name += '_no_bias'
        if arguments.caption_use_dropout > 0:
            arguments.model_name += '_with_dropout_{}'.format(arguments.caption_use_dropout).replace('.', '_')
        else:
            arguments.model_name += '_no_dropout'
        arguments.model_name += '_nembed_{}'.format(arguments.nembedding)
        arguments.model_name += '_nhidden_{}'.format(arguments.nhidden_caption)

        if arguments.region_bbox_reg:
            arguments.model_name += '_with_region_regression'

    if arguments.resume_training:
        arguments.model_name += '_resume'

    if arguments.finetune_language_model:
        arguments.model_name += '_finetune'
    if arguments.optimizer == 0:
        arguments.model_name += '_SGD'
        arguments.solver = 'SGD'
    elif arguments.optimizer == 1:
        arguments.model_name += '_Adam'
        arguments.solver = 'Adam'
    elif arguments.optimizer == 2:    
        arguments.model_name += '_Adagrad'
        arguments.solver = 'Adagrad'
    else:
        raise Exception('Unrecognized optimization algorithm specified!')

    return arguments


def group_features(net_):
    vgg_features_fix = list(net_.rpn.features.parameters())[:8]
    vgg_features_var = list(net_.rpn.features.parameters())[8:]
    vgg_feature_len = len(list(net_.rpn.features.parameters()))
    rpn_feature_len = len(list(net_.rpn.parameters())) - vgg_feature_len
    rpn_features = list(net_.rpn.parameters())[vgg_feature_len:]
    language_features = list(net_.caption_prediction.parameters())
    language_feature_len = len(language_features)
    hdn_features = list(net_.parameters())[(rpn_feature_len + vgg_feature_len):(-1 * language_feature_len)]
    print 'vgg feature length:', vgg_feature_len
    print 'rpn feature length:', rpn_feature_len
    print 'HDN feature length:', len(hdn_features)
    print 'language_feature_len:', language_feature_len
    return vgg_features_fix, vgg_features_var, rpn_features, hdn_features, language_features



def check_recall(rois, gt_objects, top_N, thres=0.5):
    overlaps = bbox_overlaps(
        np.ascontiguousarray(rois.cpu().data.numpy()[:top_N, 1:5], dtype=np.float),
        np.ascontiguousarray(gt_objects[:,:4], dtype=np.float))

    overlap_gt = np.amax(overlaps, axis=0)
    correct_cnt = np.sum(overlap_gt >= thres)
    total_cnt = overlap_gt.size 
    return correct_cnt, total_cnt


def check_relationship_recall(gt_objects, gt_relationships, 
        subject_inds, object_inds, predicate_inds, 
        subject_boxes, object_boxes, top_Ns, thres=0.5, only_predicate=False):
    # rearrange the ground truth
    gt_rel_sub_idx, gt_rel_obj_idx = np.where(gt_relationships > 0) # ground truth number
    gt_sub = gt_objects[gt_rel_sub_idx, :5]
    gt_obj = gt_objects[gt_rel_obj_idx, :5]
    gt_rel = gt_relationships[gt_rel_sub_idx, gt_rel_obj_idx]

    rel_cnt = len(gt_rel)
    rel_correct_cnt = np.zeros(len(top_Ns))
    max_topN = max(top_Ns)

    # compute the overlap
    sub_overlaps = bbox_overlaps(
        np.ascontiguousarray(subject_boxes[:max_topN], dtype=np.float),
        np.ascontiguousarray(gt_sub[:, :4], dtype=np.float))
    obj_overlaps = bbox_overlaps(
        np.ascontiguousarray(object_boxes[:max_topN], dtype=np.float),
        np.ascontiguousarray(gt_obj[:, :4], dtype=np.float))


    for idx, top_N in enumerate(top_Ns):

        for gt_id in xrange(rel_cnt):
            fg_candidate = np.where(np.logical_and(
                sub_overlaps[:top_N, gt_id] >= thres, 
                obj_overlaps[:top_N, gt_id] >= thres))[0]
            
            for candidate_id in fg_candidate:
                if only_predicate:
                    if predicate_inds[candidate_id] == gt_rel[gt_id]:
                        rel_correct_cnt[idx] += 1
                        break
                else:
                    if subject_inds[candidate_id] == gt_sub[gt_id, 4] and \
                            predicate_inds[candidate_id] == gt_rel[gt_id] and \
                            object_inds[candidate_id] == gt_obj[gt_id, 4]:

                        rel_correct_cnt[idx] += 1 
                        break
    return rel_cnt, rel_correct_cnt
