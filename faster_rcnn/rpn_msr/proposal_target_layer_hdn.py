# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------

import yaml
import numpy as np
import numpy.random as npr
import pdb

from ..utils.cython_bbox import bbox_overlaps, bbox_intersections

# TODO: make fast_rcnn irrelevant
# >>>> obsolete, because it depends on sth outside of this project
from ..fast_rcnn.config import cfg
from ..fast_rcnn.bbox_transform import bbox_transform

# <<<< obsolete

DEBUG = False


#  object_rois, object_labels, bbox_targets, bbox_inside_weights, bbox_outside_weights, mat_object, \
#              phrase_rois, phrase_label, mat_phrase, region_rois, region_seq, mat_region = \
#              proposal_target_layer_py(object_rois, region_rois, gt_objects, gt_relationships,
#                  gt_regions, n_classes_obj, n_classes_pred, is_training)




def proposal_target_layer(object_rois, region_rois, gt_objects, gt_relationships, 
                gt_regions, n_classes_obj, voc_eos, is_training, graph_generation=False):

    #     object_rois:  (1 x H x W x A, 5) [0, x1, y1, x2, y2]
    #     region_rois:  (1 x H x W x A, 5) [0, x1, y1, x2, y2]
    #     gt_objects:   (G_obj, 5) [x1 ,y1 ,x2, y2, obj_class] float
    #     gt_relationships: (G_obj, G_obj) [pred_class] int (-1 for no relationship)
    #     gt_regions:   (G_region, 4+40) [x1, y1, x2, y2, word_index] (imdb.eos for padding)
    #     # gt_ishard: (G_region, 4+40) {0 | 1} 1 indicates hard
    #     # dontcare_areas: (D, 4) [ x1, y1, x2, y2]
    #     n_classes_obj
    #     n_classes_pred
    #     is_training to indicate whether in training scheme

    # Proposal ROIs (0, x1, y1, x2, y2) coming from RPN
    # (i.e., rpn.proposal_layer.ProposalLayer), or any other source

    # TODO(rbg): it's annoying that sometimes I have extra info before
    # and other times after box coordinates -- normalize to one format

    # Include ground-truth boxes in the set of candidate rois

    # assert is_training == True, 'Evaluation Code haven\'t been implemented'

    

    # Sample rois with classification labels and bounding box regression
    # targets
    if is_training:
        all_rois = object_rois
        zeros = np.zeros((gt_objects.shape[0], 1), dtype=gt_objects.dtype)
        all_rois = np.vstack(
            (all_rois, np.hstack((zeros, gt_objects[:, :4])))
        )

        all_rois_region = region_rois 
        zeros = np.zeros((gt_regions.shape[0], 1), dtype=gt_regions.dtype)
        all_rois_region = np.vstack(
            (all_rois_region, np.hstack((zeros, gt_regions[:, :4])))
        )

        # Sanity check: single batch only
        assert np.all(all_rois[:, 0] == 0), \
            'Only single item batches are supported'

        object_labels, object_rois, bbox_targets, bbox_inside_weights, \
            phrase_labels, phrase_rois, \
            region_labels, region_rois, bbox_targets_region, bbox_inside_weights_region, \
                mat_object, mat_phrase, mat_region = _sample_rois(all_rois, all_rois_region, \
                    gt_objects, gt_relationships, gt_regions, 1, n_classes_obj, voc_eos, is_training)


        # assert region_labels.shape[1] == cfg.TRAIN.LANGUAGE_MAX_LENGTH
        object_labels = object_labels.reshape(-1, 1)
        bbox_targets = bbox_targets.reshape(-1, n_classes_obj * 4)
        bbox_targets_region = bbox_targets_region.reshape(-1, 4)
        bbox_inside_weights = bbox_inside_weights.reshape(-1, n_classes_obj * 4)
        bbox_inside_weights_region = bbox_inside_weights_region.reshape(-1, 4)
        phrase_labels = phrase_labels.reshape(-1, 1)
        bbox_outside_weights = np.array(bbox_inside_weights > 0).astype(np.float32)
        bbox_outside_weights_region = np.array(bbox_inside_weights_region > 0).astype(np.float32)
    else:
        object_rois, phrase_rois, region_rois, mat_object, mat_phrase, mat_region  = \
                    _setup_connection(object_rois, region_rois, graph_generation=graph_generation)
        object_labels, bbox_targets, bbox_inside_weights, bbox_outside_weights, phrase_labels, region_labels, \
             bbox_targets_region, bbox_inside_weights_region, bbox_outside_weights_region= [None] * 9
    # print 'region_roi', region_roi
    # print 'object_rois'
    # print object_rois
    # print 'phrase_rois'
    # print phrase_rois

    if DEBUG:
        # print 'region_roi'
        # print region_roi
        # print 'object num fg: {}'.format((object_labels > 0).sum())
        # print 'object num bg: {}'.format((object_labels == 0).sum())
        # print 'relationship num fg: {}'.format((phrase_labels > 0).sum())
        # print 'relationship num bg: {}'.format((phrase_labels == 0).sum())
        count = 1
        fg_num = (object_labels > 0).sum()
        bg_num = (object_labels == 0).sum()
        print 'object num fg avg: {}'.format(fg_num / count)
        print 'object num bg avg: {}'.format(bg_num / count)
        print 'ratio: {:.3f}'.format(float(fg_num) / float(bg_num))
        count_rel = 1
        fg_num_rel = (phrase_labels > 0).sum()
        bg_num_rel = (phrase_labels == 0).sum()
        print 'relationship num fg avg: {}'.format(fg_num_rel / count_rel)
        print 'relationship num bg avg: {}'.format(bg_num_rel / count_rel)
        print 'ratio: {:.3f}'.format(float(fg_num_rel) / float(bg_num_rel))
        # print mat_object.shape
        # print mat_phrase.shape
        # print 'region_roi'
        # print region_roi

    # mps_object [object_batchsize, 2, n_phrase] : the 2 channel means inward(object) and outward(subject) list
    # mps_phrase [phrase_batchsize, 2 + n_region]
    # mps_region [region_batchsize, n_phrase]
    assert object_rois.shape[1] == 5
    assert phrase_rois.shape[1] == 5

    return object_labels, object_rois, bbox_targets, bbox_inside_weights, bbox_outside_weights, mat_object, \
            phrase_labels, phrase_rois, mat_phrase, \
            region_labels, region_rois, \
            bbox_targets_region, bbox_inside_weights_region, bbox_outside_weights_region, mat_region \


def _get_bbox_regression_labels(bbox_target_data, num_classes):
    """Bounding-box regression targets (bbox_target_data) are stored in a
    compact form N x (class, tx, ty, tw, th)

    This function expands those targets into the 4-of-4*K representation used
    by the network (i.e. only one class has non-zero targets).

    Returns:
        bbox_target (ndarray): N x 4K blob of regression targets
        bbox_inside_weights (ndarray): N x 4K blob of loss weights
    """

    clss = bbox_target_data[:, 0]
    bbox_targets = np.zeros((clss.size, 4 * num_classes), dtype=np.float32)
    bbox_inside_weights = np.zeros(bbox_targets.shape, dtype=np.float32)
    inds = np.where(clss > 0)[0]
    for ind in inds:
        cls = int(clss[ind])
        start = 4 * cls
        end = start + 4
        bbox_targets[ind, start:end] = bbox_target_data[ind, 1:]
        bbox_inside_weights[ind, start:end] = cfg.TRAIN.BBOX_INSIDE_WEIGHTS
    return bbox_targets, bbox_inside_weights


def _compute_targets(ex_rois, gt_rois, labels):
    """Compute bounding-box regression targets for an image."""

    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] == 4

    targets = bbox_transform(ex_rois, gt_rois)
    # if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
    #     # Optionally normalize targets by a precomputed mean and stdev
    #     targets = ((targets - np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS))
    #                / np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS))
    return np.hstack(
        (labels[:, np.newaxis], targets)).astype(np.float32, copy=False)


def _sample_rois(object_rois, region_rois, gt_objects, gt_relationships, gt_regions, num_images, num_classes, voc_eos, is_training):
    """Generate a random sample of RoIs comprising foreground and background
    examples.
    """
    # overlaps: (rois x gt_boxes)

    rois_per_image = cfg.TRAIN.BATCH_SIZE / num_images
    fg_rois_per_image = np.round(cfg.TRAIN.FG_FRACTION * rois_per_image)

    overlaps = bbox_overlaps(
        np.ascontiguousarray(object_rois[:, 1:5], dtype=np.float),
        np.ascontiguousarray(gt_objects[:, :4], dtype=np.float))
    gt_assignment = overlaps.argmax(axis=1)
    max_overlaps = overlaps.max(axis=1)
    labels = gt_objects[gt_assignment, 4]

    # Select foreground RoIs as those with >= FG_THRESH overlap
    fg_inds = np.where(max_overlaps >= cfg.TRAIN.FG_THRESH)[0]

    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_inds = np.where((max_overlaps < cfg.TRAIN.BG_THRESH_HI) &
                       (max_overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]

    # Guard against the case when an image has fewer than fg_rois_per_image
    # foreground RoIs
    fg_rois_per_this_image = int(min(fg_rois_per_image, fg_inds.size))
    # fg_rois_per_this_image = int(min(bg_inds.size, fg_inds.size))
    # Sample foreground regions without replacement
    if fg_inds.size > 0:
        fg_inds = npr.choice(fg_inds, size=fg_rois_per_this_image, replace=False)

    # Compute number of background RoIs to take from this image (guarding
    # against there being fewer than desired)
    bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
    bg_rois_per_this_image = min(bg_rois_per_this_image, bg_inds.size)
    # bg_rois_per_this_image = fg_rois_per_this_image
    # Sample background regions without replacement
    if bg_inds.size > 0:
        bg_inds = npr.choice(bg_inds, size=bg_rois_per_this_image, replace=False)

    # The indices that we're selecting (both fg and bg)
    keep_inds = np.append(fg_inds, bg_inds)
    # Select sampled values from various arrays:
    labels = labels[keep_inds]
    # Clamp labels for the background RoIs to 0
    labels[fg_rois_per_this_image:] = 0
    rois = object_rois[keep_inds]

    bbox_target_data = _compute_targets(
        rois[:, 1:5], gt_objects[gt_assignment[keep_inds], :4], labels)

    bbox_targets, bbox_inside_weights = \
        _get_bbox_regression_labels(bbox_target_data, num_classes)

#### prepare relationships targets


    rel_per_image = int(cfg.TRAIN.BATCH_SIZE_RELATIONSHIP / num_images)
    rel_bg_num = rel_per_image
    if fg_inds.size > 0:
        assert fg_inds.size == fg_inds.shape[0]
        id_i, id_j = np.meshgrid(xrange(fg_inds.size), xrange(fg_inds.size), indexing='ij') # Grouping the input object rois
        id_i = id_i.reshape(-1) 
        id_j = id_j.reshape(-1)
        pair_labels = gt_relationships[gt_assignment[fg_inds[id_i]], gt_assignment[fg_inds[id_j]]]
        fg_id_rel = np.where(pair_labels > 0)[0]
        rel_fg_num = fg_id_rel.size
        rel_fg_num = int(min(np.round(rel_per_image * cfg.TRAIN.FG_FRACTION_RELATIONSHIP), rel_fg_num))
        # print 'rel_fg_num'
        # print rel_fg_num
        if rel_fg_num > 0:
            fg_id_rel = npr.choice(fg_id_rel, size=rel_fg_num, replace=False)
        else:
            fg_id_rel = np.empty(0, dtype=int)
        rel_labels_fg = pair_labels[fg_id_rel]
        sub_assignment_fg = id_i[fg_id_rel]
        obj_assignment_fg = id_j[fg_id_rel]
        sub_list_fg = fg_inds[sub_assignment_fg]
        obj_list_fg = fg_inds[obj_assignment_fg]
        rel_bg_num = rel_per_image - rel_fg_num

    phrase_labels = np.zeros(rel_bg_num, dtype=np.float)
    sub_assignment = npr.choice(xrange(keep_inds.size), size=rel_bg_num, replace=True)
    obj_assignment = npr.choice(xrange(keep_inds.size), size=rel_bg_num, replace=True)
    sub_list = keep_inds[sub_assignment]
    obj_list = keep_inds[obj_assignment]

    if fg_inds.size > 0:
        phrase_labels = np.append(phrase_labels, rel_labels_fg)
        sub_list = np.append(sub_list, sub_list_fg)
        obj_list = np.append(obj_list, obj_list_fg)
        sub_assignment = np.append(sub_assignment, sub_assignment_fg)
        obj_assignment = np.append(obj_assignment, obj_assignment_fg)

    phrase_rois = box_union(object_rois[sub_list, :], object_rois[obj_list, :])

### prepare region targets
    region_labels, region_rois, mat_phrase_part, mat_region, bbox_targets_region, bbox_inside_weight_region = \
                    _sample_regions(region_rois, phrase_rois, gt_regions, num_images, voc_eos)


### prepare connection matrix
    mat_object, mat_phrase = _prepare_mat(sub_assignment, obj_assignment, keep_inds.size)
    mat_phrase = np.concatenate((mat_phrase, mat_phrase_part), 1)

    return labels, rois, bbox_targets, bbox_inside_weights, \
           phrase_labels, phrase_rois, \
           region_labels, region_rois, bbox_targets_region, bbox_inside_weight_region, \
           mat_object, mat_phrase, mat_region


def _setup_connection(object_rois, region_rois, graph_generation=False):
    """Generate a random sample of RoIs comprising foreground and background
    examples.
    """
    # overlaps: (rois x gt_boxes)
    roi_num = cfg.TEST.BBOX_NUM
    keep_inds = np.array(range(min(roi_num, object_rois.shape[0])))
    roi_num = len(keep_inds)
    rois = object_rois[keep_inds]

    region_roi_entire = np.concatenate((np.amin(object_rois[:, :3], 0), np.amax(object_rois[:, 3:5], 0)), 0)
    region_rois = np.vstack((region_roi_entire, region_rois))

    id_i, id_j = _generate_pairs(keep_inds) # Grouping the input object rois and remove the diagonal items
    phrase_rois = box_union(object_rois[id_i, :], object_rois[id_j, :])
    # print 'before union', object_rois[id_i[0], :], object_rois[id_j[0], :]
    # print 'after union', phrase_rois[0, :]
### prepare connection matrix
    mat_object, mat_phrase = _prepare_mat(id_i, id_j, rois.shape[0])


    region_num = cfg.TEST.REGION_NUM
    overlaps_phrase = bbox_intersections(
        np.ascontiguousarray(region_rois[:, 1:5], dtype=np.float),
        np.ascontiguousarray(phrase_rois[:, 1:5], dtype=np.float))

    max_overlaps_phrase = overlaps_phrase.max(axis=1)

    if graph_generation:
        keep_inds = np.where(max_overlaps_phrase >= cfg.PHRASE_REGION_OVERLAP_THRESH)[0]
    else:
        keep_inds = range(region_rois.shape[0])

    if len(keep_inds) > region_num:
        keep_inds = npr.choice(keep_inds, size=region_num, replace=False)

    region_rois = region_rois[keep_inds, :]

    mat_region = (overlaps_phrase[keep_inds, :] > cfg.PHRASE_REGION_OVERLAP_THRESH).astype(np.int64)
    mat_phrase = np.concatenate((mat_phrase, mat_region.transpose()), 1)

    return rois, phrase_rois, region_rois, mat_object, mat_phrase, mat_region

def box_union(box1, box2):
    return np.concatenate((np.minimum(box1[:, :3], box2[:, :3]), np.maximum(box1[:, 3:], box2[:, 3:])), 1)


def _sample_regions(region_rois, phrase_rois, gt_regions, num_images, voc_eos):
    """Generate a random sample of RoIs comprising foreground and background
    examples.
    """
    # overlaps: (rois x gt_regions)
    rois_per_image = cfg.TRAIN.BATCH_SIZE_REGION / num_images
    fg_rois_per_image = np.round(cfg.TRAIN.FG_FRACTION_REGION * rois_per_image)
    overlaps_gt = bbox_overlaps(
        np.ascontiguousarray(region_rois[:, 1:5], dtype=np.float),
        np.ascontiguousarray(gt_regions[:, :4], dtype=np.float))
    # gt_assignment = overlaps_gt.argmax(axis=1)
    max_overlaps_gt = overlaps_gt.max(axis=1)
    # labels = gt_regions[gt_assignment, 4:]

    overlaps_phrase = bbox_intersections(
        np.ascontiguousarray(region_rois[:, 1:5], dtype=np.float),
        np.ascontiguousarray(phrase_rois[:, 1:5], dtype=np.float))
    # phrase_assignment = overlaps_phrase.argmax(axis=1)
    max_overlaps_phrase = overlaps_phrase.max(axis=1)

    # Select foreground RoIs as those with >= FG_THRESH overlap
    # fg_inds = np.where(np.logical_and(max_overlaps_gt >= cfg.TRAIN.FG_THRESH_REGION,
    #                                   max_overlaps_phrase >= cfg.PHRASE_REGION_OVERLAP_THRESH))[0]
    fg_inds = np.where(max_overlaps_gt >= cfg.TRAIN.FG_THRESH_REGION)[0]

    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)

    # print 'bg original:', np.sum((max_overlaps_gt < cfg.TRAIN.BG_THRESH_HI_REGION) & (max_overlaps_gt >= cfg.TRAIN.BG_THRESH_LO_REGION))
    # print 'connections:', np.sum((max_overlaps_phrase >= cfg.PHRASE_REGION_OVERLAP_THRESH))

    # bg_inds = np.where(np.logical_and(
    #     (max_overlaps_gt < cfg.TRAIN.BG_THRESH_HI_REGION) & (max_overlaps_gt >= cfg.TRAIN.BG_THRESH_LO_REGION),
    #     (max_overlaps_phrase >= cfg.PHRASE_REGION_OVERLAP_THRESH)))[0]
    bg_inds = np.where(
        (max_overlaps_gt < cfg.TRAIN.BG_THRESH_HI_REGION) & (max_overlaps_gt >= cfg.TRAIN.BG_THRESH_LO_REGION))[0]

    # print 'bg_candidate:', len(bg_inds)
    # Guard against the case when an image has fewer than fg_rois_per_image
    # foreground RoIs
    fg_rois_per_this_image = int(min(fg_rois_per_image, fg_inds.size))
    # fg_rois_per_this_image = int(min(bg_inds.size, fg_inds.size))
    # Sample foreground regions without replacement
    if fg_inds.size > 0:
        fg_inds = npr.choice(fg_inds, size=fg_rois_per_this_image, replace=False)

    # Compute number of background RoIs to take from this image (guarding
    # against there being fewer than desired)
    # pdb.set_trace()

    if bg_inds.size == 0:
        keep_inds = fg_inds
        print 'No background in this instance'
    else:
        bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
        bg_rois_per_this_image = min(bg_rois_per_this_image, bg_inds.size)
        bg_inds = npr.choice(bg_inds, size=bg_rois_per_this_image, replace=False)
        # The indices that we're selecting (both fg and bg)
        keep_inds = np.append(fg_inds, bg_inds)

    # Select sampled values from various arrays:
    labels = np.zeros((len(keep_inds), gt_regions.shape[1] - 4), dtype=np.int64)
    # Here we randomly select regions overlapped with proposed ROI more than 0.7
    gt_assignment = np.zeros(fg_rois_per_this_image, dtype=np.int64)
    for i in range(fg_rois_per_this_image):
        gt_assignment[i] = npr.choice(np.where(overlaps_gt[fg_inds[i]] > cfg.TRAIN.FG_THRESH_REGION)[0], size=1)
        labels[i] = gt_regions[gt_assignment[i], 4:]

    # add start label to background and padding them with <end> sign
    labels[fg_rois_per_this_image:, 0].fill(voc_eos['start'])
    labels[fg_rois_per_this_image:, 1:].fill(voc_eos['end'])
    rois = region_rois[keep_inds]

    mat_region = (overlaps_phrase[keep_inds, :] > cfg.PHRASE_REGION_OVERLAP_THRESH).astype(np.int64)
    mat_phrase_part = mat_region.transpose()

    targets_fg = bbox_transform(rois[:fg_rois_per_this_image, 1:5], gt_regions[gt_assignment, :4])
    bbox_inside_weights_fg = np.ones(targets_fg.shape, dtype=np.float32) * cfg.TRAIN.BBOX_INSIDE_WEIGHTS
    targets_bg = np.zeros((bg_inds.size, targets_fg.shape[1]), dtype=np.float32)
    bbox_inside_weight_bg = np.zeros(targets_bg.shape, dtype=np.float32)
    bbox_targets = np.vstack([targets_fg, targets_bg])
    bbox_inside_weight = np.vstack([bbox_inside_weights_fg, bbox_inside_weight_bg])

    # inds = np.where(clss > 0)[0]
    # for ind in inds:
    #     cls = int(clss[ind])
    #     start = 4 * cls
    #     end = start + 4
    #     bbox_targets[ind, start:end] = bbox_target_data[ind, 1:]
    #     bbox_inside_weights[ind, start:end] = cfg.TRAIN.BBOX_INSIDE_WEIGHTS


    return labels, rois, mat_phrase_part, mat_region, bbox_targets, bbox_inside_weight




def _prepare_mat(sub_list, obj_list, object_batchsize):
    # mps_object [object_batchsize, 2, n_phrase] : the 2 channel means inward(object) and outward(subject) list
    # mps_phrase [phrase_batchsize, 2 + n_region]
    # mps_region [region_batchsize, n_phrase]

    
    phrase_batchsize = sub_list.size
    # print 'phrase_batchsize', phrase_batchsize

    mat_object = np.zeros((object_batchsize, 2, phrase_batchsize), dtype=np.int64)
    mat_phrase = np.zeros((phrase_batchsize, 2), dtype=np.int64)
    mat_phrase[:, 0] = sub_list
    mat_phrase[:, 1] = obj_list

    for i in xrange(phrase_batchsize):
        mat_object[sub_list[i], 0, i] = 1
        mat_object[obj_list[i], 1, i] = 1

    return mat_object, mat_phrase

def _generate_pairs(ids):
    id_i, id_j = np.meshgrid(ids, ids, indexing='ij') # Grouping the input object rois
    id_i = id_i.reshape(-1) 
    id_j = id_j.reshape(-1)
    # remove the diagonal items
    id_num = len(ids)
    diagonal_items = np.array(range(id_num))
    diagonal_items = diagonal_items * id_num + diagonal_items
    all_id = range(len(id_i))
    selected_id = np.setdiff1d(all_id, diagonal_items)
    id_i = id_i[selected_id]
    id_j = id_j[selected_id]

    return id_i, id_j
