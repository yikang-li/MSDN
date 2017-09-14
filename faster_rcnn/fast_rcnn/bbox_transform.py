# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import numpy as np
from sympy.physics.paulialgebra import delta
from config import cfg

np.seterr(all='warn')

def bbox_transform(ex_rois, gt_rois):

    # print 'ex_rois', ex_rois
    # print 'gt_rois', gt_rois


    ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
    ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
    ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
    ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights

    # print 'ex_widths', ex_widths
    # print 'ex_heights', ex_heights
    # print 'ex_ctr_x', ex_ctr_x
    # print 'ex_ctr_y', ex_ctr_y


    gt_widths = gt_rois[:, 2] - gt_rois[:, 0] + 1.0
    gt_heights = gt_rois[:, 3] - gt_rois[:, 1] + 1.0
    gt_ctr_x = gt_rois[:, 0] + 0.5 * gt_widths
    gt_ctr_y = gt_rois[:, 1] + 0.5 * gt_heights


    # print 'gt_widths', gt_widths
    # print 'gt_heights', gt_heights
    # print 'gt_ctr_x', gt_ctr_x
    # print 'gt_ctr_y', gt_ctr_y


    targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = np.log(gt_widths / ex_widths)
    targets_dh = np.log(gt_heights / ex_heights)

    # print 'targets_dx', targets_dx.mean(), targets_dx.std()
    # print 'targets_dy', targets_dy.mean(), targets_dy.std()
    # print 'targets_dw', targets_dw.mean(), targets_dw.std()
    # print 'targets_dh', targets_dh.mean(), targets_dh.std()


    targets = np.vstack(
        (targets_dx, targets_dy, targets_dw, targets_dh)).transpose()


    if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
        targets = ((targets - np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS))
                   / np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS))

    # print 'targets_dx(normalized)', targets[:, 0].mean(), targets[:, 0].std()
    # print 'targets_dy(normalized)', targets[:, 1].mean(), targets[:, 1].std()
    # print 'targets_dw(normalized)', targets[:, 2].mean(), targets[:, 2].std()
    # print 'targets_dh(normalized)', targets[:, 3].mean(), targets[:, 3].std()

    return targets


def bbox_transform_inv(boxes, deltas):
    return bbox_transform_inv_hdn(boxes, deltas)


def bbox_transform_inv_hdn(boxes, deltas):
    if boxes.shape[0] == 0:
        return np.zeros((0,), dtype=deltas.dtype)

    boxes = boxes.astype(deltas.dtype, copy=False)

    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
        deltas = deltas * np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS) + np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS)

    dx = deltas[:, 0::4]
    dy = deltas[:, 1::4]
    dw = deltas[:, 2::4]
    dh = deltas[:, 3::4]

    pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
    pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
    pred_w = np.exp(dw) * widths[:, np.newaxis]
    pred_h = np.exp(dh) * heights[:, np.newaxis]

    pred_boxes = np.zeros(deltas.shape, dtype=deltas.dtype)
    # x1
    pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
    # y1
    pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
    # x2
    pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w - 1.0 
    # y2
    pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h - 1.0 

    return pred_boxes


def clip_boxes(boxes, im_shape):
    """
    Clip boxes to image boundaries.
    """
    if boxes.shape[0] == 0:
        return boxes

    # x1 >= 0
    boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], im_shape[1] - 1), 0)
    # y1 >= 0
    boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], im_shape[0] - 1), 0)
    # x2 < im_shape[1]
    boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], im_shape[1] - 1), 0)
    # y2 < im_shape[0]
    boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], im_shape[0] - 1), 0)
    return boxes
