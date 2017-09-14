import cv2
import numpy as np
import numpy.random as npr
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.model_zoo as model_zoo
import torchvision.models as models
import os.path as osp

from utils.timer import Timer
from utils.blob import im_list_to_blob
from fast_rcnn.nms_wrapper import nms
from fast_rcnn.bbox_transform import bbox_transform_inv_hdn, clip_boxes
from fast_rcnn.config import cfg
from utils.cython_bbox import bbox_overlaps

import network
from network import Conv2d, FC
# from roi_pooling.modules.roi_pool_py import RoIPool
from roi_pooling.modules.roi_pool import RoIPool
from vgg16 import VGG16

import pdb

DEBUG = False
TIME_IT = cfg.TIME_IT


def nms_detections(pred_boxes, scores, nms_thresh, inds=None):
    dets = np.hstack((pred_boxes,
                      scores[:, np.newaxis])).astype(np.float32)
    keep = nms(dets, nms_thresh)
    if inds is None:
        return pred_boxes[keep], scores[keep], keep
    return pred_boxes[keep], scores[keep], inds[keep], keep

class HDN_base(nn.Module):
    
    PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]])
    SCALES = (600,)
    MAX_SIZE = 1000
    MPS_iter_range = range(1, cfg.TRAIN.MAX_MPS_ITER_NUM + 1)

    def __init__(self, nhidden, n_object_cats, n_predicate_cats, n_vocab, voc_sign, 
                 max_word_length, MPS_iter, use_language_loss, object_loss_weight, 
                 predicate_loss_weight, dropout, 
                 use_kmeans_anchors, 
                 nhidden_caption, 
                 nembedding,
                 rnn_type, use_region_reg=False):

        super(HDN_base, self).__init__()
        assert n_object_cats is not None and n_predicate_cats is not None
        if rnn_type == 'LSTM_normal':
            nembedding = nhidden
        if MPS_iter < 0:
            print 'Use random interation from 1 to 5'

        self.n_classes_obj = n_object_cats
        self.n_classes_pred = n_predicate_cats
        self.max_word_length = max_word_length
        self.MPS_iter = MPS_iter
        self.use_language_loss = use_language_loss
        self.object_loss_weight = object_loss_weight
        self.predicate_loss_weight = predicate_loss_weight
        self.dropout = dropout
        self.nhidden = nhidden
        self.nhidden_caption = nhidden_caption
        self.nembedding = nembedding
        self.rnn_type = rnn_type
        self.voc_sign = voc_sign
        self.n_vocab = n_vocab
        self.use_region_reg = use_region_reg

        # loss
        self.cross_entropy_object = None
        self.cross_entropy_predicate = None
        self.region_caption_loss = None
        self.loss_obj_box = None
        self.loss_region_box = Variable(torch.zeros(1)).cuda()

        self.timer = Timer()

    def reinitialize_fc_layers(self):

        print 'Reinitialize the fc layers...',
        weight_multiplier = 4096. / self.nhidden
        vgg16 = models.vgg16(pretrained=True)
        self.fc6_obj.fc.weight.data.copy_(vgg16.classifier[0].weight.data[:self.nhidden] * weight_multiplier)
        self.fc6_obj.fc.bias.data.copy_(vgg16.classifier[0].bias.data[:self.nhidden] * weight_multiplier)
        self.fc6_phrase.fc.weight.data.copy_(vgg16.classifier[0].weight.data[:self.nhidden] * weight_multiplier)
        self.fc6_phrase.fc.bias.data.copy_(vgg16.classifier[0].bias.data[:self.nhidden] * weight_multiplier)
        self.fc6_region.fc.weight.data.copy_(vgg16.classifier[0].weight.data[:self.nhidden] * weight_multiplier)
        self.fc6_region.fc.bias.data.copy_(vgg16.classifier[0].bias.data[:self.nhidden] * weight_multiplier)

        self.fc7_obj.fc.weight.data.copy_(vgg16.classifier[3].weight.data[:self.nhidden, :self.nhidden] * weight_multiplier)
        self.fc7_obj.fc.bias.data.copy_(vgg16.classifier[3].bias.data[:self.nhidden])
        self.fc7_phrase.fc.weight.data.copy_(vgg16.classifier[3].weight.data[:self.nhidden, :self.nhidden] * weight_multiplier)
        self.fc7_phrase.fc.bias.data.copy_(vgg16.classifier[3].bias.data[:self.nhidden])
        self.fc7_region.fc.weight.data.copy_(vgg16.classifier[3].weight.data[:self.nhidden, :self.nhidden] * weight_multiplier)
        self.fc7_region.fc.bias.data.copy_(vgg16.classifier[3].bias.data[:self.nhidden])
        # network.weights_normal_init(self.caption_prediction, 0.01)
        print 'Done.'


    @property
    def loss(self):
        return self.cross_entropy_object + self.loss_obj_box + \
               self.cross_entropy_predicate * 1 + self.region_caption_loss + self.loss_region_box
    


    def build_loss_object(self, cls_score, bbox_pred, roi_data):
        # classification loss
        label = roi_data[1].squeeze()
        fg_cnt = torch.sum(label.data.ne(0))
        bg_cnt = label.data.numel() - fg_cnt

        ce_weights = np.sqrt(self.object_loss_weight)
        ce_weights[0] = float(fg_cnt) / (bg_cnt + 1e-5)
        ce_weights = ce_weights.cuda()

        maxv, predict = cls_score.data.max(1)
        if fg_cnt > 0:
            self.tp = torch.sum(predict[:fg_cnt].eq(label.data[:fg_cnt]))
        else:
            self.tp = 0.
        if bg_cnt > 0:
            self.tf = torch.sum(predict[fg_cnt:].eq(label.data[fg_cnt:]))
        else:
            self.tp = 0.
        self.fg_cnt = fg_cnt
        self.bg_cnt = bg_cnt

        # print '[object]:'
        # if predict.sum() > 0:
        # print predict

        # print 'accuracy: %2.2f%%' % (((self.tp + self.tf) / float(fg_cnt + bg_cnt)) * 100)
        # print predict
        cross_entropy = F.cross_entropy(cls_score, label, weight=ce_weights)
        # print cross_entropy

        # bounding box regression L1 loss
        bbox_targets, bbox_inside_weights, bbox_outside_weights = roi_data[2:]

        # b = bbox_targets.data.cpu().numpy()

        bbox_targets = torch.mul(bbox_targets, bbox_inside_weights)
        bbox_pred = torch.mul(bbox_pred, bbox_inside_weights)

        # a = bbox_pred.data.cpu().numpy()
        loss_box = F.smooth_l1_loss(bbox_pred, bbox_targets, size_average=False) / (fg_cnt + 1e-5)
        # print loss_box

        return cross_entropy, loss_box


    def build_loss_bbox(self, bbox_pred, roi_data):
        bbox_targets, bbox_inside_weights, bbox_outside_weights = roi_data[2:]
        bbox_targets = torch.mul(bbox_targets, bbox_inside_weights)
        bbox_pred = torch.mul(bbox_pred, bbox_inside_weights)
        fg_cnt = torch.sum(bbox_inside_weights[:, 0].data.ne(0)) 
        loss_box = F.smooth_l1_loss(bbox_pred, bbox_targets, size_average=False) / (fg_cnt + 1e-5)
        return loss_box


    def build_loss_cls(self, cls_score, labels):
        labels = labels.squeeze()
        fg_cnt = torch.sum(labels.data.ne(0))
        bg_cnt = labels.data.numel() - fg_cnt

        ce_weights = np.sqrt(self.predicate_loss_weight)
        ce_weights[0] = float(fg_cnt) / (bg_cnt + 1e-5)
        ce_weights = ce_weights.cuda()
        # print '[relationship]:'
        # print 'ce_weights:'
        # print ce_weights
        # print 'cls_score:'
        # print cls_score 
        # print 'labels'
        # print labels
        ce_weights = ce_weights.cuda()
        cross_entropy = F.cross_entropy(cls_score, labels, weight=ce_weights)

        maxv, predict = cls_score.data.max(1)
        # if DEBUG:
        # print '[predicate]:'
        # if predict.sum() > 0:
        # print predict
        # print 'labels'
        # print labels

        if fg_cnt == 0:
            tp = 0
        else:
            tp = torch.sum(predict[bg_cnt:].eq(labels.data[bg_cnt:]))
        tf = torch.sum(predict[:bg_cnt].eq(labels.data[:bg_cnt]))
        fg_cnt = fg_cnt
        bg_cnt = bg_cnt

        return cross_entropy, tp, tf, fg_cnt, bg_cnt


    def get_image_blob_noscale(self, im):
        im_orig = im.astype(np.float32, copy=True)
        im_orig -= self.PIXEL_MEANS
        im_shape = im_orig.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])

        processed_ims = []
        im_scale_factors = []

        target_size = self.SCALES[0]
        im_scale = float(target_size) / float(im_size_min)
        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(im_scale * im_size_max) > self.MAX_SIZE:
            im_scale = float(self.MAX_SIZE) / float(im_size_max)
        im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                        interpolation=cv2.INTER_LINEAR)
        im_scale_factors.append(im_scale)
        processed_ims.append(im)

        blob = im_list_to_blob(processed_ims)

        return blob, np.array(im_scale_factors)

    def get_image_blob(self, im):
        """Converts an image into a network input.
        Arguments:
            im (ndarray): a color image in BGR order
        Returns:
            blob (ndarray): a data blob holding an image pyramid
            im_scale_factors (list): list of image scales (relative to im) used
                in the image pyramid
        """
        im_orig = im.astype(np.float32, copy=True)
        im_orig -= self.PIXEL_MEANS

        im_shape = im_orig.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])

        processed_ims = []
        im_scale_factors = []

        for target_size in self.SCALES:
            im_scale = float(target_size) / float(im_size_min)
            # Prevent the biggest axis from being more than MAX_SIZE
            if np.round(im_scale * im_size_max) > self.MAX_SIZE:
                im_scale = float(self.MAX_SIZE) / float(im_size_max)
            im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                            interpolation=cv2.INTER_LINEAR)
            im_scale_factors.append(im_scale)
            processed_ims.append(im)

        # Create a blob to hold the input images
        blob = im_list_to_blob(processed_ims)

        return blob, np.array(im_scale_factors)


    def get_gt_objects(self, imdb):
        gt_boxes_object = np.empty((len(imdb['objects']), 5), dtype=np.float32)
        gt_boxes_object[:, 0:4] = np.array([obj['box'] for obj in imdb['objects']])
        gt_boxes_object[:, 4] = np.array([obj['class'] for obj in imdb['objects']])
        return gt_boxes_object

    def get_gt_regions(self, imdb):
        gt_boxes_region= np.empty((len(imdb['regions']), 4), dtype=np.float32)
        gt_boxes_region = np.array([reg['box'] for reg in imdb['regions']])
        return gt_boxes_region
    

    def load_from_npz(self, params):
        self.rpn.load_from_npz(params)

        pairs = {'fc6.fc': 'fc6', 'fc7.fc': 'fc7', 'score_fc.fc': 'cls_score', 'bbox_fc.fc': 'bbox_pred'}
        own_dict = self.state_dict()
        for k, v in pairs.items():
            key = '{}.weight'.format(k)
            param = torch.from_numpy(params['{}/weights:0'.format(v)]).permute(1, 0)
            own_dict[key].copy_(param)

            key = '{}.bias'.format(k)
            param = torch.from_numpy(params['{}/biases:0'.format(v)])
            own_dict[key].copy_(param)




    def object_detection(self, image_path, gt_boxes=None):
        min_score = 1/150.
        image = cv2.imread(image_path)
        # print 'image.shape', image.shape
        im_data, im_scales = self.get_image_blob_noscale(image)
        if gt_boxes is not None:
            gt_boxes[:, :4] = gt_boxes[:, :4] * im_scales[0]
        # print 'im_data.shape', im_data.shape
        # print 'im_scales', im_scales
        im_info = np.array(
            [[im_data.shape[1], im_data.shape[2], im_scales[0]]],
            dtype=np.float32)
        object_result = self(im_data, im_info)[0]
        cls_prob_object, bbox_object, object_rois = object_result[:]

        prob_object = F.softmax(cls_prob_object)
        prob = prob_object.cpu().data.numpy()
        boxes = object_rois.data.cpu().numpy()[:, 1:5] / im_info[0][2]
        fg_id = np.where(prob > min_score)
        box_id = fg_id[0]
        cls_id = fg_id[1]
        box_id = box_id[cls_id > 0]
        cls_id = cls_id[cls_id > 0]
        box_deltas = bbox_object.data.cpu().numpy()
        new_box_delta = np.asarray([
            box_deltas[box_id[i], (cls_id[i] * 4): (cls_id[i] * 4 + 4)] for i in range(len(cls_id))
        ], dtype=np.float)
        regressed_boxes = bbox_transform_inv_hdn(boxes[box_id], new_box_delta)
        regressed_boxes = clip_boxes(regressed_boxes, image.shape)


        object_score = np.asarray([
            prob[box_id[i], cls_id[i]] for i in range(len(cls_id))
        ], dtype=np.float)

        # print 'im_scales[0]', im_scales[0]
        return (cls_id, object_score, regressed_boxes)


    def object_detection_gt_boxes(self, image_path, gt_boxes):
        min_score = 1/150.
        image = cv2.imread(image_path)
        # print 'image.shape', image.shape
        im_data, im_scales = self.get_image_blob_noscale(image)
        gt_boxes[:, :4] = gt_boxes[:, :4] * im_scales[0]
        # print 'im_data.shape', im_data.shape
        # print 'im_scales', im_scales
        im_info = np.array(
            [[im_data.shape[1], im_data.shape[2], im_scales[0]]],
            dtype=np.float32)
        object_result = self(im_data, im_info, gt_boxes)[0]
        cls_prob_object, bbox_object, object_rois = object_result[:]

        prob_object = F.softmax(cls_prob_object)
        prob = prob_object.cpu().data
        top_5_cls = torch.topk(prob[:, 1:], 5, dim=1)
        # print 'im_scales[0]', im_scales[0]
        return top_5_cls[1].numpy()