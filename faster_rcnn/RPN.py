import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from utils.timer import Timer
from utils.blob import im_list_to_blob
from fast_rcnn.nms_wrapper import nms
from rpn_msr.proposal_layer import proposal_layer as proposal_layer_py
from rpn_msr.anchor_target_layer import anchor_target_layer as anchor_target_layer_py
from fast_rcnn.bbox_transform import bbox_transform_inv, clip_boxes


import network
from network import Conv2d, FC
# from roi_pooling.modules.roi_pool_py import RoIPool
from roi_pooling.modules.roi_pool import RoIPool
from vgg16 import VGG16
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
import math

DEBUG = False


def nms_detections(pred_boxes, scores, nms_thresh, inds=None):
    dets = np.hstack((pred_boxes,
                      scores[:, np.newaxis])).astype(np.float32)
    keep = nms(dets, nms_thresh)
    if inds is None:
        return pred_boxes[keep], scores[keep]
    return pred_boxes[keep], scores[keep], inds[keep]



class RPN(nn.Module):
    _feat_stride = [16, ]
    
    anchor_scales_kmeans = [19.944, 9.118, 35.648, 42.102, 23.476, 15.882, 6.169, 9.702, 6.072, 32.254, 3.294, 10.148, 22.443, \
            13.831, 16.250, 27.969, 14.181, 27.818, 34.146, 29.812, 14.219, 22.309, 20.360, 24.025, 40.593, ]
    anchor_ratios_kmeans =  [2.631, 2.304, 0.935, 0.654, 0.173, 0.720, 0.553, 0.374, 1.565, 0.463, 0.985, 0.914, 0.734, 2.671, \
            0.209, 1.318, 1.285, 2.717, 0.369, 0.718, 0.319, 0.218, 1.319, 0.442, 1.437, ]

    anchor_scales_kmeans_region = [18.865, 27.466, 35.138, 9.383, 34.770, 31.223, 14.003, 40.663, 20.187, 6.062, 31.354, 21.213, \
            19.379, 9.843, 5.980, 3.271, 14.700, 12.794, 25.936, 24.221, 9.690, 27.328, 41.850, 16.087, 23.949,]
    anchor_ratios_kmeans_region =  [2.796, 2.810, 0.981, 0.416, 0.381, 0.422, 2.358, 1.445, 1.298, 1.690, 0.680, 0.201, 0.636, 0.979, \
            0.590, 1.006, 0.956, 0.327, 0.872, 0.455, 2.201, 1.478, 0.657, 0.224, 0.181, ]

    anchor_scales_normal = [2, 4, 8, 16, 32, 64]
    anchor_ratios_normal = [0.25, 0.5, 1, 2, 4]
    anchor_scales_normal_region = [4, 8, 16, 32, 64]
    anchor_ratios_normal_region = [0.25, 0.5, 1, 2, 4]

    def __init__(self, use_kmeans_anchors=False):
        super(RPN, self).__init__()

        if use_kmeans_anchors:
            print 'using k-means anchors'
            self.anchor_scales = self.anchor_scales_kmeans
            self.anchor_ratios = self.anchor_ratios_kmeans
            self.anchor_scales_region = self.anchor_scales_kmeans_region
            self.anchor_ratios_region = self.anchor_ratios_kmeans_region
        else:
            print 'using normal anchors'
            self.anchor_scales, self.anchor_ratios = \
                np.meshgrid(self.anchor_scales_normal, self.anchor_ratios_normal, indexing='ij')
            self.anchor_scales = self.anchor_scales.reshape(-1)
            self.anchor_ratios = self.anchor_ratios.reshape(-1)
            self.anchor_scales_region, self.anchor_ratios_region = \
                np.meshgrid(self.anchor_scales_normal_region, self.anchor_ratios_normal_region, indexing='ij')
            self.anchor_scales_region = self.anchor_scales_region.reshape(-1)
            self.anchor_ratios_region = self.anchor_ratios_region.reshape(-1)

        self.anchor_num = len(self.anchor_scales)
        self.anchor_num_region = len(self.anchor_scales_region)

        # self.features = VGG16(bn=False)
        self.features = models.vgg16(pretrained=True).features
        self.features.__delattr__('30') # to delete the max pooling
        # by default, fix the first four layers
        network.set_trainable_param(list(self.features.parameters())[:8], requires_grad=False) 

        # self.features = models.vgg16().features
        self.conv1 = Conv2d(512, 512, 3, same_padding=True)
        self.score_conv = Conv2d(512, self.anchor_num * 2, 1, relu=False, same_padding=False)
        self.bbox_conv = Conv2d(512, self.anchor_num * 4, 1, relu=False, same_padding=False)

        self.conv1_region = Conv2d(512, 512, 3, same_padding=True)
        self.score_conv_region = Conv2d(512, self.anchor_num_region * 2, 1, relu=False, same_padding=False)
        self.bbox_conv_region = Conv2d(512, self.anchor_num_region * 4, 1, relu=False, same_padding=False)

        # loss
        self.cross_entropy = None
        self.loss_box = None
        self.cross_entropy_region = None
        self.loss_box_region = None

        # initialize the parameters
        self.initialize_parameters()

    def initialize_parameters(self, normal_method='normal'):


        if normal_method == 'normal':
            normal_fun = network.weights_normal_init
        elif normal_method == 'MSRA':
            normal_fun = network.weights_MSRA_init
        else:
            raise(Exception('Cannot recognize the normal method:'.format(normal_method)))

        normal_fun(self.conv1, 0.025)
        normal_fun(self.score_conv, 0.025)
        normal_fun(self.bbox_conv, 0.01)
        normal_fun(self.conv1_region, 0.025)
        normal_fun(self.score_conv_region, 0.025)
        normal_fun(self.bbox_conv_region, 0.01)

    @property
    def loss(self):
        return self.cross_entropy + self.loss_box * 0.5 + self.cross_entropy_region + 1. * self.loss_box_region

    def forward(self, im_data, im_info, gt_objects=None, gt_regions=None, dontcare_areas=None):


        im_data = Variable(im_data.cuda())

        features = self.features(im_data)
        # print 'features.std()', features.data.std()
        rpn_conv1 = self.conv1(features)
        # print 'rpn_conv1.std()', rpn_conv1.data.std()
        # object proposal score
        rpn_cls_score = self.score_conv(rpn_conv1)
        # print 'rpn_cls_score.std()', rpn_cls_score.data.std()
        rpn_cls_score_reshape = self.reshape_layer(rpn_cls_score, 2)
        rpn_cls_prob = F.softmax(rpn_cls_score_reshape)
        rpn_cls_prob_reshape = self.reshape_layer(rpn_cls_prob, self.anchor_num*2)
        # rpn boxes
        rpn_bbox_pred = self.bbox_conv(rpn_conv1)
        # print 'rpn_bbox_pred.std()', rpn_bbox_pred.data.std() * 4

        rpn_conv1_region = self.conv1_region(features)
        # print 'rpn_conv1_region.std()', rpn_conv1_region.data.std()
        # object proposal score
        rpn_cls_score_region = self.score_conv(rpn_conv1_region)
        # print 'rpn_cls_score_region.std()', rpn_cls_score_region.data.std()
        rpn_cls_score_region_reshape = self.reshape_layer(rpn_cls_score_region, 2)
        rpn_cls_prob_region = F.softmax(rpn_cls_score_region_reshape)
        rpn_cls_prob_region_reshape = self.reshape_layer(rpn_cls_prob_region, self.anchor_num*2)
        # rpn boxes
        rpn_bbox_pred_region = self.bbox_conv(rpn_conv1_region)
        # print 'rpn_bbox_pred_region.std()', rpn_bbox_pred_region.data.std() * 4

        # proposal layer
        cfg_key = 'TRAIN' if self.training else 'TEST'
        rois = self.proposal_layer(rpn_cls_prob_reshape, rpn_bbox_pred, im_info,
                                   cfg_key, self._feat_stride, self.anchor_scales, self.anchor_ratios, 
                                   is_region=False)
        region_rois = self.proposal_layer(rpn_cls_prob_region_reshape, rpn_bbox_pred_region, im_info,
                                   cfg_key, self._feat_stride, self.anchor_scales_region, self.anchor_ratios_region, 
                                   is_region=True)

        # generating training labels and build the rpn loss
        if self.training:
            rpn_data = self.anchor_target_layer(rpn_cls_score, gt_objects, dontcare_areas,
                                                im_info, self.anchor_scales, self.anchor_ratios, self._feat_stride, )
            rpn_data_region = self.anchor_target_layer(rpn_cls_score_region, gt_regions[:, :4], dontcare_areas,
                                                im_info, self.anchor_scales_region, self.anchor_ratios_region, \
                                                self._feat_stride, is_region=True)
            if DEBUG:
                print 'rpn_data', rpn_data
                print 'rpn_cls_score_reshape', rpn_cls_score_reshape

            self.cross_entropy, self.loss_box = \
                self.build_loss(rpn_cls_score_reshape, rpn_bbox_pred, rpn_data)
            self.cross_entropy_region, self.loss_box_region = \
                self.build_loss(rpn_cls_score_region_reshape, rpn_bbox_pred_region, rpn_data_region, is_region=True)

        return features, rois, region_rois

    def build_loss(self, rpn_cls_score_reshape, rpn_bbox_pred, rpn_data, is_region=False):
        # classification loss
        rpn_cls_score = rpn_cls_score_reshape.permute(0, 2, 3, 1).contiguous().view(-1, 2)
        rpn_label = rpn_data[0]

        # print rpn_label.size(), rpn_cls_score.size()

        rpn_keep = Variable(rpn_label.data.ne(-1).nonzero().squeeze()).cuda()
        rpn_cls_score = torch.index_select(rpn_cls_score, 0, rpn_keep)
        rpn_label = torch.index_select(rpn_label, 0, rpn_keep)

        fg_cnt = torch.sum(rpn_label.data.ne(0))
        bg_cnt = rpn_label.data.numel() - fg_cnt
        # ce_weights = torch.ones(rpn_cls_score.size()[1])
        # ce_weights[0] = float(fg_cnt) / bg_cnt
        # ce_weights = ce_weights.cuda()

        _, predict = torch.max(rpn_cls_score.data, 1)
        error = torch.sum(torch.abs(predict - rpn_label.data))
        #  try:
        if predict.size()[0] < 256:
            print predict.size()
            print rpn_label.size()
            print fg_cnt

        if is_region:
            self.tp_region = torch.sum(predict[:fg_cnt].eq(rpn_label.data[:fg_cnt]))
            self.tf_region = torch.sum(predict[fg_cnt:].eq(rpn_label.data[fg_cnt:]))
            self.fg_cnt_region = fg_cnt
            self.bg_cnt_region = bg_cnt
            if DEBUG:
                print 'accuracy: %2.2f%%' % ((self.tp + self.tf) / float(fg_cnt + bg_cnt) * 100)
        else:
            self.tp = torch.sum(predict[:fg_cnt].eq(rpn_label.data[:fg_cnt]))
            self.tf = torch.sum(predict[fg_cnt:].eq(rpn_label.data[fg_cnt:]))
            self.fg_cnt = fg_cnt
            self.bg_cnt = bg_cnt
            if DEBUG:
                print 'accuracy: %2.2f%%' % ((self.tp + self.tf) / float(fg_cnt + bg_cnt) * 100)

        rpn_cross_entropy = F.cross_entropy(rpn_cls_score, rpn_label)
        # print rpn_cross_entropy

        # box loss
        rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = rpn_data[1:]
        rpn_bbox_targets = torch.mul(rpn_bbox_targets, rpn_bbox_inside_weights)
        rpn_bbox_pred = torch.mul(rpn_bbox_pred, rpn_bbox_inside_weights)


        # print 'Smooth L1 loss: ', F.smooth_l1_loss(rpn_bbox_pred, rpn_bbox_targets, size_average=False)
        # print 'fg_cnt', fg_cnt
        rpn_loss_box = F.smooth_l1_loss(rpn_bbox_pred, rpn_bbox_targets, size_average=False) /  (fg_cnt + 1e-4)
        # print 'rpn_loss_box', rpn_loss_box
        # print rpn_loss_box

        return rpn_cross_entropy, rpn_loss_box

    @staticmethod
    def reshape_layer(x, d):
        input_shape = x.size()
        # x = x.permute(0, 3, 1, 2)
        # b c w h
        x = x.view(
            input_shape[0],
            int(d),
            int(float(input_shape[1] * input_shape[2]) / float(d)),
            input_shape[3]
        )
        # x = x.permute(0, 2, 3, 1)
        return x

    @staticmethod
    def proposal_layer(rpn_cls_prob_reshape, rpn_bbox_pred, im_info, cfg_key, 
                    _feat_stride, anchor_scales, anchor_ratios, is_region):
        rpn_cls_prob_reshape = rpn_cls_prob_reshape.data.cpu().numpy()
        rpn_bbox_pred = rpn_bbox_pred.data.cpu().numpy()
        x = proposal_layer_py(rpn_cls_prob_reshape, rpn_bbox_pred, im_info, 
                    cfg_key, _feat_stride, anchor_scales, anchor_ratios, is_region=is_region)
        x = network.np_to_variable(x, is_cuda=True)
        return x.view(-1, 5)

    @staticmethod
    def anchor_target_layer(rpn_cls_score, gt_boxes, dontcare_areas, im_info, _feat_stride, anchor_scales, anchor_rotios, is_region=False):
        """
        rpn_cls_score: for pytorch (1, Ax2, H, W) bg/fg scores of previous conv layer
        gt_boxes: (G, 5) vstack of [x1, y1, x2, y2, class]
        #gt_ishard: (G, 1), 1 or 0 indicates difficult or not
        dontcare_areas: (D, 4), some areas may contains small objs but no labelling. D may be 0
        im_info: a list of [image_height, image_width, scale_ratios]
        _feat_stride: the downsampling ratio of feature map to the original input image
        anchor_scales: the scales to the basic_anchor (basic anchor is [16, 16])
        ----------
        Returns
        ----------
        rpn_labels : (1, 1, HxA, W), for each anchor, 0 denotes bg, 1 fg, -1 dontcare
        rpn_bbox_targets: (1, 4xA, H, W), distances of the anchors to the gt_boxes(may contains some transform)
                        that are the regression objectives
        rpn_bbox_inside_weights: (1, 4xA, H, W) weights of each boxes, mainly accepts hyper param in cfg
        rpn_bbox_outside_weights: (1, 4xA, H, W) used to balance the fg/bg,
        beacuse the numbers of bgs and fgs mays significiantly different
        """
        rpn_cls_score = rpn_cls_score.data.cpu().numpy()
        rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = \
            anchor_target_layer_py(rpn_cls_score, gt_boxes, dontcare_areas, im_info, _feat_stride, anchor_scales, anchor_rotios, is_region=is_region)

        rpn_labels = network.np_to_variable(rpn_labels, is_cuda=True, dtype=torch.LongTensor)
        rpn_bbox_targets = network.np_to_variable(rpn_bbox_targets, is_cuda=True)
        rpn_bbox_inside_weights = network.np_to_variable(rpn_bbox_inside_weights, is_cuda=True)
        rpn_bbox_outside_weights = network.np_to_variable(rpn_bbox_outside_weights, is_cuda=True)

        return rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights

    def load_from_npz(self, params):
        # params = np.load(npz_file)
        self.features.load_from_npz(params)

        pairs = {'conv1.conv': 'rpn_conv/3x3', 'score_conv.conv': 'rpn_cls_score', 'bbox_conv.conv': 'rpn_bbox_pred'}
        own_dict = self.state_dict()
        for k, v in pairs.items():
            key = '{}.weight'.format(k)
            param = torch.from_numpy(params['{}/weights:0'.format(v)]).permute(3, 2, 0, 1)
            own_dict[key].copy_(param)

            key = '{}.bias'.format(k)
            param = torch.from_numpy(params['{}/biases:0'.format(v)])
            own_dict[key].copy_(param)
