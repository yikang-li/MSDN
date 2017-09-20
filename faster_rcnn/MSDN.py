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
from utils.HDN_utils import check_relationship_recall
from fast_rcnn.nms_wrapper import nms
from rpn_msr.proposal_layer import proposal_layer as proposal_layer_py
from rpn_msr.anchor_target_layer import anchor_target_layer as anchor_target_layer_py
from rpn_msr.proposal_target_layer_hdn import proposal_target_layer as proposal_target_layer_py
from fast_rcnn.bbox_transform import bbox_transform_inv_hdn, clip_boxes
from fast_rcnn.hierarchical_message_passing_structure import Hierarchical_Message_Passing_Structure
from Language_Model import Language_Model
from RPN import RPN
from fast_rcnn.config import cfg
from utils.cython_bbox import bbox_overlaps

import network
from network import Conv2d, FC
# from roi_pooling.modules.roi_pool_py import RoIPool
from roi_pooling.modules.roi_pool import RoIPool
from vgg16 import VGG16
from MSDN_base import HDN_base
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

class Hierarchical_Descriptive_Model(HDN_base):
    def __init__(self,nhidden, n_object_cats, n_predicate_cats, n_vocab, voc_sign, 
                 max_word_length, MPS_iter, use_language_loss, object_loss_weight, 
                 predicate_loss_weight, 
                 dropout=False, 
                 use_kmeans_anchors=False, 
                 gate_width=128, 
                 nhidden_caption=256, 
                 nembedding = 256,
                 rnn_type='LSTM_normal', 
                 rnn_droptout=0.0, rnn_bias=False, 
                 use_region_reg=False, use_kernel=False):
    
        super(Hierarchical_Descriptive_Model, self).__init__(nhidden, n_object_cats, n_predicate_cats, n_vocab, voc_sign, 
                 max_word_length, MPS_iter, use_language_loss, object_loss_weight, predicate_loss_weight, 
                 dropout, use_kmeans_anchors, nhidden_caption, nembedding, rnn_type, use_region_reg)

        self.rpn = RPN(use_kmeans_anchors)
        self.roi_pool_object = RoIPool(7, 7, 1.0/16)
        self.roi_pool_phrase = RoIPool(7, 7, 1.0/16)
        self.roi_pool_region = RoIPool(7, 7, 1.0/16)
        self.fc6_obj = FC(512 * 7 * 7, nhidden, relu=True)
        self.fc7_obj = FC(nhidden, nhidden, relu=False)
        self.fc6_phrase = FC(512 * 7 * 7, nhidden, relu=True)
        self.fc7_phrase = FC(nhidden, nhidden, relu=False)
        self.fc6_region = FC(512 * 7 * 7, nhidden, relu=True)
        self.fc7_region = FC(nhidden, nhidden, relu=False)
        if MPS_iter == 0:
            self.mps = None
        else:
            self.mps = Hierarchical_Message_Passing_Structure(nhidden, dropout, 
                            gate_width=gate_width, use_kernel_function=use_kernel) # the hierarchical message passing structure
            network.weights_normal_init(self.mps, 0.01)   

        self.score_obj = FC(nhidden, self.n_classes_obj, relu=False)
        self.bbox_obj = FC(nhidden, self.n_classes_obj * 4, relu=False)
        self.score_pred = FC(nhidden, self.n_classes_pred, relu=False)
        if self.use_region_reg:
            self.bbox_region = FC(nhidden, 4, relu=False)
            network.weights_normal_init(self.bbox_region, 0.01)
        else:
            self.bbox_region = None

        self.objectiveness = FC(nhidden, 2, relu=False)

        if use_language_loss:
            self.caption_prediction = \
                Language_Model(rnn_type=self.rnn_type, ntoken=self.n_vocab, nimg=self.nhidden, nhidden=self.nhidden_caption, 
                                nembed=self.nembedding, nlayers=2, nseq=self.max_word_length, voc_sign = self.voc_sign, 
                                bias=rnn_bias, dropout=rnn_droptout) 
        else:
            self.caption_prediction = Language_Model(rnn_type=self.rnn_type, ntoken=self.n_vocab, nimg=1, nhidden=1, 
                                nembed=1, nlayers=1, nseq=1, voc_sign = self.voc_sign) # just to make the program run

        network.weights_normal_init(self.score_obj, 0.01)
        network.weights_normal_init(self.bbox_obj, 0.005)
        network.weights_normal_init(self.score_pred, 0.01)
        network.weights_normal_init(self.objectiveness, 0.01)

        self.objectiveness_loss = None



    def forward(self, im_data, im_info, gt_objects=None, gt_relationships=None, gt_regions=None, 
                    use_beam_search=False, graph_generation=False):

        self.timer.tic()
        features, object_rois, region_rois = self.rpn(im_data, im_info, gt_objects, gt_regions)

        if not self.training and gt_objects is not None:
            zeros = np.zeros((gt_objects.shape[0], 1), dtype=gt_objects.dtype)
            object_rois_gt = np.hstack((zeros, gt_objects[:, :4]))
            object_rois_gt = network.np_to_variable(object_rois_gt, is_cuda=True)
            object_rois[:object_rois_gt.size(0)] = object_rois_gt


        if not self.training and gt_regions is not None:
            zeros = np.zeros((gt_regions.shape[0], 1), dtype=gt_regions.dtype)
            region_rois = np.hstack((zeros, gt_regions[:, :4]))
            region_rois = network.np_to_variable(region_rois, is_cuda=True)
            # print 'region_rois[gt]:', region_rois


        # print 'object_rois.shape', object_rois.size()

        # print 'features.std'
        # print features.data.std()
        if TIME_IT:
            torch.cuda.synchronize()
            print '\t[RPN]:      %.3fs' % self.timer.toc(average=False)


        self.timer.tic()
        roi_data_object, roi_data_predicate, roi_data_region, mat_object, mat_phrase, mat_region = \
            self.proposal_target_layer(object_rois, region_rois, gt_objects, gt_relationships, gt_regions, 
                    self.n_classes_obj, self.voc_sign, self.training, graph_generation=graph_generation)
        if TIME_IT:
            torch.cuda.synchronize()
            print '\t[Proposal]: %.3fs' % self.timer.toc(average=False)


        self.timer.tic()
        object_rois = roi_data_object[0]
        phrase_rois = roi_data_predicate[0]
        region_rois = roi_data_region[0]

        # print 'object_rois_num: {}'.format(object_rois.size()[0])
        # print 'phrase_rois_num: {}'.format(phrase_rois.size()[0])
        # print 'region_rois_num: {}'.format(region_rois.size()[0])

        # roi pool
        pooled_object_features = self.roi_pool_object(features, object_rois)
        if TIME_IT:
            torch.cuda.synchronize()
            print '\t\t[object_pooling]: %.3fs' % self.timer.toc(average=False)
        #print 'pool5_object.std'
        #print pooled_object_features.data.std()
        pooled_object_features = pooled_object_features.view(pooled_object_features.size()[0], -1)
        if TIME_IT:
            torch.cuda.synchronize()
            print '\t\t[object_feature_view]: %.3fs' % self.timer.toc(average=False)
        pooled_object_features = self.fc6_obj(pooled_object_features)
        if TIME_IT:
            torch.cuda.synchronize()
            print '\t\t[object_feature_fc6]: %.3fs' % self.timer.toc(average=False)
        if self.dropout:
            pooled_object_features = F.dropout(pooled_object_features, training = self.training)
        #print 'fc6_object.std'
        #print pooled_object_features.data.std()
        pooled_object_features = self.fc7_obj(pooled_object_features)
        if TIME_IT:
            torch.cuda.synchronize()
            print '\t\t[object_feature_fc7]: %.3fs' % self.timer.toc(average=False)
        if self.dropout:
            pooled_object_features = F.dropout(pooled_object_features, training = self.training)
        #print 'fc7_object.std'
        #print pooled_object_features.data.std()

        pooled_phrase_features = self.roi_pool_phrase(features, phrase_rois)
        if TIME_IT:
            torch.cuda.synchronize()
            print '\t\t[phrase_pooling]: %.3fs' % self.timer.toc(average=False)
        #print 'pool5_phrase.std'
        #print pooled_phrase_features.data.std()
        pooled_phrase_features = pooled_phrase_features.view(pooled_phrase_features.size()[0], -1)
        if TIME_IT:
            torch.cuda.synchronize()
            print '\t\t[phrase_feature_view]: %.3fs' % self.timer.toc(average=False)
        pooled_phrase_features = self.fc6_phrase(pooled_phrase_features)
        if TIME_IT:
            torch.cuda.synchronize()
            print '\t\t[phrase_feature_fc6]: %.3fs' % self.timer.toc(average=False)
        if self.dropout:
            pooled_phrase_features = F.dropout(pooled_phrase_features, training = self.training)
        #print 'fc6_phrase.std'
        #print pooled_phrase_features.data.std()
        pooled_phrase_features = self.fc7_phrase(pooled_phrase_features)
        if TIME_IT:
            torch.cuda.synchronize()
            print '\t\t[phrase_feature_fc7]: %.3fs' % self.timer.toc(average=False)
        if self.dropout:
            pooled_phrase_features = F.dropout(pooled_phrase_features, training = self.training)
        #print 'fc7_phrase.std'
        #print pooled_phrase_features.data.std()

        pooled_region_features = self.roi_pool_region(features, region_rois)
        if TIME_IT:
            torch.cuda.synchronize()
            print '\t\t[region_pooling]: %.3fs' % self.timer.toc(average=False)
        #print 'pool5_region.std'
        #print pooled_region_features.data.std()
        pooled_region_features = pooled_region_features.view(pooled_region_features.size()[0], -1)
        if TIME_IT:
            torch.cuda.synchronize()
            print '\t\t[region_feature_view]: %.3fs' % self.timer.toc(average=False)
        pooled_region_features = self.fc6_region(pooled_region_features)
        if TIME_IT:
            torch.cuda.synchronize()
            print '\t\t[region_feature_fc6]: %.3fs' % self.timer.toc(average=False)
        if self.dropout:
            pooled_region_features = F.dropout(pooled_region_features, training = self.training)
        #print 'fc6_region.std'
        #print pooled_region_features.data.std()
        pooled_region_features = self.fc7_region(pooled_region_features)
        if TIME_IT:
            torch.cuda.synchronize()
            print '\t\t[region_feature_fc7]: %.3fs' % self.timer.toc(average=False)
        if self.dropout:
            pooled_region_features = F.dropout(pooled_region_features, training = self.training)
        #print 'fc7_region.std'
        #print pooled_region_features.data.std()

        # print 'pre_mps_object.std', pooled_object_features.data.std()
        # print 'pre_mps_phrase.std', pooled_phrase_features.data.std()
        # print 'pre_mps_region.std', pooled_region_features.data.std()

        # bounding box regression before message passing
        bbox_object = self.bbox_obj(F.relu(pooled_object_features))

        if self.use_region_reg:
            bbox_region = self.bbox_region(F.relu(pooled_region_features))

        if TIME_IT:
            torch.cuda.synchronize()
            print '\t[Pre-MPS]:  %.3fs' % self.timer.toc(average=False)

        self.timer.tic()
        # hierarchical message passing structure
        if self.MPS_iter < 0:
            if self.training:
                self.MPS_iter = npr.choice(self.MPS_iter_range)
            else:
                self.MPS_iter = cfg.TEST.MPS_ITER_NUM

        for i in xrange(self.MPS_iter):
            pooled_object_features, pooled_phrase_features, pooled_region_features = \
                self.mps(pooled_object_features, pooled_phrase_features, pooled_region_features, \
                            mat_object, mat_phrase, mat_region)
        if TIME_IT:
            torch.cuda.synchronize()
            print '\t[Passing]:  %.3fs' % self.timer.toc(average=False)

            
        # print 'post_mps_object.std', pooled_object_features.data.std()
        # print 'post_mps_phrase.std', pooled_phrase_features.data.std()
        # print 'post_mps_region.std', pooled_region_features.data.std()

        self.timer.tic()

        pooled_object_features = F.relu(pooled_object_features)
        pooled_phrase_features = F.relu(pooled_phrase_features)
        pooled_region_features = F.relu(pooled_region_features)

        cls_score_object = self.score_obj(pooled_object_features)
        cls_prob_object = F.softmax(cls_score_object)

        cls_score_predicate = self.score_pred(pooled_phrase_features)
        cls_prob_predicate = F.softmax(cls_score_predicate)

        if not self.use_region_reg:
            bbox_region = Variable(torch.zeros(pooled_region_features.size(0), 4).cuda())


        cls_objectiveness_region = self.objectiveness(pooled_region_features)
        
        # print 'cls_score_object.std', cls_score_object.data.std()
        # print 'cls_pred_box.std', bbox_object.data.std()
        # print 'cls_score_phrase.std', cls_score_predicate.data.std()
        
        if TIME_IT:
            torch.cuda.synchronize()
            print '\t[Post-MPS]: %.3fs' % self.timer.toc(average=False)

        # if DEBUG:
        #     print 'cls_score_predicate'
        #     print cls_score_predicate
        #     print 'roi_data_predicate[1]'
        #     print roi_data_predicate[1]
        if self.training:

            self.cross_entropy_object, self.loss_obj_box = self.build_loss_object(cls_score_object, bbox_object, roi_data_object)
            self.cross_entropy_predicate, self.tp_pred, self.tf_pred, self.fg_cnt_pred, self.bg_cnt_pred = \
                    self.build_loss_cls(cls_score_predicate, roi_data_predicate[1])
            # print 'accuracy: %2.2f%%' % (((self.tp_pred + self.tf_pred) / float(self.fg_cnt_pred + self.bg_cnt_pred)) * 100)
            # self.timer.tic()
            if self.use_language_loss:
                self.region_caption_loss = self.caption_prediction(pooled_region_features, roi_data_region[1])
            else:
                self.region_caption_loss = Variable(torch.zeros(1).cuda())

            if self.use_region_reg:
                self.loss_region_box = self.build_loss_bbox(bbox_region, roi_data_region)
            # print '\t[Caption]:   %.3fs' % self.timer.toc(average=False)
            region_caption = None
            self.objectiveness_loss = self.build_loss_objectiveness(cls_objectiveness_region, \
                                        roi_data_region[3][:, 0].ne(0).type(torch.cuda.LongTensor))
        else:
            # assert False, 'Have not implemented!\n'
            if self.use_language_loss:
                # region_caption, caption_logprobs = self.caption_prediction.beamsearch(pooled_region_features, 10)
                if use_beam_search:
                    search_func = self.caption_prediction.beamsearch
                else:
                    search_func = self.caption_prediction.baseline_search
                region_caption = search_func(pooled_region_features, 5)
                # pdb.set_trace()
            else:
                region_caption = None
                caption_logprobs = None 

        caption_logprobs = F.log_softmax(cls_objectiveness_region)[:, 1].squeeze().cpu().data

        return (cls_prob_object, bbox_object, object_rois), \
                (cls_prob_predicate, mat_phrase), \
                (region_caption, bbox_region, region_rois, caption_logprobs)

    

    @staticmethod
    def proposal_target_layer(object_rois, region_rois, gt_objects, gt_relationships, 
            gt_regions, n_classes_obj, voc_sign, is_training=False, graph_generation=False):

        """
        ----------
        object_rois:  (1 x H x W x A, 5) [0, x1, y1, x2, y2]
        region_rois:  (1 x H x W x A, 5) [0, x1, y1, x2, y2]
        gt_objects:   (G_obj, 5) [x1 ,y1 ,x2, y2, obj_class] int
        gt_relationships: (G_obj, G_obj) [pred_class] int (-1 for no relationship)
        gt_regions:   (G_region, 4+40) [x1, y1, x2, y2, word_index] (-1 for padding)
        # gt_ishard: (G_region, 4+40) {0 | 1} 1 indicates hard
        # dontcare_areas: (D, 4) [ x1, y1, x2, y2]
        n_classes_obj
        n_classes_pred
        is_training to indicate whether in training scheme
        ----------
        Returns
        ----------
        rois: (1 x H x W x A, 5) [0, x1, y1, x2, y2]
        labels: (1 x H x W x A, 1) {0,1,...,_num_classes-1}
        bbox_targets: (1 x H x W x A, K x4) [dx1, dy1, dx2, dy2]
        bbox_inside_weights: (1 x H x W x A, Kx4) 0, 1 masks for the computing loss
        bbox_outside_weights: (1 x H x W x A, Kx4) 0, 1 masks for the computing loss
        """

        object_rois = object_rois.data.cpu().numpy()
        region_rois = region_rois.data.cpu().numpy()

        object_labels, object_rois, bbox_targets, bbox_inside_weights, bbox_outside_weights, mat_object, \
            phrase_label, phrase_rois, mat_phrase, region_seq, region_rois, \
            bbox_targets_region, bbox_inside_weights_region, bbox_outside_weights_region, mat_region= \
            proposal_target_layer_py(object_rois, region_rois, gt_objects, gt_relationships, 
                gt_regions, n_classes_obj, voc_sign, is_training, graph_generation=graph_generation)

        # print labels.shape, bbox_targets.shape, bbox_inside_weights.shape
        if is_training:
            object_labels = network.np_to_variable(object_labels, is_cuda=True, dtype=torch.LongTensor)
            bbox_targets = network.np_to_variable(bbox_targets, is_cuda=True)
            bbox_inside_weights = network.np_to_variable(bbox_inside_weights, is_cuda=True)
            bbox_outside_weights = network.np_to_variable(bbox_outside_weights, is_cuda=True)
            phrase_label = network.np_to_variable(phrase_label, is_cuda=True, dtype=torch.LongTensor)
            region_seq = network.np_to_variable(region_seq, is_cuda=True, dtype=torch.LongTensor)
            bbox_targets_region = network.np_to_variable(bbox_targets_region, is_cuda=True)
            bbox_inside_weights_region = network.np_to_variable(bbox_inside_weights_region, is_cuda=True)
            bbox_outside_weights_region = network.np_to_variable(bbox_outside_weights_region, is_cuda=True)

        object_rois = network.np_to_variable(object_rois, is_cuda=True)
        phrase_rois = network.np_to_variable(phrase_rois, is_cuda=True)
        region_rois = network.np_to_variable(region_rois, is_cuda=True)

        return (object_rois, object_labels, bbox_targets, bbox_inside_weights, bbox_outside_weights), \
                (phrase_rois, phrase_label), \
                (region_rois, region_seq, bbox_targets_region, bbox_inside_weights_region, bbox_outside_weights_region), \
                mat_object, mat_phrase, mat_region

    def interpret_HDN(self, cls_prob, bbox_pred, rois, cls_prob_predicate, 
                        mat_phrase, im_info, nms=True, clip=True, min_score=0.0, 
                        top_N=100, use_gt_boxes=False):
        scores, inds = cls_prob[:, 1:].data.max(1)
        inds += 1
        scores, inds = scores.cpu().numpy(), inds.cpu().numpy()
        predicate_scores, predicate_inds = cls_prob_predicate[:, 1:].data.max(1)
        predicate_inds += 1
        predicate_scores, predicate_inds = predicate_scores.cpu().numpy(), predicate_inds.cpu().numpy()
        


        keep = np.where((inds > 0) & (scores >= min_score))
        scores, inds = scores[keep], inds[keep]

        # Apply bounding-box regression deltas
        keep = keep[0]
        box_deltas = bbox_pred.data.cpu().numpy()[keep]
        box_deltas = np.asarray([
            box_deltas[i, (inds[i] * 4): (inds[i] * 4 + 4)] for i in range(len(inds))
        ], dtype=np.float)
        boxes = rois.data.cpu().numpy()[keep, 1:5] / im_info[0][2]
        if use_gt_boxes:
            nms = False
            clip = False
            pred_boxes = boxes
        else:
            pred_boxes = bbox_transform_inv_hdn(boxes, box_deltas)

        if clip:
            pred_boxes = clip_boxes(pred_boxes, im_info[0][:2] / im_info[0][2])

        # nms
        if nms and pred_boxes.shape[0] > 0:
            pred_boxes, scores, inds, keep_keep = nms_detections(pred_boxes, scores, 0.60, inds=inds)
            keep = keep[keep_keep]

        
        sub_list = np.array([], dtype=int)
        obj_list = np.array([], dtype=int)
        pred_list = np.array([], dtype=int)

        # print 'keep', keep
        # print 'mat_phrase', mat_phrase


        for i in range(mat_phrase.shape[0]):
            sub_id = np.where(keep == mat_phrase[i, 0])[0]
            obj_id = np.where(keep == mat_phrase[i, 1])[0]
            if len(sub_id) > 0 and len(obj_id) > 0:
                sub_list = np.append(sub_list, sub_id[0])
                obj_list = np.append(obj_list, obj_id[0])
                pred_list = np.append(pred_list, i)

        total_scores = predicate_scores.squeeze()[pred_list] \
                        * scores[sub_list].squeeze() * scores[obj_list].squeeze()
        top_N_list = total_scores.argsort()[::-1][:top_N]
        predicate_inds = predicate_inds.squeeze()[pred_list[top_N_list]]

        subject_inds = inds[sub_list[top_N_list]]
        object_inds = inds[obj_list[top_N_list]]
        subject_boxes = pred_boxes[sub_list[top_N_list]]
        object_boxes = pred_boxes[obj_list[top_N_list]]
        

        return pred_boxes, scores, inds, subject_inds, object_inds, subject_boxes, object_boxes, predicate_inds
               


    def interpret_result(self, cls_prob, bbox_pred, rois, cls_prob_predicate, 
                        mat_phrase, im_info, im_shape, nms=True, clip=True, min_score=0.01, 
                        use_gt_boxes=False):
        scores, inds = cls_prob[:, 0:].data.max(1)
        # inds += 1
        scores, inds = scores.cpu().numpy(), inds.cpu().numpy()
        predicate_scores, predicate_inds = cls_prob_predicate[:, 0:].data.max(1)
        # predicate_inds += 1
        predicate_scores, predicate_inds = predicate_scores.cpu().numpy(), predicate_inds.cpu().numpy()
        
        keep = np.where((inds > 0) & (scores >= min_score))
        scores, inds = scores[keep], inds[keep]

        # Apply bounding-box regression deltas
        keep = keep[0]
        box_deltas = bbox_pred.data.cpu().numpy()[keep]
        box_deltas = np.asarray([
            box_deltas[i, (inds[i] * 4): (inds[i] * 4 + 4)] for i in range(len(inds))
        ], dtype=np.float)
        boxes = rois.data.cpu().numpy()[keep, 1:5] / im_info[0][2]
        if use_gt_boxes:
            nms = False
            clip = False
            pred_boxes = boxes
        else:
            pred_boxes = bbox_transform_inv_hdn(boxes, box_deltas)

        if clip:
            pred_boxes = clip_boxes(pred_boxes, im_shape)

        # nms
        if nms and pred_boxes.shape[0] > 0:
            pred_boxes, scores, inds, keep_keep = nms_detections(pred_boxes, scores, 0.3, inds=inds)
            keep = keep[keep_keep]

        
        sub_list = np.array([], dtype=int)
        obj_list = np.array([], dtype=int)
        pred_list = np.array([], dtype=int)

        # print 'keep', keep
        # print 'mat_phrase', mat_phrase


        for i in range(mat_phrase.shape[0]):
            sub_id = np.where(keep == mat_phrase[i, 0])[0]
            obj_id = np.where(keep == mat_phrase[i, 1])[0]
            if len(sub_id) > 0 and len(obj_id) > 0:
                sub_list = np.append(sub_list, sub_id[0])
                obj_list = np.append(obj_list, obj_id[0])
                pred_list = np.append(pred_list, i)

        predicate_scores = predicate_scores.squeeze()[pred_list]
        final_list = predicate_scores.argsort()[::-1]
        predicate_inds = predicate_inds.squeeze()[pred_list[final_list]]
        sub_list = sub_list[final_list]
        obj_list = obj_list[final_list]
        region_list = mat_phrase[pred_list[final_list], 2:]
        

        return pred_boxes, scores, inds, sub_list, obj_list, predicate_inds, region_list


    def caption(self, im_path, gt_objects=None, gt_regions=None, thr=0.0, nms=False, top_N=100, clip=True, use_beam_search=False):
            image = cv2.imread(im_path)
            # print 'image.shape', image.shape
            im_data, im_scales = self.get_image_blob_noscale(image)
            # print 'im_data.shape', im_data.shape
            # print 'im_scales', im_scales
            if gt_objects is not None:
                gt_objects[:, :4] = gt_objects[:, :4] * im_scales[0]
            if gt_regions is not None:
                gt_regions[:, :4] = gt_regions[:, :4] * im_scales[0]

            im_info = np.array(
                [[im_data.shape[1], im_data.shape[2], im_scales[0]]],
                dtype=np.float32)
            # pdb.set_trace()
            region_result = self(im_data, im_info, gt_objects, gt_regions=gt_regions, use_beam_search=use_beam_search)[2]
            region_caption, bbox_pred, region_rois, logprobs = region_result[:]

            boxes = region_rois.data.cpu().numpy()[:, 1:5] / im_info[0][2]
            box_deltas = bbox_pred.data.cpu().numpy()
            pred_boxes = bbox_transform_inv_hdn(boxes, box_deltas)
            if clip:
                pred_boxes = clip_boxes(pred_boxes, image.shape)

            # print 'im_scales[0]', im_scales[0]
            return (region_caption.numpy(), logprobs.numpy(), pred_boxes)

    def describe(self, im_path, top_N=10):
            image = cv2.imread(im_path)
            # print 'image.shape', image.shape
            im_data, im_scales = self.get_image_blob_noscale(image)
            # print 'im_data.shape', im_data.shape
            # print 'im_scales', im_scales

            im_info = np.array(
                [[im_data.shape[1], im_data.shape[2], im_scales[0]]],
                dtype=np.float32)

            object_result, predicate_result, region_result = self(im_data, im_info)

            object_boxes, object_scores, object_inds, sub_assignment, obj_assignment, predicate_inds, region_assignment\
                     = self.interpret_result(object_result[0], object_result[1], object_result[2], \
                        predicate_result[0], predicate_result[1], \
                        im_info, image.shape) 

            region_caption, bbox_pred, region_rois, logprobs = region_result[:]
            boxes = region_rois.data.cpu().numpy()[:, 1:5] / im_info[0][2]
            box_deltas = bbox_pred.data.cpu().numpy()
            pred_boxes = bbox_transform_inv_hdn(boxes, box_deltas)
            pred_boxes = clip_boxes(pred_boxes, image.shape)

            # print 'im_scales[0]', im_scales[0]
            return (region_caption.numpy(), logprobs.numpy(), pred_boxes, \
                    object_boxes, object_inds, object_scores, \
                sub_assignment, obj_assignment, predicate_inds, region_assignment)


    def evaluate(self, im_data, im_info, gt_objects, gt_relationships, gt_regions, 
        thr=0.5, nms=False, top_Ns = [100], use_gt_boxes=False, use_gt_regions=False, only_predicate=False):
        
        if use_gt_boxes:
            gt_boxes_object = gt_objects[:, :4] * im_info[2]
        else:
            gt_boxes_object = None


        if use_gt_regions:
            gt_boxes_regions = gt_regions[:, :4] * im_info[2]
        else:
            gt_boxes_regions = None
        
        object_result, predicate_result, region_result = \
            self(im_data, im_info, gt_boxes_object, gt_regions=gt_boxes_regions, graph_generation=True)

        cls_prob_object, bbox_object, object_rois = object_result[:3]
        cls_prob_predicate, mat_phrase = predicate_result[:2]

        # interpret the model output
        obj_boxes, obj_scores, obj_inds, subject_inds, object_inds, \
            subject_boxes, object_boxes, predicate_inds = \
                self.interpret_HDN(cls_prob_object, bbox_object, object_rois, 
                            cls_prob_predicate, mat_phrase, im_info, 
                            nms=nms, top_N=max(top_Ns), use_gt_boxes=use_gt_boxes)

        gt_objects[:, :4] /= im_info[0][2]
        rel_cnt, rel_correct_cnt = check_relationship_recall(gt_objects, gt_relationships, 
                                        subject_inds, object_inds, predicate_inds, 
                                        subject_boxes, object_boxes, top_Ns, thres=thr, 
                                        only_predicate=only_predicate)

        return rel_cnt, rel_correct_cnt



    def build_loss_objectiveness(self, region_objectiveness, targets):
        loss_objectiveness = F.cross_entropy(region_objectiveness, targets)
        maxv, predict = region_objectiveness.data.max(1)
        labels = targets.squeeze()
        fg_cnt = torch.sum(labels.data.ne(0))
        bg_cnt = labels.data.numel() - fg_cnt
        if fg_cnt > 0:
            self.tp_reg = torch.sum(predict[:fg_cnt].eq(labels.data[:fg_cnt]))
        else:
            self.tp_reg = 0.
        if bg_cnt > 0:
            self.tf_reg = torch.sum(predict[fg_cnt:].eq(labels.data[fg_cnt:]))
        else:
            self.tp_reg = 0.
        self.fg_cnt_reg = fg_cnt
        self.bg_cnt_reg = bg_cnt
        return loss_objectiveness
