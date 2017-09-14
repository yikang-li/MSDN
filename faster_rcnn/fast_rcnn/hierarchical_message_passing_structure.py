import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Parameter
from ..utils.timer import Timer
import pdb
from mps_base import Hierarchical_Message_Passing_Structure_base
from config import cfg

TIME_IT = cfg.TIME_IT

class Hierarchical_Message_Passing_Structure(Hierarchical_Message_Passing_Structure_base):
	def __init__(self, fea_size, dropout=False, gate_width=128, use_kernel_function=False):
		super(Hierarchical_Message_Passing_Structure, self).__init__(fea_size, dropout, gate_width, 
						use_region=True, use_kernel_function=use_kernel_function)

	def forward(self, feature_obj, feature_phrase, feature_region, mps_object, mps_phrase, mps_region):

		# mps_object [object_batchsize, 2, n_phrase] : the 2 channel means inward(object) and outward(subject) list
		# mps_phrase [phrase_batchsize, 2 + n_region]
		# mps_region [region_batchsize, n_phrase]
		t = Timer()
		# pdb.set_trace()
		t.tic()

		# object updating
		object_sub = self.prepare_message(feature_obj, feature_phrase, mps_object[:, 0, :], self.gate_pred2sub)
		object_obj = self.prepare_message(feature_obj, feature_phrase, mps_object[:, 1, :], self.gate_pred2obj)
		GRU_input_feature_object = (object_sub + object_obj) / 2.
		out_feature_object = feature_obj + self.GRU_object(GRU_input_feature_object, feature_obj)
		if TIME_IT:
			torch.cuda.synchronize()
			print '\t\t[object pass]:\t%.3fs' % (t.toc(average=False))


		# phrase updating
		t.tic()
		indices_sub = Variable(torch.from_numpy(mps_phrase[:, 0]).type(torch.LongTensor)).cuda().detach()
		indices_obj = Variable(torch.from_numpy(mps_phrase[:, 1]).type(torch.LongTensor)).cuda().detach()
		fea_sub2pred = torch.index_select(feature_obj, 0, indices_sub)
		fea_obj2pred = torch.index_select(feature_obj, 0, indices_obj)
		phrase_sub = self.gate_sub2pred(feature_phrase, fea_sub2pred)
		phrase_obj = self.gate_obj2pred(feature_phrase,  fea_obj2pred)
		# pdb.set_trace()
		phrase_region = self.prepare_message(feature_phrase, feature_region, mps_phrase[:, 2:], self.gate_reg2pred)
		GRU_input_feature_phrase =  phrase_sub / 3. + phrase_obj / 3. + phrase_region / 3.
		if TIME_IT:
			torch.cuda.synchronize()
			print '\t\t[phrase pass]:\t%.3fs' % (t.toc(average=False))
		out_feature_phrase = feature_phrase + self.GRU_phrase(GRU_input_feature_phrase, feature_phrase)

		# region updating
		t.tic()
		GRU_input_feature_region = self.prepare_message(feature_region, feature_phrase, mps_region, self.gate_pred2reg)
		out_feature_region = feature_region + self.GRU_region(GRU_input_feature_region, feature_region)
		if TIME_IT:
			torch.cuda.synchronize()
			print '\t\t[region pass]:\t%.3fs' % (t.toc(average=False))

		return out_feature_object, out_feature_phrase, out_feature_region