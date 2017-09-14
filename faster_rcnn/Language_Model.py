import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import pdb
from network import FC



# class Img_Encoder_Structure_v2(nn.Module):
#     '''
#     Here, we define another image encoder structure to transform the image vector
#     we Directly use several FC layers to transform the image
#     '''
#     def __init__(self, ninput, nhidden, nlayers):
#         super(Img_Encoder_Structure_v2, self).__init__()


class Img_Encoder_Structure(nn.Module):
    def __init__(self, ninput, nembed, nhidden, nlayers, bias, dropout):
        super(Img_Encoder_Structure, self).__init__()
        self.image_encoder = FC(ninput, nembed, relu=True)
        self.rnn = nn.LSTM(nembed, nhidden, nlayers, bias=bias, dropout=dropout)

    def forward(self, feat_im_with_seq_dim):
        # pdb.set_trace() 
        feat_im = self.image_encoder(feat_im_with_seq_dim[0])
        output, feat_im = self.rnn(feat_im.unsqueeze(0))
        return output, feat_im


class Language_Model(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, nimg, nhidden, nembed, nlayers, nseq, voc_sign, bias=False, dropout=0.):
        super(Language_Model, self).__init__()
        self.encoder = nn.Embedding(ntoken, nembed)
        if rnn_type == 'LSTM_im':
            self.lstm_im = nn.LSTM(nimg, nhidden, nlayers, bias=bias, dropout=dropout)
            self.lstm_word = nn.LSTM(nembed, nhidden, nlayers, bias=bias, dropout=dropout)
        elif rnn_type == 'LSTM_normal':
            self.lstm_im = nn.LSTM(nimg, nhidden, nlayers, bias=bias, dropout=dropout)
            self.lstm_word = self.lstm_im
        elif rnn_type == 'LSTM_baseline':
            self.lstm_im = Img_Encoder_Structure(nimg, nembed, nhidden, nlayers, bias=bias, dropout=dropout)
            self.lstm_word = self.lstm_im.rnn
        else:
            raise Exception('Cannot recognize LSTM type')
        self.decoder = nn.Linear(nhidden, ntoken, bias=bias)
        self.nseq = nseq
        self.end = voc_sign['end']
        self.null = voc_sign['null']
        self.start = voc_sign['start']
        self.word_weight = torch.ones(ntoken).cuda()
        self.word_weight[self.null] = 0.
        self.word_weight[self.end] = 0.1
        self.ntoken = ntoken
        self.bias = bias
        self.init_weights()
        self.nlayers = nlayers
        self.nhidden = nhidden
        self.nembed = nembed


    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        if self.bias:
            self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, seq=None):
        # input: image feature [N x F]
        # hidden: initial hidden state [nlayer x N x nhid]
        # seq: region descriptions [N x nseq], 1 for the 'start_token'
        # output: decoded: [N x nseq x ntoken]  
        if self.training:
            seq = torch.t(seq)
            im_batch_size = input.size()[0]
            im_feature_size = input.size()[1]
            seq_batch_size = seq.size()[1]
            seq_len = [np.where(seq[:, i].cpu().data.numpy() == self.end)[0][0] + 1 for i in range(seq.size(1))]
            input_seq = seq[:max(seq_len)- 1]
            target_seq = seq[1:max(seq_len)].clone()
            output_mask = input_seq.eq(self.end)
            target_seq[output_mask] = self.null
            seq_embed = self.encoder(input_seq)
            hidden_feat = self.lstm_im(\
                    input.view(1, im_batch_size, im_feature_size).expand(1, seq_batch_size, im_feature_size))[1]
            output, hidden_feat = self.lstm_word(seq_embed, hidden_feat)
            output = self.decoder(output.view(-1, output.size(2)))
            loss = F.cross_entropy(output, target_seq.view(-1), weight=self.word_weight)
            return loss
        else:
            batch_size = input.size(0)
            hidden_feat = self.lstm_im(input.view(1, input.size()[0], input.size()[1]))[1]
            x = Variable(torch.ones(1, batch_size,).type(torch.LongTensor) * self.start, requires_grad=False).cuda() # <start>
            output = []
            scores = torch.zeros(batch_size)
            flag = torch.ones(batch_size)
            for i in range(self.nseq):
                input_x = self.encoder(x.view(1, -1))
                output_feature, hidden_feat = self.lstm_word(input_x, hidden_feat)
                output_t = self.decoder(output_feature.view(-1, output_feature.size(2)))
                output_t = F.log_softmax(output_t)
                logprob, x = output_t.max(1)
                output.append(x)
                scores += logprob.cpu().data * flag
                flag[x.cpu().eq(self.end).data] = 0
                if flag.sum() == 0:
                    break
            output = torch.stack(output, 0).squeeze().transpose(0, 1)
            return output

    def baseline_search(self, input, beam_size=None):
        # This is the simple greedy search
        batch_size = input.size(0)
        hidden_feat = self.lstm_im(input.view(1, input.size()[0], input.size()[1]))[1]
        x = Variable(torch.ones(1, batch_size,).type(torch.LongTensor) * self.start, requires_grad=False).cuda() # <start>
        output = []
        flag = torch.ones(batch_size)
        for i in range(self.nseq):
            input_x = self.encoder(x.view(1, -1))
            output_feature, hidden_feat = self.lstm_word(input_x, hidden_feat)
            output_t = self.decoder(output_feature.view(-1, output_feature.size(2)))
            output_t = F.log_softmax(output_t)
            logprob, x = output_t.max(1)
            output.append(x)
            flag[x.cpu().eq(self.end).data] = 0
            if flag.sum() == 0:
                break
        output = torch.stack(output, 0).squeeze().transpose(0, 1).cpu().data
        return output

    def beamsearch(self, img_feat, beam_size):

        # There is something wrong with this part
        raise NotImplementedError 
        # pdb.set_trace()
        feature_len = img_feat.size(1)
        batch_size = img_feat.size(0)
        # First step transform the 
        hidden_feat = self.lstm_im(img_feat.unsqueeze(0))[1]

        batch_size = img_feat.size(0)
        seq = torch.LongTensor(self.nseq, batch_size).zero_()
        seqLogprobs = torch.FloatTensor(batch_size)
        self.done_beams = [[] for _ in range(batch_size)]

        # pdb.set_trace()

        for k in range(batch_size):
            # pdb.set_trace()
            state = (hidden_feat[0][:, k:k+1].expand(self.nlayers, beam_size, self.nhidden).contiguous(), \
                hidden_feat[1][:, k:k+1].expand(self.nlayers, beam_size, self.nhidden).contiguous())
            beam_seq = torch.LongTensor(self.nseq, beam_size).zero_()
            beam_seq_logprobs = torch.FloatTensor(self.nseq, beam_size).zero_()
            beam_logprobs_sum = torch.zeros(beam_size) # running sum of logprobs for each beam
            beam_end_state = torch.zeros(beam_size).type(torch.ByteTensor) # indicate whether predicting beams 

            for t in range(self.nseq+1):
                if t == 0:
                    it = Variable(torch.ones(beam_size,).type(torch.LongTensor) * self.start, requires_grad=False).cuda() # <start>
                    xt = self.encoder(it)
                else:
                    """perform a beam merge. that is,
                    for every previous beam we now many new possibilities to branch out
                    we need to resort our beams to maintain the loop invariant of keeping
                    the top beam_size most likely sequences."""
                    logprobsf = logprobs.cpu().data # lets go to CPU for more efficiency in indexing operations
                    ys,ix = logprobsf.topk(beam_size,1,True) # sorted array of logprobs along each previous beam (last true = descending)
                    candidates = []
                    assert ys.size(1) == beam_size
                    cols = beam_size
                    rows = beam_size
                    if t == 1:  # at first time step only the first beam is active
                        rows = 1
                    for q in range(rows):
                        for c in range(cols):
                            # compute logprob of expanding beam q with word in (sorted) position c
                            # if c > 0 and ix.data[q,c] == self.end: # don't 
                            #     c = cols # to replace with (beam_size + 1)th item
                            if beam_end_state[q]:
                                local_logprob = 0.
                                candidate_logprob = beam_logprobs_sum[q]
                                candidates.append({'c':self.end, 'q':q, 'p':candidate_logprob, 'r':local_logprob})
                                
                                break
                            else:
                                local_logprob = ys[q,c]
                                candidate_logprob = ((beam_logprobs_sum[q]) * (t-1) + local_logprob) / float(t)
                                candidates.append({'c':ix[q,c], 'q':q, 'p':candidate_logprob, 'r':local_logprob})
                            
                    candidates = sorted(candidates, key=lambda x: -x['p'])

                    # construct new beams
                    new_state = [_.clone() for _ in state]
                    if t > 1:
                        # well need these as reference when we fork beams around
                        beam_seq_prev = beam_seq[:t-1].clone()
                        beam_seq_logprobs_prev = beam_seq_logprobs[:t-1].clone()

                    beam_end_state_prev = beam_end_state.clone()
                    for vix in range(beam_size):
                        v = candidates[vix]
                        # To change the beam state accordingly
                        beam_end_state[vix] = True if v['c'] == self.end else beam_end_state_prev[v['q']]
                        # fork beam index q into index vix
                        if t > 1:
                            beam_seq[:t-1, vix] = beam_seq_prev[:, v['q']]
                            beam_seq_logprobs[:t-1, vix] = beam_seq_logprobs_prev[:, v['q']]

                        # rearrange recurrent states
                        for state_ix in range(len(new_state)):
                            # copy over state in previous beam q to new beam at vix
                            new_state[state_ix][0, vix] = state[state_ix][0, v['q']] # dimension one is time step

                        # append new end terminal at the end of this beam
                        beam_seq[t-1, vix] = v['c'] # c'th word is the continuation
                        beam_seq_logprobs[t-1, vix] = v['r'] # the raw logprob here
                        try:
                            beam_logprobs_sum[vix] = v['p'] # the new (sum) logprob along this beam
                        except Exception:
                            pdb.set_trace()

                    if beam_end_state.all() or t == self.nseq:

                        for vix in range(beam_size):
                            # END token special case here, or we reached the end.
                            # add the beam to a set of done beams
                            self.done_beams[k].append({'seq': beam_seq[:, vix].clone(), 
                                                'logps': beam_seq_logprobs[:, vix].clone(),
                                                'p': beam_logprobs_sum[vix]
                                                })
                            
                        break
                    # encode as vectors
                    it = beam_seq[t-1]
                    xt = self.encoder(Variable(it.cuda()))
                
                if t >= 1:
                    state = new_state

                output, state = self.lstm_word(xt.unsqueeze(0), state)
                logprobs = F.log_softmax(self.decoder(output.squeeze(0)))
            
            self.done_beams[k] = sorted(self.done_beams[k], key=lambda x: -x['p'])
            seq[:, k] = self.done_beams[k][0]['seq'] # the first beam has highest cumulative score
            seqLogprobs[k] = self.done_beams[k][0]['p']
        # return the samples and their log likelihoods
        # pdb.set_trace()
        return seq.transpose(0, 1)





    def init_hidden(self, bsz):
        return (Variable(weight.new(self.num_layers, bsz, self.rnn_size).zero_()),
                Variable(weight.new(self.num_layers, bsz, self.rnn_size).zero_()))