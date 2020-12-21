import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)


class DSMN_SAIM(nn.Module):

    def __init__(self, embedding_size, embedding_dimension, embedding_matrix, hidden_size, n_hop):
        super(DSMN_SAIM, self).__init__()

        self.embedding_dimension = embedding_dimension
        self.hidden_size = hidden_size
        self.n_hop = n_hop

        self.embedding_layer = nn.Embedding(embedding_size, embedding_dimension)
        self.embedding_layer.weight.data.copy_(torch.from_numpy(embedding_matrix))
        self.embedding_layer.weight.requires_grad = False
        self.activate = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

        self.sen_bilstm = nn.LSTM(input_size=self.embedding_dimension, hidden_size=self.hidden_size, batch_first=True,
                                  bidirectional=True).cuda()
        self.asp_bilstm = nn.LSTM(input_size=self.embedding_dimension, hidden_size=self.hidden_size, batch_first=True,
                                  bidirectional=True).cuda()

        self.sen_att1 = nn.Linear(self.hidden_size, 1).cuda()
        self.sen_att2 = nn.Linear(self.hidden_size * 2, self.hidden_size).cuda()
        self.asp_att1 = nn.Linear(self.hidden_size, 1).cuda()
        self.asp_att2 = nn.Linear(self.hidden_size * 2, self.hidden_size).cuda()

        self.var_linear1 = nn.Linear(self.hidden_size * 2, self.hidden_size * 2).cuda()
        self.var_linear2 = nn.Linear(self.hidden_size * 2, 2).cuda()

        self.attention = nn.Linear(self.hidden_size * 2, self.hidden_size * 2).cuda()
        self.transform_linear = nn.Linear(self.hidden_size * 2, self.hidden_size * 2).cuda()

        self.aspect_attention = nn.Linear(self.hidden_size * 2, self.hidden_size * 2).cuda()
        self.gate = nn.Linear(self.hidden_size * 4, self.hidden_size * 2).cuda()
        self.fuse = nn.Linear(self.hidden_size * 4, self.hidden_size * 2).cuda()
        self.predict_linear = nn.Linear(self.hidden_size * 2, 3).cuda()
        self.loss = torch.nn.CrossEntropyLoss()

        self.initialize_weights()

    def soft_cross_entropy(self, input, target):
        logsoftmax = nn.LogSoftmax(dim=1)
        return torch.sum(-target * logsoftmax(input), dim=1)

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_normal_(param)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0.0)
                        param.chunk(4)[1].fill_(1)

    def lstm_forward(self, lstm, inputs, seq_lengths):
        sorted_seq_lengths, indices = torch.sort(seq_lengths, descending=True)
        _, desorted_indices = torch.sort(indices, descending=False)
        inputs = inputs[indices]
        packed_inputs = nn.utils.rnn.pack_padded_sequence(inputs, sorted_seq_lengths, batch_first=True)
        lstm.flatten_parameters()
        res, state = lstm(packed_inputs)
        padded_res, _ = nn.utils.rnn.pad_packed_sequence(res, batch_first=True)
        desorted_res = padded_res[desorted_indices]
        return desorted_res

    def forward(self, data, dropout=0.0):
        sentences = torch.tensor(data['sentences']).type(torch.cuda.LongTensor)
        num = torch.tensor(data['num']).type(torch.cuda.LongTensor)
        sentence_lens = torch.tensor(data['sentence_lens']).type(torch.cuda.LongTensor)
        aspects = torch.tensor(data['aspects']).type(torch.cuda.LongTensor)
        aspect_lens = torch.tensor(data['aspect_lens']).type(torch.cuda.LongTensor)
        sentence_infos_pos = torch.tensor(data['sentence_infos_pos']).type(torch.cuda.FloatTensor)
        sentence_locs_pos = torch.tensor(data['sentence_locs_pos']).type(torch.cuda.FloatTensor)
        aspect_locs_pos = torch.tensor(data['aspect_locs_pos']).type(torch.cuda.FloatTensor)
        sentence_infos_sem = torch.tensor(data['sentence_infos_sem']).type(torch.cuda.FloatTensor)
        sentence_locs_sem = torch.tensor(data['sentence_locs_sem']).type(torch.cuda.FloatTensor)
        aspect_locs_sem = torch.tensor(data['aspect_locs_sem']).type(torch.cuda.FloatTensor)
        aspect_vars_pos = torch.tensor(data['aspect_vars_pos']).type(torch.cuda.FloatTensor)
        aspect_vars_sem = torch.tensor(data['aspect_vars_sem']).type(torch.cuda.FloatTensor)
        labels = torch.tensor(data['labels']).type(torch.cuda.LongTensor)

        dropout_layer = nn.Dropout(dropout)

        sentence_inputs = self.embedding_layer(sentences)
        sentence_inputs = dropout_layer(sentence_inputs)
        sentence_outputs = self.lstm_forward(self.sen_bilstm, sentence_inputs, sentence_lens)
        
        max_sentence_len = sentence_outputs.size()[1]

        specific_aspects, specific_aspect_lens, specific_aspect_vars = [], [], []
        specific_sentence_infos, specific_sentence_locs, specific_sentence_outputs, specific_sentence_lens = [], [], [], []
        specific_nums, specific_labels = [], []
        total_num = 0
        for i in range(len(sentences)):
            specific_aspects.append(aspects[i, :num[i], :])
            specific_aspect_lens.append(aspect_lens[i, :num[i]])
            specific_aspect_vars.append(aspect_vars_pos[i, :num[i], :])
            specific_sentence_infos.append(sentence_infos_pos[i, :num[i], :, :])
            specific_sentence_locs.append(sentence_locs_pos[i, :num[i], :max_sentence_len])
            aspect_num = num[i].item()
            specific_sentence_outputs.append(sentence_outputs[i, :, :].expand(aspect_num, max_sentence_len,
                                                                              self.hidden_size * 2))
            specific_sentence_lens.append(sentence_lens[i].expand(aspect_num))
            specific_nums.append(num[i].expand(aspect_num))
            specific_labels.append(labels[i, :num[i], :])
            total_num += aspect_num

        specific_aspects = torch.cat(specific_aspects, dim=0)
        specific_aspect_lens = torch.cat(specific_aspect_lens, dim=0)
        specific_aspect_vars = torch.cat(specific_aspect_vars, dim=0)
        specific_sentence_infos = torch.cat(specific_sentence_infos, dim=0)
        specific_sentence_locs = torch.cat(specific_sentence_locs, dim=0)
        specific_sentence_outputs = torch.cat(specific_sentence_outputs, dim=0)
        specific_sentence_lens = torch.cat(specific_sentence_lens, dim=0)
        specific_nums = torch.cat(specific_nums, dim=0)
        specific_labels = torch.cat(specific_labels, dim=0)

        aspect_inputs = self.embedding_layer(specific_aspects)
        aspect_inputs = dropout_layer(aspect_inputs)
        aspect_outputs = self.lstm_forward(self.asp_bilstm, aspect_inputs, specific_aspect_lens)

        max_aspect_len = aspect_outputs.size()[1]
        aspect_mask = torch.ones(total_num, max_aspect_len).cuda()
        for i in range(total_num):
            aspect_mask[i, specific_aspect_lens[i]:] = 0
        aspect_outputs_flatten = aspect_outputs.view(-1, self.hidden_size * 2)
        aspect_outputs_weight = self.asp_att1(self.activate(self.asp_att2(aspect_outputs_flatten)))
        aspect_outputs_weight = aspect_outputs_weight.view(total_num, max_aspect_len)
        aspect_outputs_weight = aspect_outputs_weight - (1 - aspect_mask) * 1e12
        aspect_outputs_weight = F.softmax(aspect_outputs_weight, dim=1).unsqueeze(-1).expand(total_num,
                                                                                             max_aspect_len,
                                                                                             self.hidden_size * 2)
        weighted_aspect_outputs = aspect_outputs_weight * aspect_outputs
        weighted_aspect_outputs = weighted_aspect_outputs.view(-1, max_aspect_len, self.hidden_size * 2)
        aspect_output = torch.sum(weighted_aspect_outputs, dim=1)

        # local feature
        e = torch.zeros([total_num, self.hidden_size * 2]).cuda()
        scores_list = []
        for h in range(self.n_hop):
            sentences_info = specific_sentence_infos[:, h, :max_sentence_len]
            # select
            memory = specific_sentence_outputs * sentences_info.unsqueeze(-1).expand(total_num, max_sentence_len,
                                                                                     self.hidden_size * 2)
            attention_score = torch.bmm(self.attention(memory), aspect_output.unsqueeze(-1))
            attention_score = attention_score.squeeze(-1) - (1 - sentences_info) * 1e12
            attention_score = F.softmax(attention_score, dim=1)
            scores_list.append(attention_score)
            i_AL = torch.sum(
                attention_score.unsqueeze(-1).expand(total_num, max_sentence_len, self.hidden_size * 2) * memory,
                dim=1)

            # i_AL = i_AL1 + i_AL2
            T = self.sigmoid(self.transform_linear(e))
            e = i_AL * T + e * (1 - T)
        sentence_mask = torch.ones(total_num, max_sentence_len).cuda()
        for i in range(total_num):
            sentence_mask[i, specific_sentence_lens[i]:] = 0
        aspect_sentence_outputs = specific_sentence_outputs * specific_sentence_locs.unsqueeze(-1).expand(total_num,
                                                                                                          max_sentence_len,
                                                                                                          self.hidden_size * 2)
        sentence_outputs_flatten = aspect_sentence_outputs.view(-1, self.hidden_size * 2)
        sentence_outputs_weight = self.sen_att1(self.activate(self.sen_att2(sentence_outputs_flatten)))
        sentence_outputs_weight = sentence_outputs_weight.view(total_num, max_sentence_len)
        sentence_outputs_weight = sentence_outputs_weight - (1 - sentence_mask) * 1e12
        sentence_outputs_weight = F.softmax(sentence_outputs_weight, dim=1).unsqueeze(-1).expand(total_num,
                                                                                                 max_sentence_len,
                                                                                                 self.hidden_size * 2)
        weighted_sentence_outputs = sentence_outputs_weight * specific_sentence_outputs
        weighted_sentence_outputs = weighted_sentence_outputs.view(-1, max_sentence_len, self.hidden_size * 2)
        sentence_output = torch.sum(weighted_sentence_outputs, dim=1)
        var_vec = self.var_linear1(sentence_output)

        sen_asp_output, specific_var_vec, specific_glo_vec = [], [], []
        cnt = 0
        for i in range(len(num)):
            cur_num = num[i].item()
            cur_e = e[cnt:cnt + cur_num, :]
            cur_aspect_loc = aspect_locs_pos[i, :cur_num, :cur_num]
            cur_aspect_loc = cur_aspect_loc * (1 -  torch.eye(cur_num).cuda())
            weighted_cur_e = cur_e.unsqueeze(0).expand(cur_num, cur_num,
                                                       self.hidden_size * 2) * cur_aspect_loc.unsqueeze(
                2).expand(cur_num, cur_num, self.hidden_size * 2)
            glo_vec = torch.sum(weighted_cur_e, dim=1)
            specific_glo_vec.append(glo_vec)
            cnt += cur_num

        specific_glo_vec = torch.cat(specific_glo_vec, dim=0).view(total_num, self.hidden_size * 2)

        global_feature = torch.cat([specific_glo_vec, var_vec], dim=-1)
        global_feature = self.fuse(global_feature)
        global_feature = (specific_nums > 1).float().unsqueeze(-1).expand(total_num,
                                                                          self.hidden_size * 2) * global_feature

        pvar = self.var_linear2(global_feature)
        sentence_cost = self.soft_cross_entropy(pvar, specific_aspect_vars)
        sentence_cost = (specific_nums > 1).float() * sentence_cost
        sentence_cost = torch.mean(sentence_cost)

        gate_var_vec = (specific_nums > 1).float().unsqueeze(-1).expand(total_num,
                                                                          self.hidden_size * 2) * var_vec
        gate_rep = torch.cat([e, gate_var_vec], dim=-1)
        gate_res = self.sigmoid(self.gate(gate_rep))
        final_rep = gate_res * e + (1 - gate_res) * global_feature
        predict = self.predict_linear(final_rep)
        specific_predict_labels = torch.argmax(predict, dim=1)
        specific_labels = torch.argmax(specific_labels, dim=1)
        correct_num = (specific_predict_labels.eq(specific_labels)).sum()
        cost = self.loss(predict, specific_labels)

        return cost, sentence_cost, total_num, correct_num, specific_predict_labels, specific_labels
