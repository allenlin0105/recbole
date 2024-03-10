# -*- coding: utf-8 -*-
# @Time    : 2020/9/18 11:33
# @Author  : Hui Wang
# @Email   : hui.wang@ruc.edu.cn

"""
SASRec
################################################

Reference:
    Wang-Cheng Kang et al. "Self-Attentive Sequential Recommendation." in ICDM 2018.

Reference:
    https://github.com/kang205/SASRec

"""

import json
import random
import copy
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from recbole.model.sequential_recommender.sasrec import SASRec


class AugRec(SASRec):
    r"""
    SASRec is the first sequential recommender based on self-attentive mechanism.

    NOTE:
        In the author's implementation, the Point-Wise Feed-Forward Network (PFFN) is implemented
        by CNN with 1x1 kernel. In this implementation, we follows the original BERT implementation
        using Fully Connected Layer to implement the PFFN.
    """

    def __init__(self, config, dataset):
        super(AugRec, self).__init__(config, dataset)
        
        # load parameters info
        self.batch_size = config['train_batch_size']
        self.tau = config['tau']
        self.cl_lambda = config['cl_lambda']
        self.cl_loss_type = config['cl_loss_type']
        self.similarity_type = config['similarity_type']

        # define layers and loss
        self.item_embedding = nn.Embedding(self.n_items + 1, self.hidden_size, padding_idx=0)  # for mask
        self.default_mask = self.mask_correlated_samples(self.batch_size)

        if self.similarity_type == 'dot':
            self.sim = torch.mm
        elif self.similarity_type == 'cos':
            self.sim = F.cosine_similarity

        if self.cl_loss_type == 'infonce':
            self.cl_loss_fct = nn.CrossEntropyLoss()
        
        # parameters initialization
        self.apply(self._init_weights)

        # load augmented sequence
        with open(Path(dataset.dataset_path, "user_id2token_id.json"), "r") as fp:
            user_id2token_id = json.load(fp)

        with open(Path(dataset.dataset_path, "item_id2token_id.json"), "r") as fp:
            item_id2token_id = json.load(fp)

        self.user_id2aug_sequence = {}
        for seq_file in Path(dataset.dataset_path, "aug_sequence").iterdir():
            user_id = seq_file.name.removesuffix("_1.txt")
            if user_id not in user_id2token_id:
                continue
            
            user_id = user_id2token_id[user_id]
            with open(seq_file, "r") as fp:
                sequence = []
                for line in fp:
                    if line.strip() in item_id2token_id:
                        sequence.append(item_id2token_id[line.strip()])
                self.user_id2aug_sequence[user_id] = sequence

    def mask_correlated_samples(self, batch_size):
        N = batch_size
        mask = torch.ones((2 * N, 2 * N)).bool()
        mask = mask.fill_diagonal_(0)
        mask *= ~ torch.diagflat(torch.ones(N), offset=N).bool()
        mask *= ~ torch.diagflat(torch.ones(N), offset=-N).bool()
        return mask

    def calculate_loss(self, interaction):
        loss = super().calculate_loss(interaction)
        cl_loss = self.calculate_cl_loss(copy.deepcopy(interaction))
        if cl_loss != None:
            return loss, self.cl_lambda * cl_loss
        else:
            return loss
   
    def calculate_cl_loss(self, interaction):
        """
        Need to modify:
        - self.ITEM_SEQ: (batch_size, seq_len) -> each sequence item ids
        - self.ITEM_SEQ_LEN: (batch_size) -> each sequence's length
        - self.POS_ITEM_ID: (batch_size) -> last item id in sequence
        """
        pad_to_len = interaction[self.ITEM_SEQ].shape[1]

        item_seq = []
        item_seq_len = []
        pos_item_id = []

        user_ids = interaction[self.USER_ID].detach().cpu()
        for user_id in user_ids:
            if user_id not in self.user_id2aug_sequence:
                continue
            aug_sequence = self.user_id2aug_sequence[user_id]
            if len(aug_sequence) + 1 > pad_to_len:
                aug_sequence = aug_sequence[:pad_to_len + 1]

            pos_item_id.append(aug_sequence[-1])

            aug_sequence = aug_sequence[:-1]
            item_seq.append(aug_sequence + [0] * (pad_to_len - len(aug_sequence)))
            item_seq_len.append(len(aug_sequence))
        
        if len(item_seq) == 0:
            return None
        
        device = interaction[self.ITEM_SEQ].get_device()
        interaction[self.ITEM_SEQ] = torch.tensor(item_seq).to(device)
        interaction[self.ITEM_SEQ_LEN] = torch.tensor(item_seq_len).to(device)
        interaction[self.POS_ITEM_ID] = torch.tensor(pos_item_id).to(device)

        cl_loss = super().calculate_loss(interaction)
        # aug_item_seq1, aug_len1, aug_item_seq2, aug_len2 = \
        #     interaction['aug1'], interaction['aug_len1'], interaction['aug2'], interaction['aug_len2']
        # seq_output1 = self.forward(aug_item_seq1, aug_len1)
        # seq_output2 = self.forward(aug_item_seq2, aug_len2)

        # logits, labels = self.info_nce(seq_output1, seq_output2)

        # if self.cl_loss_type == 'dcl': # decoupled contrastive learning
        #     cl_loss = self.calculate_decoupled_cl_loss(logits, labels)
        # else: # original infonce
        #     cl_loss = self.cl_loss_fct(logits, labels)
        return cl_loss
    
    # def calculate_decoupled_cl_loss(self, input, target):
    #     input_pos = torch.gather(input, 1, target.unsqueeze(-1)).squeeze(-1)
    #     input_exp = torch.exp(input)
    #     input_pos_exp = torch.exp(input_pos)
    #     input_neg_exp_sum = torch.sum(input_exp, dim=1) - input_pos_exp
    #     dcl_loss = torch.mean(-input_pos + torch.log(input_neg_exp_sum))
    #     return dcl_loss

    # def info_nce(self, z_i, z_j):
    #     """
    #     We do not sample negative examples explicitly.
    #     Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
    #     """
    #     cur_batch_size = z_i.size(0)
    #     N = 2 * cur_batch_size
    #     if cur_batch_size != self.batch_size:
    #         mask = self.mask_correlated_samples(cur_batch_size)
    #     else:
    #         mask = self.default_mask
    #     z = torch.cat((z_i, z_j), dim=0)  # [2B H]
    
    #     if self.similarity_type == 'cos':
    #         sim = self.sim(z.unsqueeze(1), z.unsqueeze(0), dim=2) / self.tau
    #     elif self.similarity_type == 'dot':
    #         sim = self.sim(z, z.T) / self.tau

    #     sim_i_j = torch.diag(sim, cur_batch_size)
    #     sim_j_i = torch.diag(sim, -cur_batch_size)

    #     positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)  # [2B, 1]
    #     negative_samples = sim[mask].reshape(N, -1)  # [2B, 2(B-1)]

    #     logits = torch.cat((positive_samples, negative_samples), dim=1)  # [2B, 2B-1]
    #     # the first column stores positive pair scores
    #     labels = torch.zeros(N, dtype=torch.long, device=z_i.device)
    #     return logits, labels

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        test_items_emb = self.item_embedding.weight[:self.n_items]
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B n_items]
        return scores
