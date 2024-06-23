import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import trange
from sklearn.decomposition import PCA

from recbole.model.sequential_recommender.cl4rec import CL4Rec


class LLMRec(CL4Rec):
    def __init__(self, config, dataset):
        super(LLMRec, self).__init__(config, dataset)

        # load parameters info
        self.device = config["gpu_id"]
        self.cor_lambda = config["cor_lambda"]
        self.cor_loss_fct = nn.KLDivLoss(reduction="batchmean")

        self._load_embed(dataset)

        # self.item_embedding.weight = torch.nn.Parameter(self.item_embed.to(torch.float32))
        # self.item_embedding.weight.requires_grad = True

    def _load_embed(self, dataset):
        # MODELPATH = "meta-llama/Llama-2-7b-chat-hf"
        MODELPATH = "meta-llama/Meta-Llama-3-8B-Instruct"

        llm = AutoModelForCausalLM.from_pretrained(
            MODELPATH,    
            device_map={"": self.device},
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            ),
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,      
        )

        tokenizer = AutoTokenizer.from_pretrained(
            MODELPATH,
            trust_remote_code=True,
            padding_side="left",
        )      
        tokenizer.pad_token = tokenizer.eos_token

        user_id2token_id = dataset.field2token_id["user_id"]
        item_id2token_id = dataset.field2token_id["item_id"]
        n_user, n_item = len(user_id2token_id), len(item_id2token_id)
        n_batch = 16
        # self.item_embed = torch.rand((n_item + 1, 64)).to(self.device)
        # self.interest_embed = torch.rand((n_user, 64)).to(self.device)
        # return

        """
        load item embed
        """
        item_names = [None] * n_item
        with open(Path(dataset.dataset_path, f"{dataset.dataset_name}.item"), "r") as fp:
            reader = csv.reader(fp, delimiter="\t")
            next(reader)

            for row in reader:
                item_id, item_name = row[0], row[1]
                if item_id not in item_id2token_id:
                    continue
                token_id = int(item_id2token_id[item_id])
                item_names[token_id] = item_name
        
        # self.item_embed = torch.zeros((n_item + 1, 4096))
        item_embed = []
        for i in trange(1, n_item, n_batch, desc="Item embed"):
            batch = min(n_batch, n_item - i)
            item_embed.append(self._encode_texts(
                item_names[i:i+batch], tokenizer, llm
            ).float().numpy())
        item_embed = np.concatenate(item_embed)

        pca = PCA(n_components=self.hidden_size)
        self.item_embed = torch.cat([
            torch.zeros((1, self.hidden_size)),  # [PAD]
            torch.from_numpy(pca.fit_transform(item_embed)),  # n_item
            torch.zeros((1, self.hidden_size)),  # [MASK]
        ]).to(self.device)
        print("item embed shape:", self.item_embed.shape)

        """
        load interest embed
        """
        # interests = [None] * n_user
        # for interest_file in Path("dataset", dataset.dataset_name, "interest/llama3_v1").iterdir():
        # # for interest_file in Path("../llama3_v1").iterdir():
        #     with open(interest_file, "r") as fp:
        #         token_id = int(user_id2token_id[interest_file.stem])
        #         # interests[token_id] = np.array([line.strip() for line in fp])
        #         interests[token_id] = ", ".join([line.strip() for line in fp])

        # # self.interest_embed = torch.zeros((n_user, 5, 4096))
        # interest_embed = []
        # for i in trange(1, n_user, n_batch, desc="Interest embed"):
        #     batch = min(n_batch, n_user - i)
        #     interest_embed.append(self._encode_texts(
        #         interests[i:i+batch], tokenizer, llm
        #     ).float().numpy())
        # interest_embed = np.concatenate(interest_embed)
        
        # pca = PCA(n_components=self.hidden_size)
        # self.interest_embed = torch.cat([
        #     torch.zeros((1, self.hidden_size)),  # [PAD]
        #     torch.from_numpy(pca.fit_transform(interest_embed))  # n_item
        # ]).to(self.device)
        # print("interest embed shape:", self.interest_embed.shape)

    def _encode_texts(self, texts, tokenizer, llm):
        t_input = tokenizer(texts, padding=True, return_tensors="pt").to(self.device)
        attention_mask = t_input.attention_mask
        with torch.no_grad():
            last_hidden_state = llm(**t_input, output_hidden_states=True).hidden_states[-1]

        sum_embeddings = torch.sum(last_hidden_state * attention_mask.unsqueeze(-1), dim=1)
        num_of_none_padding_tokens = torch.sum(attention_mask, dim=-1).unsqueeze(-1)
        sentence_embeddings = sum_embeddings / num_of_none_padding_tokens
        return sentence_embeddings.to("cpu")

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]

        if self.loss_type == 'BPR':
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)  # [B]
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)  # [B]
            loss = self.loss_fct(pos_score, neg_score)
        else:  # self.loss_type = 'CE'
            test_item_emb = self.item_embedding.weight
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            loss = self.loss_fct(logits, pos_items)
        
        un_aug_seq_output = self.forward(item_seq, item_seq_len)
        aug_item_seq, aug_item_seq_len = interaction['aug'], interaction['aug_len']
        su_aug_seq_output = self.forward(aug_item_seq, aug_item_seq_len)

        # logits, labels = self.info_nce(un_aug_seq_output, su_aug_seq_output)
        logits, labels = self.new_info_nce(un_aug_seq_output, su_aug_seq_output, pos_items)
        cl_loss = self.cl_lambda * self.cl_loss_fct(logits, labels)

        cor_loss = self.cor_lambda * self.calculate_cor_loss(interaction, un_aug_seq_output, su_aug_seq_output)
        return tuple([loss, cl_loss, cor_loss])

    def new_info_nce(self, z_i, z_j, pos_items):
        cur_batch_size = z_i.size(0)
        N = 2 * cur_batch_size
        z = torch.cat((z_i, z_j), dim=0)  # [2B H]
    
        if self.similarity_type == 'cos':
            sim = self.sim(z.unsqueeze(1), z.unsqueeze(0), dim=2) / self.tau
        elif self.similarity_type == 'dot':
            sim = self.sim(z, z.T) / self.tau
        # print(sim.shape)

        pos_llama_embed = self.item_embed[pos_items.cpu().tolist()]  # [B H]
        pos_llama_embed = torch.cat((pos_llama_embed, pos_llama_embed), dim=0)
        pos_sim = F.cosine_similarity(pos_llama_embed.unsqueeze(1), pos_llama_embed.unsqueeze(0), dim=2)  # [2B 2B]

        mask = torch.ones((N, N)).bool()

        _, mask_index = torch.topk(pos_sim, 32, dim=1)
        batch_index = torch.arange(N).reshape(N, 1)
        mask[batch_index, mask_index] = 0

        sim_i_j = torch.diag(sim, cur_batch_size)
        sim_j_i = torch.diag(sim, -cur_batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)  # [2B, 1]
        negative_samples = sim[mask].reshape(N, -1)  # [2B, 2(B-1)]

        logits = torch.cat((positive_samples, negative_samples), dim=1)  # [2B, 2B-1]
        # the first column stores positive pair scores
        labels = torch.zeros(N, dtype=torch.long, device=z_i.device)
        return logits, labels

    def calculate_cor_loss(self, interaction, seq_output1, seq_output2):
        # user_ids = interaction[self.USER_ID].detach().cpu().tolist()
        # batch_size = len(user_ids)
        # user_embed = self.interest_embed[user_ids]

        item_seq, item_seq_len = interaction[self.ITEM_SEQ], interaction[self.ITEM_SEQ_LEN]
        aug_embed_seq1 = torch.stack([
            torch.sum(self.item_embed[seq], dim=0) / item_seq_len[i].cpu()
            for i, seq in enumerate(item_seq.detach().cpu().tolist())
        ]) # (B, D)
        # aug_confidence1 = F.cosine_similarity(user_embed, aug_embed_seq1) / self.tau # (B,)
        target_logits1 = F.softmax(torch.matmul(aug_embed_seq1, self.item_embed.transpose(0, 1)), dim=1)

        aug_item_seq, aug_item_seq_len = interaction['aug'], interaction['aug_len']
        aug_embed_seq2 = torch.stack([
            torch.sum(self.item_embed[seq], dim=0) / aug_item_seq_len[i].cpu()
            for i, seq in enumerate(aug_item_seq.detach().cpu().tolist())
        ]) # (B, D)
        # aug_confidence2 = F.cosine_similarity(user_embed, aug_embed_seq2) / self.tau # (B,)
        target_logits2 = F.softmax(torch.matmul(aug_embed_seq2, self.item_embed.transpose(0, 1)), dim=1)

        target_logits = torch.cat([target_logits1, target_logits2])

        test_item_emb = self.item_embedding.weight
        aug_logits1 = F.softmax(torch.matmul(seq_output1, test_item_emb.transpose(0, 1)), dim=1)
        aug_logits2 = F.softmax(torch.matmul(seq_output2, test_item_emb.transpose(0, 1)), dim=1)
        aug_logits = torch.cat([aug_logits1, aug_logits2])
        
        loss = self.cor_loss_fct(aug_logits.log(), target_logits)
        return loss

        # confidence_deficit = aug_confidence1 - aug_confidence2
        # distance = 1 - F.cosine_similarity(seq_output1, seq_output2) / self.tau
        # loss = self.cor_loss_fct(distance, confidence_deficit)
        # return loss

        # mask = torch.ones((batch_size, batch_size)).triu(diagonal=1).to(self.device)
        # user_distance = (1 - torch.matmul(
        #     F.normalize(user_embed),
        #     F.normalize(user_embed).transpose(0, 1),
        # )) * mask
        # neg_seq_distance1 = (1 - torch.matmul(
        #     F.normalize(seq_output1),
        #     F.normalize(seq_output1).transpose(0, 1),
        # )) * mask
        # neg_seq_distance2 = (1 - torch.matmul(
        #     F.normalize(seq_output1),
        #     F.normalize(seq_output2).transpose(0, 1),
        # )) * mask

        # neg_loss1 = self.cor_loss_fct(neg_seq_distance1, user_distance) * 2
        # neg_loss2 = self.cor_loss_fct(neg_seq_distance2, user_distance) * 2

        # return loss + neg_loss1 + neg_loss2
