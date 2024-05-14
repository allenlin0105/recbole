import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from accelerate import PartialState
from tqdm import trange

from recbole.model.sequential_recommender.sasrec import SASRec


class LLMRec(SASRec):
    def __init__(self, config, dataset):
        super(LLMRec, self).__init__(config, dataset)

        # load parameters info
        self.device = config["gpu_id"]
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

        # load llm
        # MODELPATH = "meta-llama/Llama-2-7b-chat-hf"
        MODELPATH = "meta-llama/Meta-Llama-3-8B-Instruct"

        self.llm = AutoModelForCausalLM.from_pretrained(
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

        self.tokenizer = AutoTokenizer.from_pretrained(
            MODELPATH,
            trust_remote_code=True,
            padding_side="left",
        )      
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self._load_interest_embed(dataset)
        self._load_item_embed(dataset)

        # load augmented sequence
        # with open(Path(dataset.dataset_path, "user_id2token_id.json"), "r") as fp:
        #     user_id2token_id = json.load(fp)
        # with open(Path(dataset.dataset_path, "item_id2token_id.json"), "r") as fp:
        #     item_id2token_id = json.load(fp)

        # self.user_id2aug_sequence = {}
        # for seq_file in Path(dataset.dataset_path, "llama_prob_v1").iterdir():
        #     if seq_file.suffix == ".log":
        #         continue

        #     user_id = seq_file.name.removesuffix(".txt")
        #     if user_id not in user_id2token_id:
        #         continue
            
        #     user_id = user_id2token_id[user_id]
        #     sequence = []
        #     with open(seq_file, "r") as fp:
        #         for i, line in enumerate(fp):
        #             if i % 2 == 0:
        #                 item_token = item_id2token_id[line.strip()]
        #             else:
        #                 try:
        #                     prob = float(line)
        #                     sequence.append((item_token, prob))
        #                 except:
        #                     pass
        #     self.user_id2aug_sequence[user_id] = sorted(sequence, key=lambda x: x[1], reverse=True)

    def _load_interest_embed(self, dataset):
        user_id2token_id = dataset.field2token_id["user_id"]

        interests = [None] * len(user_id2token_id)
        for interest_file in Path("../template-testing/result", dataset.dataset_name, "interest/llama3_v1").iterdir():
            with open(interest_file, "r") as fp:
                token_id = int(user_id2token_id[interest_file.stem])
                interests[token_id] = np.array([line.strip() for line in fp])

        self.interest_embed = torch.zeros((len(interests, 4096)))
        n_batch = 1
        for i in trange(1, len(interests) + 1, n_batch):
            batch = min(n_batch, len(interests) - i)
            texts = np.concatenate(interests[i:i+batch]).tolist()

            self.interest_embed[i:i+batch] = torch.sum(
                self._encode_texts(texts).view(batch, 5, -1),
                axis=1
            )

    def _load_item_embed(self, dataset):
        item_id2token_id = dataset.field2token_id["item_id"]

        item_names = [None] * len(item_id2token_id)
        with open(Path(dataset.dataset_path, f"{dataset.dataset_name}.item"), "r") as fp:
            reader = csv.reader(fp, delimiter="\t")
            next(reader)

            for row in reader:
                item_id, item_name = row[0], row[1]
                if item_id not in item_id2token_id:
                    continue
                token_id = int(item_id2token_id[item_id])
                item_names[token_id] = item_name
        
        self.item_embed = torch.zeros((len(item_names), 4096))
        n_batch = 1
        for i in trange(1, len(item_names), n_batch):
            batch = min(n_batch, len(item_names) - i)
            self.item_embed[i:i+batch] = self._encode_texts(item_names[i:i+batch])

    def _encode_texts(self, texts):
        t_input = self.tokenizer(texts, padding=True, return_tensors="pt").to(self.device)
        attention_mask = t_input.attention_mask
        with torch.no_grad():
            last_hidden_state = self.llm(**t_input, output_hidden_states=True).hidden_states[-1]

        sum_embeddings = torch.sum(last_hidden_state * attention_mask.unsqueeze(-1), dim=1)
        num_of_none_padding_tokens = torch.sum(attention_mask, dim=-1).unsqueeze(-1)
        sentence_embeddings = sum_embeddings / num_of_none_padding_tokens
        return sentence_embeddings.to("cpu")

    def mask_correlated_samples(self, batch_size):
        N = batch_size
        mask = torch.ones((2 * N, 2 * N)).bool()
        mask = mask.fill_diagonal_(0)
        mask *= ~ torch.diagflat(torch.ones(N), offset=N).bool()
        mask *= ~ torch.diagflat(torch.ones(N), offset=-N).bool()
        return mask

    def calculate_loss(self, interaction):
        loss = super().calculate_loss(interaction)
        cor_loss = self.calculate_cor_loss(interaction)
        return loss, self.cl_lambda * cor_loss

        # if cl_loss != None:
        #     return loss, self.cl_lambda * cl_loss
        # else:
        #     return loss
    
    def calculate_cor_loss(self, interaction):
        aug_item_seq1, aug_len1, aug_item_seq2, aug_len2 = \
            interaction['aug1'], interaction['aug_len1'], interaction['aug2'], interaction['aug_len2']
        seq_output1 = self.forward(aug_item_seq1, aug_len1)
        seq_output2 = self.forward(aug_item_seq2, aug_len2)

        user_ids = interaction[self.USER_ID].detach().cpu().tolist()
        user_embed = self.interest_embed[user_ids].unsqueeze(2)

        aug_embed_seq1 = torch.stack([self.item_embed[seq] for seq in aug_item_seq1.detach().cpu().tolist()])
        r1 = torch.sum(F.normalize(
            torch.bmm(aug_embed_seq1, user_embed).squeeze(2),
            dim=(0, 1),
        ), dim=1)

        aug_embed_seq2 = torch.stack([self.item_embed[seq] for seq in aug_item_seq2.detach().cpu().tolist()])
        r2 = torch.sum(F.normalize(
            torch.bmm(aug_embed_seq2, user_embed).squeeze(2),
            dim=(0, 1),
        ), dim=1)

        # org_seq, org_len = interaction[self.ITEM_SEQ], interaction[self.ITEM_SEQ_LEN]
        # seq_output1 = self.forward(org_seq, org_len)

        # pad_to_len = interaction[self.ITEM_SEQ].shape[1]

        # item_seq = []
        # score_seq = []
        # item_seq_len = []

        # user_ids = interaction[self.USER_ID].detach().cpu()
        # for user_id in user_ids:
        #     if user_id not in self.user_id2aug_sequence:
        #         continue
        #     aug_sequence = self.user_id2aug_sequence[user_id.item()]
        #     if len(aug_sequence) > pad_to_len:
        #         aug_sequence = aug_sequence[:pad_to_len]

        #     item_seq.append([aug[0] for aug in aug_sequence] + [0] * (pad_to_len - len(aug_sequence)))
        #     score_seq.append(np.mean([aug[1] for aug in aug_sequence]))
        #     item_seq_len.append(len(aug_sequence))

        # if len(item_seq_len) == 0:
        #     return None

        # device = interaction[self.ITEM_SEQ].get_device()
        # seq_output2 = self.forward(
        #     torch.tensor(item_seq).to(device), 
        #     torch.tensor(item_seq_len).to(device)
        # )

        return self.pairwise_loss(seq_output1, seq_output2, r1, r2)

    def pairwise_loss(self, z_i, z_j, r_i, r_j):
        cur_batch_size = z_i.size(0)

        if self.similarity_type == 'cos':
            sim = self.sim(z_i.unsqueeze(1), z_j.unsqueeze(0), dim=2) / self.tau
        elif self.similarity_type == 'dot':
            sim = self.sim(z_i, z_j.T) / self.tau
        pair_sim = torch.diag(sim)

        r_deficit = torch.abs(r_i - r_j).to(pair_sim.device) 

        x_def = pair_sim - torch.mean(pair_sim) + 1e-7
        y_def = r_deficit - torch.mean(r_deficit) + 1e-7
        loss = torch.sum(x_def * y_def) / torch.sqrt(torch.sum(x_def.pow(2)) * torch.sum(y_def.pow(2)))
        return loss

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        test_items_emb = self.item_embedding.weight[:self.n_items]
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B n_items]
        return scores