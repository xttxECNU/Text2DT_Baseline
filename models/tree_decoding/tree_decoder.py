import logging

import torch
import torch.nn as nn

from models.embedding_models.chinese_bert_embedding_model import ChineseEmbedModel
from modules.token_embedders.bert_encoder import BertLinear

logger = logging.getLogger(__name__)

#整个模型
class TreeJointDecoder(nn.Module):
    def __init__(self, cfg):

        super().__init__()
        self.activation = nn.GELU()
        self.device = cfg.device

        self.embedding_model_chinese = ChineseEmbedModel(cfg)
        self.encoder_output_size = self.embedding_model_chinese.get_hidden_size()

        self.head_mlp = BertLinear(input_size=self.encoder_output_size,
                                   output_size=cfg.mlp_hidden_size,
                                   activation=self.activation,
                                   dropout=cfg.dropout)
        self.tail_mlp = BertLinear(input_size=self.encoder_output_size,
                                   output_size=cfg.mlp_hidden_size,
                                   activation=self.activation,
                                   dropout=cfg.dropout)

        self.U = nn.Parameter(
            torch.FloatTensor(4, cfg.mlp_hidden_size + 1,
                              cfg.mlp_hidden_size + 1))
        self.U.data.zero_()

        if cfg.logit_dropout > 0:
            self.logit_dropout = nn.Dropout(p=cfg.logit_dropout)
        else:
            self.logit_dropout = lambda x: x


        self.none_idx = 0
        self.element_loss = nn.CrossEntropyLoss()

    def forward(self, batch_inputs):
        """forward

        Arguments:
            batch_inputs {dict} -- batch input data

        Returns:
            dict -- results: ent_loss, ent_pred
        """

        results = {}

        self.embedding_model_chinese(batch_inputs)
        batch_seq_tokens_encoder_repr = batch_inputs['seq_encoder_reprs']


        batch_seq_tokens_head_repr = self.head_mlp(batch_seq_tokens_encoder_repr)
        batch_seq_tokens_head_repr = torch.cat(
            [batch_seq_tokens_head_repr,
             torch.ones_like(batch_seq_tokens_head_repr[..., :1])], dim=-1)
        batch_seq_tokens_tail_repr = self.tail_mlp(batch_seq_tokens_encoder_repr)
        batch_seq_tokens_tail_repr = torch.cat(
            [batch_seq_tokens_tail_repr,
             torch.ones_like(batch_seq_tokens_tail_repr[..., :1])], dim=-1)

        batch_joint_score = torch.einsum('bxi, oij, byj -> boxy', batch_seq_tokens_head_repr, self.U,
                                         batch_seq_tokens_tail_repr).permute(0, 2, 3, 1)

        batch_normalized_joint_score = torch.softmax(
            batch_joint_score, dim=-1) * batch_inputs['label_matrix_mask'].unsqueeze(-1).float()

        results['pred_label_matrix'] = torch.argmax(batch_normalized_joint_score, dim=-1)
        if self.training:
            results['loss'] = self.element_loss(
                self.logit_dropout(batch_joint_score[batch_inputs['label_matrix_mask']]),
                batch_inputs['label_matrix'][batch_inputs['label_matrix_mask']])
            results['gold_label']= batch_inputs['label_matrix'][batch_inputs['label_matrix_mask']].tolist()
        results['probability_matrix'] = batch_normalized_joint_score[batch_inputs['label_matrix_mask']].tolist()
        return results

