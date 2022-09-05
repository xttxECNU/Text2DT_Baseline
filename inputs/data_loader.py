import sys

from torch.utils.data import DataLoader, Dataset
import json
import torch
import numpy as np
from transformers import *

def nodematrix(tree):
    nodelist=[]
    for i in range(len(tree)):
        nodelist.append(tree[i]["role"])
    #print("nodelist", nodelist)
    node_matrix = [[0 for i in range(len(nodelist))] for j in range(len(nodelist))]
    while (nodelist[0] != 'D'):
        for i in range(len(nodelist)):
            flag, leaf1, leaf2 = 0, 0, 0
            for j in range(i+1,len(nodelist)):
                if nodelist[j]=='D' and flag==0:
                    flag = 1
                    leaf1 = j
                elif nodelist[j]=='X':
                    continue
                elif nodelist[j]=='D' and flag==1:
                    #print(i)
                    leaf2 = j
                    nodelist[i]='D'
                    node_matrix[leaf1][i]='F'
                    node_matrix[leaf2][i] = 'F'
                    node_matrix[i][leaf1] = 'L'
                    node_matrix[i][leaf2] = 'R'
                    for k in range(i+1,leaf2+1):
                        nodelist[k]='X'
                    #print((nodelist))
                    break
                elif nodelist[j]=='C':
                    break

    return(node_matrix)

class Text2DTDataset(Dataset):
    def __init__(self, config, file_path, is_test):
        self.config = config
        self.file_path = file_path
        self.is_test = is_test
        self.tokenizer = BertTokenizer.from_pretrained(self.config.bert_model_name)
        self.tokenizer.add_special_tokens({'additional_special_tokens': ['[unused1]', '[unused2]']})
        self.json_data = json.load(open(file_path))
        self.rel2id = json.load(open(self.config.label_file))['label2id']

    def __len__(self):
        return len(self.json_data)

    def __getitem__(self, idx):
        # 由于绝大多数三元组的尾实体不相同，因此这里用尾实体来代替整个三元组拼接在句子后以控制句子长度以及避免头实体的大量重复
        ins_json_data = self.json_data[idx]
        text = ins_json_data['text']
        tail_entitys=[]
        entity_index = []
        tail_entitys_to_index = []

        if not self.is_test:

            triple_relation_in_tree = []
            triple_relation_table = []
            nodes = ins_json_data['tree']
            node_matrix = nodematrix(nodes)

            #获得三元组在树中的关系
            for i in range(len(node_matrix)):
                for j in range(len(node_matrix)):
                    node_relation = node_matrix[i][j]
                    if node_relation != 0:
                        for head_triple in nodes[i]['triples']:
                            for tail_triple in nodes[j]['triples']:
                                triple_relation_in_tree.append([tail_triple,node_relation,head_triple])

            #获取尾实体在句子中的索引，按照索引顺序将代表三元组的尾实体拼接在句子后
            for triple in triple_relation_in_tree:
                if triple[0][2] not in tail_entitys:
                    entity_index_in_text = text.find(triple[0][2])
                    if entity_index_in_text != -1:
                        tail_entitys_to_index.append([triple[0][2],entity_index_in_text, triple[0]])
                        tail_entitys.append(triple[0][2])
                if triple[2][2] not in tail_entitys:
                    entity_index_in_text = text.find(triple[2][2])
                    if entity_index_in_text != -1:
                        tail_entitys_to_index.append([triple[2][2], entity_index_in_text, triple[2]])
                        tail_entitys.append(triple[2][2])
            tail_entitys_to_index.sort(key=lambda x: x[1])

            for i in range(len(tail_entitys)):
                tail_entitys[i]=tail_entitys_to_index[i][0]

            for entity in tail_entitys_to_index:
                text=text+'[unused1]'+entity[0]+'[unused2]'

            text = ' '.join(text.split()[:self.config.max_sent_len])

            token_ids, masks = self.tokenizer.encode_plus(text)['input_ids'], self.tokenizer.encode_plus(text)['attention_mask']
            text_len= len(token_ids)
            token_ids = np.array(token_ids)

            for i in range(len(token_ids)):
                if token_ids[i] == self.tokenizer.convert_tokens_to_ids('[unused1]') :
                    entity_index.append(i)

            for i in range(len(tail_entitys)):
                tmp = [3]*len(tail_entitys)
                for j in range(len(tail_entitys)):
                    for triple_relation in triple_relation_in_tree:
                        if triple_relation[0][2] == tail_entitys[i] and triple_relation[2][2] == tail_entitys[j]:
                            tmp[j]=self.rel2id[triple_relation[1]]
                            break
                triple_relation_table.append(tmp)

            label_matrix=[[0] * text_len for i in range(text_len)]
            label_matrix_mask = [[False] * text_len for i in range(text_len)]

            for i in range(len(tail_entitys)):
                for j in range(len(tail_entitys)):
                    label_matrix_mask[entity_index[i]][entity_index[j]]=True
                    label_matrix[entity_index[i]][entity_index[j]] = triple_relation_table[i][j]

            return token_ids, masks, text_len, entity_index, label_matrix, label_matrix_mask ,tail_entitys_to_index
        else:
            for triple in ins_json_data['triple_list']:
                if triple[2] not in tail_entitys:
                    entity_index_in_text = text.find(triple[2])
                    if entity_index_in_text != -1:
                        tail_entitys_to_index.append([triple[2], entity_index_in_text,triple])
                        tail_entitys.append(triple[2])
            tail_entitys_to_index.sort(key=lambda x: x[1])

            for entity in tail_entitys_to_index:
                text=text+'[unused1]'+entity[0]+'[unused2]'

            text = ' '.join(text.split()[:self.config.max_sent_len])

            token_ids, masks = self.tokenizer.encode_plus(text)['input_ids'], self.tokenizer.encode_plus(text)[
                'attention_mask']
            text_len = len(token_ids)
            token_ids = np.array(token_ids)

            for i in range(len(token_ids)):
                if token_ids[i] == self.tokenizer.convert_tokens_to_ids('[unused1]'):
                    entity_index.append(i)
            label_matrix_mask = [[False] * text_len for i in range(text_len)]

            for i in range(len(tail_entitys)):
                for j in range(len(tail_entitys)):
                    label_matrix_mask[entity_index[i]][entity_index[j]] = True

            return token_ids, masks, text_len, entity_index, label_matrix_mask, tail_entitys_to_index
        #test:token_ids, masks, text_len, entity_index,entitys

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    batch.sort(key=lambda x: x[2], reverse=True)
    token_ids, masks, text_len, entity_index, label_matrix, label_matrix_mask, tail_entitys_to_index = zip(*batch)
    cur_batch = len(batch)
    max_text_len = max(text_len)
    batch_token_ids =[[0] * max_text_len for i in range(cur_batch)]
    batch_masks = [[0] * max_text_len for i in range(cur_batch)]
    batch_label_matrix =[[[0] * max_text_len for i in range(max_text_len)] for j in range(cur_batch)]
    batch_label_matrix_mask = [[[False] * max_text_len for i in range(max_text_len)] for j in range(cur_batch)]


    for i in range(cur_batch):

        batch_token_ids[i][:text_len[i]] = token_ids[i]
        batch_masks[i][:text_len[i]] = masks[i]
        for j in range(text_len[i]):
            batch_label_matrix[i][j][:text_len[i]] = label_matrix[i][j]
            batch_label_matrix_mask[i][j][:text_len[i]] = label_matrix_mask[i][j]


    return {'tokens': batch_token_ids,
            'tokens_lens':text_len,
            'mask': batch_masks,
            'entity_index': entity_index,
            'tail_entitys_to_index': tail_entitys_to_index,
            'label_matrix':batch_label_matrix,
            'label_matrix_mask':batch_label_matrix_mask}

def collate_fn_test(batch):
    batch = list(filter(lambda x: x is not None, batch))
    batch.sort(key=lambda x: x[2], reverse=True)
    token_ids, masks, text_len, entity_index, label_matrix_mask, tail_entitys_to_index = zip(*batch)
    cur_batch = len(batch)
    max_text_len = max(text_len)
    batch_token_ids =[[0] * max_text_len for i in range(cur_batch)]
    batch_masks = [[0] * max_text_len for i in range(cur_batch)]
    batch_label_matrix_mask = [[[False] * max_text_len for i in range(max_text_len)] for j in range(cur_batch)]

    for i in range(cur_batch):
        batch_token_ids[i][:text_len[i]] = token_ids[i]
        batch_masks[i][:text_len[i]] = masks[i]
        for j in range(text_len[i]):
            batch_label_matrix_mask[i][j][:text_len[i]] = label_matrix_mask[i][j]

    return {'tokens': batch_token_ids,
            'tokens_lens':text_len,
            'mask': batch_masks,
            'entity_index': entity_index,
            'tail_entitys_to_index': tail_entitys_to_index,
            'label_matrix_mask':batch_label_matrix_mask
            }

def get_loader(config, file_path, is_test=False, is_dev=False, num_workers=0):
    dataset = Text2DTDataset(config, file_path, is_test)
    if is_test:
        data_loader = DataLoader(dataset=dataset,
                                 batch_size=1,
                                 shuffle=False,
                                 pin_memory=True,
                                 num_workers=num_workers,
                                 collate_fn=collate_fn_test)
    elif is_dev:
        data_loader = DataLoader(dataset=dataset,
                                 batch_size=1,
                                 shuffle=False,
                                 pin_memory=True,
                                 num_workers=num_workers,
                                 collate_fn=collate_fn)
    else:
        data_loader = DataLoader(dataset=dataset,
                                 batch_size=config.train_batch_size,
                                 shuffle=True,
                                 pin_memory=True,
                                 num_workers=num_workers,
                                 collate_fn=collate_fn)
    return data_loader

class DataPreFetcher(object):
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_data = next(self.loader)
        except StopIteration:
            self.next_data = None
            return
        with torch.cuda.stream(self.stream):
            for k, v in self.next_data.items():
                if isinstance(v, torch.Tensor):
                    self.next_data[k] = self.next_data[k].cuda(non_blocking=True)
    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        data = self.next_data
        self.preload()
        return data
