import json
import os
import random
import logging

import torch
import numpy as np
from transformers import AdamW

from utils.argparse import ConfigurationParer
from utils.tree_decoder import TreeSoftDecoder
from utils.nn_utils import get_n_trainable_parameters
from utils.eval import TreeStructureEval
from inputs import data_loader
from models.tree_decoding.tree_decoder import TreeJointDecoder

logger = logging.getLogger(__name__)

def step(model, batch_inputs, device, is_test=False):

    batch_inputs["tokens"] = torch.LongTensor(batch_inputs["tokens"])
    batch_inputs["label_matrix_mask"] = torch.BoolTensor(batch_inputs["label_matrix_mask"])
    if not is_test:
        batch_inputs["label_matrix"] = torch.LongTensor(batch_inputs["label_matrix"])

    if device > -1:
        batch_inputs["tokens"] = batch_inputs["tokens"].cuda(device=device, non_blocking=True)
        if not is_test:
            batch_inputs["label_matrix"] = batch_inputs["label_matrix"].cuda(device=device, non_blocking=True)
        batch_inputs["label_matrix_mask"] = batch_inputs["label_matrix_mask"].cuda(device=device,
                                                                                               non_blocking=True)

    outputs = model(batch_inputs)
    if is_test:
        sent_output = dict()
        sent_output['pred_label_matrix'] = outputs['pred_label_matrix'][batch_inputs["label_matrix_mask"]].cpu().numpy()
        return sent_output['pred_label_matrix'].tolist(),batch_inputs["tail_entitys_to_index"],outputs['probability_matrix']
    if not model.training:
        correct_label, total_label = 0, 0
        sent_output = dict()
        sent_output['label_matrix'] = batch_inputs['label_matrix'][batch_inputs["label_matrix_mask"]].cpu().numpy()
        sent_output['pred_label_matrix'] = outputs['pred_label_matrix'][batch_inputs["label_matrix_mask"]].cpu().numpy()
        for i in range(len(sent_output['label_matrix'])):
            if sent_output['label_matrix'].tolist()[i]==sent_output['pred_label_matrix'].tolist()[i]:
                correct_label += 1
        total_label= len(sent_output['label_matrix'])

        return correct_label, total_label, sent_output['pred_label_matrix'].tolist(),batch_inputs["tail_entitys_to_index"],outputs['probability_matrix']

    return outputs['loss']

def train(cfg, dataset, dataset_dev,model):
    logger.info("Training starting...")

    for name, param in model.named_parameters():
        logger.info("{!r}: size: {} requires_grad: {}.".format(name, param.size(), param.requires_grad))

    logger.info("Trainable parameters size: {}.".format(get_n_trainable_parameters(model)))

    parameters = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    bert_layer_lr = {}
    base_lr = cfg.bert_learning_rate
    for i in range(11, -1, -1):
        bert_layer_lr['.' + str(i) + '.'] = base_lr
        base_lr *= cfg.lr_decay_rate

    optimizer_grouped_parameters = []
    for name, param in parameters:

        params = {'params': [param], 'lr': cfg.learning_rate}
        if any(item in name for item in no_decay):
            params['weight_decay_rate'] = 0.0
        else:
            if 'bert' in name:
                params['weight_decay_rate'] = cfg.adam_bert_weight_decay_rate
            else:
                params['weight_decay_rate'] = cfg.adam_weight_decay_rate

        for bert_layer_name, lr in bert_layer_lr.items():
            if bert_layer_name in name:
                params['lr'] = lr
                break

        optimizer_grouped_parameters.append(params)

    optimizer = AdamW(optimizer_grouped_parameters,
                      betas=(cfg.adam_beta1, cfg.adam_beta2),
                      lr=cfg.learning_rate,
                      eps=cfg.adam_epsilon,
                      weight_decay=cfg.adam_weight_decay_rate,
                      correct_bias=False)
    tree_dev = json.load(open(cfg.dev_file))

    model.zero_grad()
    model.train()

    global_step = 0
    loss = 0.0
    best_acc = 0.0

    for epoch in range(cfg.epochs):
        train_data_prefetcher = data_loader.DataPreFetcher(dataset)
        data = train_data_prefetcher.next()
        while data is not None:
            loss_batch = step(model, data, cfg.device)

            optimizer.zero_grad()
            loss_batch.backward()
            optimizer.step()

            global_step += 1
            loss += loss_batch.item()

            if global_step % cfg.logging_steps == 0:
                logger.info("Epoch: {} Batch: {} Loss: {} ".format(
                        epoch, global_step, loss))
                loss = 0
            data = train_data_prefetcher.next()

        if (epoch + 1) % cfg.pretrain_epochs == 0:

            correct_label, total_label, pred_result = dev(cfg, dataset_dev, model)
            tree_soft_decoder = TreeSoftDecoder(pred_result, cfg.node_separate_threshold)
            trees = tree_soft_decoder.softdecoder()
            tree_eval = TreeStructureEval(trees,tree_dev)
            tree_acc, triplet_f1, path_f1, tree_edit_distance, node_f1 = tree_eval.tree_structure_eval()

            logger.info("Epoch: {}, Acc: {}, Tree_Acc: {}, Triplet_F1: {}, Path_F1: {}, Tree_EditDistance: {}, Node_F1: {}.".
                         format(epoch, correct_label/total_label,tree_acc, triplet_f1, path_f1, tree_edit_distance, node_f1))
            model.train()

            if correct_label/total_label > best_acc:
                best_acc = correct_label/total_label
                logger.info("Save model... , Best_Acc: {}".format(best_acc))
                torch.save(model.state_dict(), open(cfg.best_model_path, "wb"))

        # manually release the unused cache
        torch.cuda.empty_cache()

    logger.info("finish training")

def dev(cfg, dataset, model):
    logger.info("Validate starting...")
    model.zero_grad()

    dev_data_prefetcher = data_loader.DataPreFetcher(dataset)
    data = dev_data_prefetcher.next()
    model.eval()

    correct_label, total_label = 0, 0
    result = {}
    result['label'] = []
    result['tail_entitys_to_index'] = []
    result['martix'] = []

    while data is not None:
        batch_correct, batch_total, label, tail_entitys_to_index, martix= step(model, data, cfg.device)
        correct_label = correct_label + batch_correct
        total_label = total_label + batch_total
        result['label'].append(label)
        result['tail_entitys_to_index'].append(tail_entitys_to_index)
        result['martix'].append(martix)
        data = dev_data_prefetcher.next()

    return correct_label, total_label, result

def test(cfg, dataset, model):
    logger.info("Testing starting...")
    model.zero_grad()
    model.eval()

    test_data_prefetcher = data_loader.DataPreFetcher(dataset)
    data = test_data_prefetcher.next()

    result = {}
    result['label'] = []
    result['tail_entitys_to_index'] = []
    result['martix'] = []

    while data is not None:
        label,tail_entitys_to_index,martix = step(model, data, cfg.device, is_test=True)
        result['label'].append(label)
        result['tail_entitys_to_index'].append(tail_entitys_to_index)
        result['martix'].append(martix)
        data = test_data_prefetcher.next()

    tree_soft_decoder = TreeSoftDecoder(result, cfg.node_separate_threshold)
    trees = tree_soft_decoder.softdecoder()


    with open('Text2DT_TreeDecoder_test_result.json', 'w', encoding='utf-8') as f:
        json.dump(trees, f, ensure_ascii=False)

    logger.info("Finishing Testing")

def main():
    # config settings
    parser = ConfigurationParer()
    parser.add_save_cfgs()
    parser.add_data_cfgs()
    parser.add_model_cfgs()
    parser.add_optimizer_cfgs()
    parser.add_run_cfgs()

    cfg = parser.parse_args()
    logger.info(parser.format_values())

    # set random seed
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    if cfg.device > -1 and not torch.cuda.is_available():
        logger.error('config conflicts: no gpu available, use cpu for training.')
        cfg.device = -1
    if cfg.device > -1:
        torch.cuda.manual_seed(cfg.seed)

    model = TreeJointDecoder(cfg=cfg)
    if cfg.test and os.path.exists(cfg.best_model_path):
        state_dict = torch.load(open(cfg.best_model_path, 'rb'), map_location=lambda storage, loc: storage)
        model.load_state_dict(state_dict)
        logger.info("Loading best training model {} successfully for testing.".format(cfg.best_model_path))

    if cfg.device > -1:
        model.cuda(device=cfg.device)

    if cfg.test:
        test_data_loader = data_loader.get_loader(cfg, cfg.test_file, is_test=cfg.test)
        test(cfg, test_data_loader, model)
    else:
        train_data_loader = data_loader.get_loader(cfg, cfg.train_file)
        dev_data_loader = data_loader.get_loader(cfg, cfg.dev_file, is_dev=True)
        train(cfg, train_data_loader, dev_data_loader,model)

if __name__ == '__main__':
    main()
