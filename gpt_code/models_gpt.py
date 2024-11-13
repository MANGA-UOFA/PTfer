import os
import json
import logging
import random

from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
from transformers import AutoConfig

import torch
import torch.nn.functional as F
import numpy as np

logger = logging.getLogger(__name__)

class Prober():

    def __init__(self, args):
        super().__init__()

        self._model_device = 'cpu'

        model_name = args.model_name
        vocab_name = model_name

        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        if torch.cuda.device_count() > 1:
            torch.cuda.manual_seed_all(args.seed)

        config = AutoConfig.from_pretrained(model_name)
        if isinstance(config, GPT2Config):
            self.model_type = 'gpt2'
            self.tokenizer = GPT2Tokenizer.from_pretrained(vocab_name)
            self.lm_model = GPT2LMHeadModel.from_pretrained(model_name)
            self.base_model = self.lm_model.transformer
        else:
            raise ValueError('Model %s not supported yet!'%(model_name))

        self.lm_model.eval()

        # original vocab
        self.map_indices = None
        self.vocab = list(self.tokenizer.get_vocab().keys())
        logger.info('Vocab size: %d'%len(self.vocab))
        self._init_inverse_vocab()

        self.EOS = self.tokenizer.eos_token
        self.BOS = self.tokenizer.bos_token
        self.pad_id = self.inverse_vocab[self.tokenizer.eos_token]
        
        # print(self.EOS, self.BOS, self.pad_id)

        # used to output top-k predictions
        self.k = args.k
    
    def _init_inverse_vocab(self):
        self.inverse_vocab = {w: i for i, w in enumerate(self.vocab)}

    def init_indices_for_filter_logprobs(self, vocab_subset, logger=None):
        index_list = []
        new_vocab_subset = []
        for word in vocab_subset:
            tokens = self.tokenizer.tokenize(' '+word)
            if len(tokens) == 1:
                index_list.append(self.tokenizer.convert_tokens_to_ids(tokens)[0])
                new_vocab_subset.append(word)
            else:
                msg = "word {} from vocab_subset not in model vocabulary!".format(word)
                if logger is not None:
                    logger.warning(msg)
                else:
                    print("WARNING: {}".format(msg))

        indices = torch.as_tensor(index_list)
        return indices, index_list
    
    def _cuda(self):
        self.lm_model.cuda()
        
    def try_cuda(self):
        """Move model to GPU if one is available."""
        if torch.cuda.is_available():
            if self._model_device != 'cuda':
                logger.info('Moving model to CUDA')
                self._cuda()
                self._model_device = 'cuda'
        else:
            logger.info('No CUDA found')

    """
    Below is adapted for gpt2
    """
    def _get_input_tensors_batch(self, sentences, samples_list, training):
        max_len = 0
        input_ids_list = []
        labels_list = []
        target_indices_list = []
        label_ids = []

        for sentence, sample in zip(sentences, samples_list):
            if len(sentence) > 1:
                logger.info(sentences)
                raise ValueError("GPT accepts maximum 1 sentences in input for each data point")
            
            target = sample['obj_label']
            tmp_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(' ' + target))
            assert(len(tmp_ids) == 1)
            label_ids.append(tmp_ids[0])
        
            # tokenized_text = [self.tokenizer.tokenize(token) if (not token.startswith('[unused')) else [token] for token in sentence[0].split()]
            # tokenized_text = [item for sublist in tokenized_text for item in sublist]
            tokenized_text = self.tokenizer.tokenize(sentence[0])
            
            indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
            # indexed_tokens = indexed_tokens + [tmp_ids[0]]
            if training:
                indexed_tokens = indexed_tokens + [tmp_ids[0]]
            input_ids = torch.tensor([indexed_tokens])
            # input_ids = self.tokenizer.encode(sentence, return_tensors='pt')

            max_len = max(max_len, input_ids.shape[1])

            # Create labels tensor
            labels = torch.full(input_ids.shape, -100)
            labels[0, -1] = tmp_ids[0]

            input_ids_list.append(input_ids)
            labels_list.append(labels)
            
            target_index = (labels[0] == tmp_ids[0]).nonzero(as_tuple=True)
            target_indices_list.append(target_index)
       

        # Pad sequences to have the same length
        padded_input_ids = torch.full((len(sentences), max_len), self.pad_id)
        padded_labels = torch.full((len(sentences), max_len), -100)

        for i, (input_ids, labels) in enumerate(zip(input_ids_list, labels_list)):
            padded_input_ids[i, :input_ids.shape[1]] = input_ids
            padded_labels[i, :labels.shape[1]] = labels

        return padded_input_ids, padded_labels, target_indices_list, label_ids

    def run_batch(self, sentences_list, samples_list, try_cuda=True, training=True, filter_indices=None, index_list=None, vocab_to_common_vocab=None):
        if try_cuda and torch.cuda.is_available():
            self.try_cuda()

        input_ids, labels, target_indices_list, label_ids = self._get_input_tensors_batch(sentences_list, samples_list, training=training)
        # print(input_ids, labels, target_indices_list, label_ids)

        if training:
            self.lm_model.train()
            outputs = self.lm_model(
                input_ids=input_ids.to(self._model_device),
                labels=labels.to(self._model_device),
                attention_mask=(input_ids != self.pad_id).to(self._model_device)
            )
            loss = outputs.loss
        else:
            self.lm_model.eval()
            with torch.no_grad():
                outputs = self.lm_model(
                    input_ids=input_ids.to(self._model_device),
                    # labels=labels.to(self._model_device),
                    attention_mask=(input_ids != self.pad_id).to(self._model_device)
                )
            logits = outputs.logits
            log_probs = F.log_softmax(logits, dim=-1).cpu()
            # loss = outputs.loss
            loss = torch.tensor([0])
        
        
        if training:
            return loss
        else:
            tot = log_probs.shape[0]
            cor = 0
            preds = []
            topk = []
            common_vocab_loss = []

            for i in range(log_probs.shape[0]):
                masked_index = target_indices_list[i][0]
                log_prob = log_probs[i][masked_index].squeeze()
                this_label_id = label_ids[i]
                if filter_indices is not None:
                    log_prob = log_prob.index_select(dim=0, index=filter_indices)
                    pred_common_vocab = torch.argmax(log_prob)
                    pred = index_list[pred_common_vocab]
                    
                    topk_preds = []
                    topk_log_prob, topk_ids = torch.topk(log_prob, self.k)
                    for log_prob_i, idx in zip(topk_log_prob, topk_ids):
                        ori_idx = index_list[idx]
                        token = self.vocab[ori_idx]
                        topk_preds.append({'token': token, 'log_prob': log_prob_i.item()})
                    topk.append(topk_preds)
                    
                    common_logits = logits[i][masked_index].squeeze().cpu().index_select(dim=0, index=filter_indices)
                    common_log_prob = -F.log_softmax(common_logits, dim=-1)
                    common_label_id = vocab_to_common_vocab[this_label_id]
                    common_vocab_loss.append(common_log_prob[common_label_id].item())
                else:
                    pred = torch.argmax(log_prob)
                    topk.append([])
                if pred == labels[i][masked_index]:
                    cor += 1
                    preds.append(1)
                else:
                    preds.append(0)
                       
            return log_probs, cor, tot, preds, topk, loss, common_vocab_loss 
        