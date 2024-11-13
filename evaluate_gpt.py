import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'gpt_code'))

import torch
from gpt_code.models_gpt import Prober
from gpt_code.utils import evaluate, load_data, batchify, load_vocab, load_file

MAX_NUM_VECTORS = 10

def get_relation_meta(args):
    relations = load_file(args.relation_profile)
    for relation in relations:
        if relation['relation'] == args.relation:
            return relation
    raise ValueError('Relation info %s not found in file %s'%(args.relation, args.relation_profile))

def convert_manual_to_dense(manual_template, model):
    def assign_embedding(new_token, token):
        """
        assign the embedding of token to new_token
        """
        id_a = model.tokenizer.convert_tokens_to_ids([new_token])[0]
        id_b = model.tokenizer.convert_tokens_to_ids([token])[0]
        with torch.no_grad():
            model.base_model.wte.weight[id_a] = model.base_model.wte.weight[id_b].detach().clone()

    new_token_id = 0
    template = []
    for word in manual_template.split():
        if word in ['[X]', '[Y]']:
            template.append(word)
        else:
            tokens = model.tokenizer.tokenize(' ' + word)
            for token in tokens:
                new_token_id += 1
                template.append(get_new_token(new_token_id))
                assign_embedding(get_new_token(new_token_id), token)

    return ' '.join(template)

def get_new_token(vid):
    assert(vid > 0 and vid <= MAX_NUM_VECTORS)
    return '[V%d]'%(vid)

def prepare_for_dense_prompt(model):
    new_tokens = [get_new_token(i+1) for i in range(MAX_NUM_VECTORS)]
    model.tokenizer.add_tokens(new_tokens)
    model.lm_model.resize_token_embeddings(len(model.tokenizer))

def init_template(args, model):
    if args.init_manual_template:
        relation = get_relation_meta(args)
        # print(relation)
        template = convert_manual_to_dense(relation['template'], model)
        template = template.split('[Y]')[0]
    else:
        template = '[X] ' + ' '.join(['[V%d]'%(i+1) for i in range(args.num_vectors)])
    return template

def load_optiprompt(model, original_vocab_size, vs):
    # copy fine-tuned new_tokens to the pre-trained model
    with torch.no_grad():
        model.base_model.wte.weight[original_vocab_size:] = torch.Tensor(vs)

class EvaluatePrompt:
    def __init__(self, args):
        args.model_name = args.tgt_model
        args.model_dir = None
        args.k = 5
        args.eval_batch_size = 8

        dev_data = os.path.join(args.data_path, args.relation, 'dev.jsonl')
        test_data = os.path.join(args.data_path, args.relation, 'test.jsonl')

        self.model = Prober(args)
        self.original_vocab_size = len(list(self.model.tokenizer.get_vocab()))
        prepare_for_dense_prompt(self.model)
        
        if args.common_vocab is not None:
            self.vocab_subset = load_vocab(args.common_vocab)
            self.filter_indices, self.index_list = self.model.init_indices_for_filter_logprobs(self.vocab_subset)
        else:
            self.filter_indices = None
            self.index_list = None

        template = init_template(args, self.model)

        self.valid_samples = load_data(dev_data, template, vocab_subset=self.vocab_subset)
        self.valid_samples_batches, self.valid_sentences_batches = batchify(self.valid_samples, args.eval_batch_size)

        self.test_samples = load_data(test_data, template, vocab_subset=self.vocab_subset)
        self.test_samples_batches, self.test_sentences_batches = batchify(self.test_samples, args.eval_batch_size)

        self.args = args
        
    def evaluate_valid(self, embeddings):
        load_optiprompt(self.model, self.original_vocab_size, embeddings)
        precision, _ = evaluate(self.model, self.valid_samples_batches, self.valid_sentences_batches, self.filter_indices, self.index_list)

        return precision
    
    def evaluate_test(self, embeddings):
        if embeddings is not None:
            load_optiprompt(self.model, self.original_vocab_size, embeddings)
        precision, _ = evaluate(self.model, self.test_samples_batches, self.test_sentences_batches, self.filter_indices, self.index_list)

        return precision

