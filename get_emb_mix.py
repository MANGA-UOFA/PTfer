import numpy as np
import torch
import random
import argparse
from rel2abs_util import get_abs_anchors
import os
import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                datefmt='%m/%d/%Y %H:%M:%S',
                level=logging.INFO)
logger = logging.getLogger(__name__)

def save_embeddings(args, embeddings):
    save_path = os.path.join(args.tgt_prompt_path, args.relation, args.transferred_prompt_filename)
    with open(save_path, 'wb') as f:
        logger.info('Saving transferred prompt embeddings to %s' % save_path)
        np.save(f, embeddings)

def fix_seed(seed):
    # random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def transfer(args):
    v_base1 = np.load(os.path.join(args.src_prompt_path1, args.relation, args.prompt_filename))
    v_base2 = np.load(os.path.join(args.src_prompt_path2, args.relation, args.prompt_filename))

    src_anchors1, tgt_anchors, tgt_distribution, src_anchors2 = get_abs_anchors(
        args.src_model1, 
        args.tgt_model, 
        args.num_anchor, 
        args.common_vocab, 
        args.seed,
        args.all_anchors,
        model_name3= args.src_model2, 
        )
    
    logger.info('Start searching for prompt embeddings for relation %s' % args.relation)
    
    from rel2abs_GO_mix import Rel2abs_Decoder
    decoder = Rel2abs_Decoder(args, 
                              logger, 
                              v_base1, 
                              src_anchors1, 
                              v_base2, 
                              src_anchors2, 
                              tgt_anchors, 
                              tgt_distribution
                              )
    x = decoder.search()

    save_embeddings(args, x)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--relation', type=str, default='P17')
    parser.add_argument('--prompt_path', type=str, default='OptiPrompt/optiprompt-outputs')
    parser.add_argument('--prompt_filename', type=str, default='prompt_vecs.npy')
    parser.add_argument('--transferred_prompt_filename', type=str, default='transfered_prompt_vecs.npy')
    parser.add_argument('--src_model1', type=str, default='bert-base-cased')
    parser.add_argument('--src_model2', type=str, default='roberta-base')
    parser.add_argument('--tgt_model', type=str, default='bert-large-cased')
    parser.add_argument('--num_anchor', type=int, default=8192)
    parser.add_argument('--common_vocab', type=str, default='OptiPrompt/common_vocabs/common_vocab_cased_be_ro_al.txt')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_vectors', type=int, default=5)
    parser.add_argument('--log_path', type=str, default='OptiPrompt/analyze_embed/output/log')
    parser.add_argument('--log_tag', type=str, default='')

    # args for validation
    parser.add_argument('--relation_profile', type=str, default='OptiPrompt/relation_metainfo/LAMA_relations.jsonl')
    parser.add_argument('--data_path', type=str, default='OptiPrompt/data/autoprompt_data')
    parser.add_argument('--init_manual_template', action='store_true')
    parser.add_argument('--budget', type=int, default=2000)
    parser.add_argument('--lr', type=float, default=5e-3)
    parser.add_argument('--all_anchors', action='store_true')
    parser.add_argument('--topk', type=int, default=0)
    parser.add_argument('--absolute', action='store_true')

    args = parser.parse_args()

    args.src_prompt_path1 = os.path.join(args.prompt_path, args.src_model1)
    args.src_prompt_path2 = os.path.join(args.prompt_path, args.src_model2)
    args.tgt_prompt_path = os.path.join(args.prompt_path, args.tgt_model)

    if args.init_manual_template:
        if args.topk > 0:
            log_file = os.path.join(args.log_path, 'mix_training_top%d_manual%s.log' % (args.topk, args.log_tag))
        else:
            log_file = os.path.join(args.log_path, 'mix_training_%d_manual%s.log' % (args.num_anchor, args.log_tag))
    else:
        if args.topk > 0:
            log_file = os.path.join(args.log_path, 'mix_training_top%d%s.log' % (args.topk, args.log_tag))
        else:
            log_file = os.path.join(args.log_path, 'mix_training_%d%s.log' % (args.num_anchor, args.log_tag))
        
    logger.addHandler(logging.FileHandler(log_file))

    logger.info(args)

    fix_seed(args.seed)
    transfer(args)