from transformers import AutoConfig
from transformers import RobertaTokenizer, RobertaModel, RobertaConfig
from transformers import BertTokenizer, BertModel, BertConfig
from transformers import AlbertConfig, AlbertTokenizer, AlbertModel
from transformers import GPT2Config, GPT2Model, GPT2Tokenizer
from transformers import BartConfig, BartTokenizer, BartModel
from transformers import T5Config, T5Tokenizer, T5Model
import numpy as np

def get_model_tokenizer(model_name):
    config = AutoConfig.from_pretrained(model_name)
    if isinstance(config, RobertaConfig):
        tokenizer = RobertaTokenizer.from_pretrained(model_name)
        base_model = RobertaModel.from_pretrained(model_name)
        model_family = 'roberta'
    elif isinstance(config, BertConfig):
        tokenizer = BertTokenizer.from_pretrained(model_name)
        base_model = BertModel.from_pretrained(model_name)
        model_family = 'bert'
    elif isinstance(config, AlbertConfig):
        tokenizer = AlbertTokenizer.from_pretrained(model_name)
        base_model = AlbertModel.from_pretrained(model_name)
        model_family = 'albert'
    elif isinstance(config, GPT2Config):
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        base_model = GPT2Model.from_pretrained(model_name)
        model_family = 'gpt2'
    elif isinstance(config, BartConfig):
        tokenizer = BartTokenizer.from_pretrained(model_name)
        base_model = BartModel.from_pretrained(model_name)
        model_family = 'bart'
    elif isinstance(config, T5Config):
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        base_model = T5Model.from_pretrained(model_name)
        model_family = 'T5'
    else:
        raise ValueError('Model %s not supported yet!'%(model_name))
    
    return tokenizer, base_model, model_family

def build_vocab(tokenizer):
    vocab = list(tokenizer.get_vocab())
    inverse_vocab = {w: i for i, w in enumerate(vocab)}
    return vocab, inverse_vocab

def select_random_samples(matrix, indices):
    k = matrix.shape[1]
    m = indices.shape[0]
    selected_embeddings = np.empty((m, k))
    
    for i, idx in enumerate(indices):
        selected_embeddings[i] = matrix[idx]
    
    return selected_embeddings

def vocab_cleaning(vocab):
    vocab = [token.replace("Ġ", "") for token in vocab] # for roberta
    vocab = [token.replace("▁", "") for token in vocab] # for albert
    return vocab

def get_abs_anchors(model_name1, model_name2, num_anchor, common_vocab, seed, all_anchors, model_name3=None):
    bert_tokenizer, bert_base_model, model_family1 = get_model_tokenizer(model_name1)
    roberta_tokenizer, roberta_base_model, model_family2 = get_model_tokenizer(model_name2)
    if model_name3 != None:
        model_tokenizer3, model3, model_family3 = get_model_tokenizer(model_name3)

    bert_UNK = bert_tokenizer.unk_token
    roberta_UNK = roberta_tokenizer.unk_token
    if model_name3 != None:
        model3_UNK = model_tokenizer3.unk_token

    if model_family1 in ['gpt2']:
        bert_base_embeddings = bert_base_model.wte.weight.detach().cpu()
    elif model_family1 in ['bart', 'T5']:
        bert_base_embeddings = bert_base_model.shared.weight.detach().cpu()
    else:
        bert_base_embeddings = bert_base_model.embeddings.word_embeddings.weight.detach().cpu()

    if model_family2 in ['gpt2']:
        roberta_base_embeddings = roberta_base_model.wte.weight.detach().cpu()
    elif model_family2 in ['bart', 'T5']:
        roberta_base_embeddings = roberta_base_model.shared.weight.detach().cpu()
    else:
        roberta_base_embeddings = roberta_base_model.embeddings.word_embeddings.weight.detach().cpu()
    
    if model_name3 != None:
        if model_family3 in ['gpt2']:
            model3_embeddings = model3.wte.weight.detach().cpu()
        elif model_family3 in ['bart', 'T5']:
            model3_embeddings = model3.shared.weight.detach().cpu()
        else:
            model3_embeddings = model3.embeddings.word_embeddings.weight.detach().cpu()

    if common_vocab != None:
        with open(common_vocab, 'r') as f:
            lines = f.readlines()
            filtered_vocab = [x.strip() for x in lines]
    else:
        bert_vocab, _ = build_vocab(bert_tokenizer)
        roberta_vocab, _ = build_vocab(roberta_tokenizer)

        if model_name3 != None:
            model_vocab3, _ = build_vocab(model_tokenizer3)


        print('model 1 and 2 vocabs:', len(bert_vocab), len(roberta_vocab))
        bert_vocab = vocab_cleaning(bert_vocab)
        roberta_vocab = vocab_cleaning(roberta_vocab)

        if model_name3 != None:
            model_vocab3 = vocab_cleaning(model_vocab3)
            filtered_vocab = list(set(bert_vocab) & set(roberta_vocab) & set(model_vocab3))
        else:
            filtered_vocab = list(set(bert_vocab) & set(roberta_vocab))

    bert_filtered_indices = []
    roberta_filtered_indices = []

    if model_name3 != None:
        model3_filtered_indices = []

    # wf = open('aligned_vocab/%s_%s_aligned_vocab.txt' % (model_family1, model_family2), 'w')
    for word in filtered_vocab:
        # wf.write(word+'\n')

        bert_tokens = bert_tokenizer.tokenize(' ' + word)
        if (len(bert_tokens) == 1) and (bert_tokens[0] != bert_UNK):
            bert_index = bert_tokenizer.convert_tokens_to_ids(bert_tokens)[0]
        else:
            continue
        
        roberta_tokens = roberta_tokenizer.tokenize(' ' + word)
        if (len(roberta_tokens) == 1) and (roberta_tokens[0] != roberta_UNK):
            roberta_index = roberta_tokenizer.convert_tokens_to_ids(roberta_tokens)[0]
        else:
            continue
        
        if model_name3 != None:
            model3_tokens = model_tokenizer3.tokenize(' ' + word)
            if (len(model3_tokens) == 1) and (model3_tokens[0] != model3_UNK):
                model3_index = model_tokenizer3.convert_tokens_to_ids(model3_tokens)[0]
            else:
                continue

            model3_filtered_indices.append(model3_index)
        
        # wf.write('%s, %s\n' % (bert_tokens[0], roberta_tokens[0]))
        bert_filtered_indices.append(bert_index)
        roberta_filtered_indices.append(roberta_index)
            
    assert len(bert_filtered_indices) == len(roberta_filtered_indices)
    if model_name3 != None:
        assert len(bert_filtered_indices) == len(model3_filtered_indices)

    print('Number of filtered shared tokens %d' % len(bert_filtered_indices))

    std = np.std(roberta_base_embeddings.reshape(-1).numpy())
    mean = np.mean(roberta_base_embeddings.reshape(-1).numpy())

    if all_anchors:
        bert_base_embeddings = select_random_samples(bert_base_embeddings.numpy(), np.array(bert_filtered_indices))
        roberta_base_embeddings = select_random_samples(roberta_base_embeddings.numpy(),np.array(roberta_filtered_indices))
        if model_name3 != None:
            model3_embeddings = select_random_samples(model3_embeddings.numpy(),np.array(model3_filtered_indices))
        else:
            model3_embeddings = None
    else:
        selected_indices = np.random.choice(list(range(0, len(bert_filtered_indices))), num_anchor, replace=False)
        bert_base_embeddings = select_random_samples(bert_base_embeddings.numpy(), np.array([bert_filtered_indices[i] for i in selected_indices]))
        roberta_base_embeddings = select_random_samples(roberta_base_embeddings.numpy(), np.array([roberta_filtered_indices[i] for i in selected_indices]))
        if model_name3 != None:
            model3_embeddings = select_random_samples(model3_embeddings.numpy(), np.array([model3_filtered_indices[i] for i in selected_indices]))
        else:
            model3_embeddings = None

    return bert_base_embeddings, roberta_base_embeddings, (mean, std), model3_embeddings
