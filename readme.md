# Zero-Shot Continuous Prompt Transfer: Generalizing Task Semantics Across Language Models
PyTorch implementation of the paper

## Setup
Install required packages using the following command:
```bash
pip install -r requirement.txt
```

Follow [OptiPrompt](https://github.com/princeton-nlp/OptiPrompt.git) to obtain the datasets and the source prompts: 
```bash
git clone https://github.com/princeton-nlp/OptiPrompt.git
```

## Run transfer
### Single source
An example for transfering prompts from bert-base to bert-large model
```bash
REL=P17
src_model=bert-base-cased
tgt_model=bert-large-cased
src_prompt_path=/OptiPrompt/optiprompt-outputs/${src_model}
log_path=/output/bert_base2bert_large_log
mkdir ${log_path}

python get_emb.py \
    --log_path ${log_path} \
    --relation ${REL} \
    --src_model ${src_model} \
    --tgt_model ${tgt_model} \
    --src_prompt_path ${src_prompt_path} \
    --prompt_filename prompt_vec.npy \
    --num_vectors 5 \
    --all_anchors \
    --topk 8192 
```

Use the script to run all relations
```bash
bash get_emb.sh
```

### Dual sources
An example for transfering prompts from a mixture of bert-base and robert-base to bert-large model
```bash
REL=P17
src_model1=bert-base-cased
src_model2=roberta-base
tgt_model=bert-large-cased
prompt_path=OptiPrompt/optiprompt-outputs
log_path=OptiPrompt/analyze_embed/mix_output/bert_base_roberta_base2bert_large
mkdir ${log_path}

python get_emb_mix.py \
    --log_path ${log_path} \
    --relation ${REL} \
    --src_model1 ${src_model1} \
    --src_model2 ${src_model2} \
    --tgt_model ${tgt_model} \
    --prompt_path ${prompt_path} \
    --prompt_filename prompt_vecs.npy \
    --num_vectors 5 \
    --all_anchors \
    --topk 8192
```

Use the script to run all relations
```bash
bash get_emb_mix.sh
```