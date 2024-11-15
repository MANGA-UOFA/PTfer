# Zero-Shot Continuous Prompt Transfer: Generalizing Task Semantics Across Language Models
PyTorch implementation of the paper. 

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

### Cross-Model Transfer
#### Getting prompt vectors from GPT-2
An example of how to run OptimPrompt on GPT-2 models
```bash
cd gpt_code
bash test_run.sh
```

#### Transfering prompts to GPT-2
In order to evaluate the transfered prompt on GPT-2, you need to replace one line in `rel2abs_GO.py` file. Replace
```python
from evaluate import EvaluatePrompt
```
with
```python
from evaluate_gpt import EvaluatePrompt
```

## Citation
If you find our work interesting and useful, feel free to cite our publication:

```bibtex
@inproceedings{
  wu2024zeroshot,
  title={Zero-Shot Continuous Prompt Transfer: Generalizing Task Semantics Across Language Models},
  author={Zijun Wu and Yongkang Wu and Lili Mou},
  booktitle={The Twelfth International Conference on Learning Representations},
  year={2024},
  url={https://openreview.net/forum?id=26XphugOcS}
}
```
