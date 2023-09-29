src_model=bert-base-cased
tgt_model=bert-large-cased
src_prompt_path=/OptiPrompt/optiprompt-outputs/${src_model}
log_path=/output/bert_base2bert_large_log

mkdir ${log_path}

for REL in P1001 P101 P103 P106 P108 P127 P1303 P131 P136 P1376 P138 P140 P1412 P159 P17 P176 P178 P19 P190 P20 P264 P27 P276 P279 P30 P31 P36 P361 P364 P37 P39 P407 P413 P449 P463 P47 P495 P527 P530 P740 P937; do

    python get_emb.py \
    --log_path ${log_path} \
    --relation ${REL} \
    --src_model ${src_model} \
    --tgt_model ${tgt_model} \
    --src_prompt_path ${src_prompt_path} \
    --prompt_filename prompt_vecs_5tokens.npy \
    --num_vectors 5 \
    --all_anchors \
    --topk 8192

done