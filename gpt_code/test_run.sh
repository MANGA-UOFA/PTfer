OUTPUTS_DIR=optiprompt-outputs/gpt2-test
MODEL=gpt2
REL=P17

DIR=${OUTPUTS_DIR}/${REL}
mkdir -p ${DIR}

python gpt_code/run_optiprompt_gpt.py \
    --relation_profile relation_metainfo/LAMA_relations.jsonl \
    --relation ${REL} \
    --common_vocab_filename common_vocabs/common_vocab_cased_be_ro_al.txt \
    --model_name ${MODEL} \
    --do_train \
    --train_data data/autoprompt_data/${REL}/train.jsonl \
    --dev_data data/autoprompt_data/${REL}/dev.jsonl \
    --do_eval \
    --test_data data/LAMA-TREx/${REL}.jsonl \
    --output_dir ${DIR} \
    --output_predictions \
    --num_vectors 5
