#!/bin/bash

# internal MERA_COMMON_SETUP will be assigned 'default' value if external MERA_COMMON_SETUP not set or null.
# The value of external MERA_COMMON_SETUP remains untouched.
MERA_COMMON_SETUP_default="--model hf-seq2seq --device cuda --batch_size=1 --inference --write_out --no_cache"
MERA_COMMON_SETUP="${MERA_COMMON_SETUP:-$MERA_COMMON_SETUP_default}"

FEWSHOTS=(
  4
  2
  5
  0
)

TASKS=(
"chegeka"
"bps lcs"
"mathlogicqa ruworldtree ruopenbookqa simplear rumultiar rummlu"
"multiq parus rcb rumodar rwsd use rudetox ruethics ruhatespeech ruhhh rutie ruhumaneval"
)

for fewshot_idx in "${!FEWSHOTS[@]}"
do
  for cur_task in ${TASKS[$fewshot_idx]}
  do
    HF_DATASETS_CACHE="${MERA_FOLDER}/ds_cache" TOKENIZERS_PARALLELISM=false HF_DATASETS_IN_MEMORY_MAX_SIZE=23400000 \
    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python main.py --model_args "${MERA_MODEL_STRING}" \
    --output_base_path="${MERA_FOLDER}" ${MERA_COMMON_SETUP} --num_fewshot=${FEWSHOTS[$fewshot_idx]} \
    --output_path="${MERA_FOLDER}/${cur_task}_result.json" --tasks $cur_task

  done
done

# Try to save submission
HF_DATASETS_CACHE="${MERA_FOLDER}/ds_cache" HF_DATASETS_IN_MEMORY_MAX_SIZE=23400000 \
python scripts/log_to_submission.py --outputs_dir "${MERA_FOLDER}" --dst_dir "${MERA_FOLDER}_submission"

# Remove datasets cache folder
rm -r "${MERA_FOLDER}/ds_cache"
