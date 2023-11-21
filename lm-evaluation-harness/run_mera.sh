#!/bin/bash

# internal MERA_COMMON_SETUP will be assigned 'default' value if external MERA_COMMON_SETUP not set or null.
# The value of external MERA_COMMON_SETUP remains untouched.
MERA_COMMON_SETUP_default="--model hf-causal --device cuda --max_batch_size=64 --batch_size=auto --inference --write_out"
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
    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python main.py --model_args "${MERA_MODEL_STRING}" \
    --output_base_path="${MERA_FOLDER}" ${MERA_COMMON_SETUP} --num_fewshot=${FEWSHOTS[$fewshot_idx]} \
    --output_path="${MERA_FOLDER}/${cur_task}_result.json" --tasks $cur_task

  done
done

# Try to save submission
python scripts/log_to_submission.py --outputs_dir "${MERA_FOLDER}" --dst_dir "${MERA_FOLDER}_submission"
