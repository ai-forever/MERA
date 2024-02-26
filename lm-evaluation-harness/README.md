# MERA with Language Model Evaluation Harness

MERA: Multimodal Evaluation for Russian-language Architectures

The LM-harness support for the MERA benchmark datasets.

## Overview

This project provides a unified framework to test generative language models on the MERA benchmark and its evaluation tasks.

## Install

To install `lm-eval` from the repository main branch, run the following command:

```bash
pip install -e .
```

To support loading GPTQ quantized models, install the package with the `auto-gptq` extra:

```bash
pip install -e ".[auto-gptq]"
```

## MERA Benchmark:

### Run full benchmark with the bash script

Below is a sample command to run the benchmark with `ai-forever/rugpt3large_based_on_gpt2` (`AutoModelForCausalLM` class compatible)
model from Huggingface Hub:

```linux
CUDA_VISIBLE_DEVICES=0 MERA_FOLDER="$PWD/mera_results/rugpt3large_760m_defaults" MERA_MODEL_STRING="pretrained=ai-forever/rugpt3large_based_on_gpt2,dtype=auto" bash run_mera.sh
```

Below is a sample command to run the benchmark with `ai-forever/FRED-T5-large` (`AutoModelForSeq2SeqLM` class compatible)
model from Huggingface Hub:

```linux
CUDA_VISIBLE_DEVICES=0 MERA_FOLDER="$PWD/mera_results/FRED-T5-large_defaults" MERA_MODEL_STRING="pretrained=ai-forever/FRED-T5-large,dtype=auto" bash run_mera_seq2seq.sh
```

Use `CUDA_VISIBLE_DEVICES` to set cuda device visibility, `MERA_FOLDER` for path to store outputs,
`MERA_MODEL_STRING` to setup `model_args` parameter of `lm-evaluation-harness`'s `main.py`.
Use `MERA_COMMON_SETUP` to change default parameters for model inferencing with `main.py` (defaults are
`--model hf-causal-experimental --device cuda --max_batch_size=64 --batch_size=auto --inference`).
See more on parameters in the next section.

Notice two different bash scripts: `run_mera.sh` for models compatible with transformers' `AutoModelForCausalLM` class,
and `run_mera_seq2seq.sh` for models compatible with transformers' `AutoModelForSeq2SeqLM` class.

### Run specific benchmark tasks manually (ruMMLU example)

Running specific benchmark tasks is available with the `main.py` script.

Example:
```shell
CUDA_VISIBLE_DEVICES=3 python main.py --model hf-causal-experimental --model_args pretrained=mistralai/Mistral-7B-v0.1,dtype=auto,max_length=11500 \
--device cuda --output_base_path="$PWD/mera_results/Mistral-7B-v0.1_defaults" --batch_size=1 \
--inference --write_out --no_cache --tasks rummlu --num_fewshot=5 \
--output_path="$PWD/mera_results/Mistral-7B-v0.1_defaults/rummlu_result.json"
```

#### Notes on `main.py` settings

Use `--tasks` to provide comma separated list of tasks to run (available options are: `bps`, `chegeka`, `lcs`,
`mathlogicqa`, `multiq`, `parus`, `rcb`, `rudetox`, `ruethics`, `ruhatespeech`, `ruhhh`, `ruhumaneval`, `rummlu`,
`rumodar`, `rumultiar`, `ruopenbookqa`, `rutie`, `ruworldtree`, `rwsd`, `simplear`, `use`).
Avoiding this argument will run all tasks with same provided settings.

`--num_fewshot` sets fewshot count. MERA supposes to run tasks with the following fewshot count:
* `--num_fewshot=0` (zeroshot) with `multiq`, `parus`, `rcb`, `rumodar`, `rwsd`, `use`, `rudetox`, `ruethics`,
`ruhatespeech`, `ruhhh`, `rutie`, and `ruhumaneval`;
* `--num_fewshot=2` with `bps` and `lcs`;
* `--num_fewshot=4` with `chegeka`;
* `--num_fewshot=5` with `mathlogicqa`, `ruworldtree`, `ruopenbookqa`, `simplear`, `rumultiar`, and `rummlu`.

Use `CUDA_VISIBLE_DEVICES` to set cuda device visibility (setting `--device cuda:3` works inconsistently).

`--model hf-causal-experimental` is used for models compatible with transformers' `AutoModelForCausalLM` class
and `hf-seq2seq` is used for models compatible with transformers' `AutoModelForSeq2SeqLM` class.

`--model_args` is for comma separated parameters of `from_pretrained` method of autoclass. One should be aware of
hardware requirements to run big models and limit the maximum input length of models with the parameter `max_length`
to avoid out-of-memory errors during a run.

`--batch_size=1` is set to use a batch size of 1 to maximize benchmark results reproducibility.
`--batch_size=auto` may be set to determine a batch size automatically based on the evaluated tasks and inputs maximum value
to start to search down is set with `--max_batch_size`. Bigger batches may speed up running the whole MERA benchmark,
but results may become irreproducible, so it is not the default suggestion.

`--output_base_path` is a path to dir (will be created) to store data for submission preparation and logs.

`--inference` is important to use this key always. It allows to run on datasets without proper replies provided
(score result 0 will be reported).

The `--write_out` command turns on extra logging necessary for public submissions. 

`--no_cache` is used to turn off the caching of tokenized inputs and model files (datasets are not cached).

`--output_path` is a path to an extra log file with parameters of run and results of the task. It is preferred to be inside
`output_base_path` directory.


### Convert lm-harness to submission
The bash script above runs the submission zip-packing routine. Below is a way to run packing manually.

For converting run

```shell
python scripts/log_to_submission.py
```

Cmd arguments:

* `--outputs_dir` — path to directory with outputs (`MERA_FOLDER` from bash script above)
* `--dst_dir` — directory for store submission zip
* `--dataset_dir` — path to `lm_eval/datasets/`
* `--logs_public_submit` (`--no-logs_public_submit`) — pack logs for public submission in separate file (true by default)
