# MERA with Language Model Evaluation Harness

MERA: Multimodal Evaluation for Russian-language Architectures

The LM-harness support for the MERA benchmark datasets.

## Overview

This project provides a unified framework to test generative language models on MERA benchmark and its evaluation tasks.

## Install

To install `lm-eval` from the repository main branch, run:

```bash
cd lm-evaluation-harness
pip install -e .
```

To support loading GPTQ quantized models, install the package with the `auto-gptq` extra:

```bash
cd lm-evaluation-harness
pip install -e ".[auto-gptq]"
```

This commands are to be run from `lm-evaluation-harness` directory of this repository.

## MERA Benchmark:

### Run full benchmark with bash script

Sample command to run benchmark with `ai-forever/rugpt3large_based_on_gpt2` (`AutoModelForCausalLM` class compatible)
model from Huggingface Hub:

```linux
CUDA_VISIBLE_DEVICES=0 MERA_FOLDER="$PWD/mera_results/rugpt3large_760m_defaults" MERA_MODEL_STRING="pretrained=ai-forever/rugpt3large_based_on_gpt2,dtype=auto" bash scripts/run_benchmark.sh
```

Sample command to run benchmark with `ai-forever/FRED-T5-large` (`AutoModelForSeq2SeqLM` class compatible)
model from Huggingface Hub:

```linux
CUDA_VISIBLE_DEVICES=0 MERA_FOLDER="$PWD/mera_results/FRED-T5-large_defaults" MERA_MODEL_STRING="pretrained=ai-forever/FRED-T5-large,dtype=auto" bash scripts/run_benchmark.sh
```

Use `CUDA_VISIBLE_DEVICES` to set cuda device visibility, `MERA_FOLDER` for path to store outputs,
`MERA_MODEL_STRING` to setup `model_args` parameter of `lm-evaluation-harness`'s `lm_eval` module.
Use `MERA_COMMON_SETUP` to change default parameters for model inferencing with `main.py` (defaults are
`--model hf --device cuda --batch_size=1 --predict_only --log_samples --seed 1234,1234,None`).
See more on parameters in the next section.

Sample command for running benchmark with OpenAI API GPT-3 based models with `run_benchmark_openai_api.sh` script:

```linux
MERA_FOLDER="./davinci-002_defaults" MERA_MODEL_STRING="model=davinci-002" OPENAI_API_KEY=*YOUR API KEY* bash scripts/run_benchmark_openai_api.sh
```

Paste your OpenAI API key instead of `*YOUR API KEY*`. Script runs only GPT-3 based models like `davinci-002` or `babbage-002`.
Running of deprecated models is not guaranteed!

### Run specific bencmark manually (ruMMLU example)

Running specific benchmark available with `lm_eval` module.

Example:
```shell
CUDA_VISIBLE_DEVICES=3 lm_eval --model hf --model_args pretrained=mistralai/Mistral-7B-v0.1,dtype=auto,max_length=11500 \
--device cuda --output_base_path="$PWD/mera_results/Mistral-7B-v0.1_defaults" --batch_size=1 \
--predict_only --log_samples --seed 1234,1234,None --tasks rummlu --num_fewshot=5 \
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

Use `CUDA_VISIBLE_DEVICES` to set cuda device visibility (setting `--device cuda:3` works inconsisitently).

`--model hf` is used for models compatible with transformers' `AutoModelForCausalLM` or `AutoModelForSeq2SeqLM` class.

`--model_args` is for comma separated parameters of `from_pretrained` method of autoclass. One should be aware of
hardware requirements to run big models and limit maximum input length of models with parameter `max_length`
to avoid out-of-memory errors during run.

`--batch_size=1` is set to use batch size of 1 to maximize benchmark results reproducibility.
`--batch_size=auto` may be set to determine batch size for run automatically based on tasks and inputs maximum value
to start search down is set with `--max_batch_size`. Bigger batches may speed up running whole MERA benchmark,
but results may become irreproducible, so it is not default suggestion.

`--output_path` is path to json inside directory (will be created) to store data for submission preparation and logs.

`--predict_only` important to use this key always, it allows to run on datasets without proper replies provided
(score result 0 may still be reported).

`--log_samples` turn on samples logging, should be always to make the submission.


### Convert lm-harness to submission
Bash script above runs submission zip packing routine. Here is the way to run packing manually.

For converting run

```shell
python scripts/log_to_submission.py
```

Cmd arguments:

* `--outputs_dir` — path to directory with outputs (`MERA_FOLDER` from bash script above)
* `--dst_dir` — directory for store submission zip
* `--logs_public_submit` (`--no-logs_public_submit`) — pack logs for public submission in separate file (true by default)
