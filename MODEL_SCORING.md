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

#### Running HF models

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
Use `MERA_COMMON_SETUP` to change default parameters for model inferencing with `lm_eval` (defaults are
`--model hf --device cuda --batch_size=1 --predict_only --log_samples --seed 1234 --verbosity ERROR`).
See more on parameters in the next section.

If you want to select only generative versions of tasks (all originally generative tasks and generative versions
of loglikelihood tasks), use `scripts/run_benchmark_gen.sh` script. To run all existing tasks
execute `scripts/run_benchmark_all.sh`. This way two separate submissions will be created: one for regular
MERA tasks (loglikelihood and generative tasks), one for generative MERA tasks only.

#### Running OpenAI models

Sample command for running benchmark with OpenAI API GPT-3 based models with `run_benchmark_openai_api.sh` script:

```linux
MERA_FOLDER="$PWD/mera_results/davinci-002_defaults" MERA_MODEL_STRING="model=davinci-002" OPENAI_API_KEY=*YOUR API KEY* bash scripts/run_benchmark_openai_api.sh
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

#### Notes on `lm_eval` settings

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

`--log_samples` turns on samples logging, should be always to make the submission.

`--apply_chat_template` turns on applying chat templates of your model to all requests (see more in [**documentation**](https://huggingface.co/docs/transformers/main/chat_templating)).

`--fewshot_as_multiturn` is used to turn fewshots in multi-turn conversation (currently cannot be used for zero-shot tasks and requires using `--apply_chat_template`).

`--system_instruction` contains a string that will be used as system prompt for one or more passed tasks (if the chat template of the model does not take into account system prompt, it will be omitted, so that no system prompt is passed to the model; without `--apply_chat_template` system prompt is added to the beginning of each request to the model).


### Convert lm-eval logs to submission
Bash script above runs submission zip packing routine. Here is the way to run packing manually.

For converting run

```shell
python scripts/log_to_submission.py --outputs_dir="$PWD/mera_results/rugpt3large_760m_defaults" --dst_dir="$PWD/mera_results/rugpt3large_760m_defaults_submission" --model_args="pretrained=ai-forever/rugpt3large_based_on_gpt2,dtype=auto"
```

Cmd arguments:

* `--outputs_dir` — path to directory with outputs (`MERA_FOLDER` from bash script above)
* `--dst_dir` — directory for store submission zip
* `--logs_public_submit` (`--no-logs_public_submit`) — pack logs for public submission in separate file (true by default)
* `--model_args` — string containing the same info that was passed in `MERA_MODEL_STRING`
* `--gen` — indicates that only generative tasks are to be packed in archive (false by default)

Be careful! After running the model the results will be stored in subdirectory of `MERA_FOLDER`.
Do not use the same `MERA_FOLDER` for running the same model twice (this way only the latest results
will be packed) or different models (this way two or more subdirectories will be created and you will
have to pass `--model_args` to determine which subfolder is to be packed). If you are not using `--model_args`
argument, make sure you provided the full path (including the subdirectory) in `--outputs_dir` argument.
For example above it will be as follows:

```shell
python scripts/log_to_submission.py --outputs_dir="$PWD/mera_results/rugpt3large_760m_defaults/ai-forever__rugpt3large_based_on_gpt2/" --dst_dir="$PWD/mera_results/rugpt3large_760m_defaults_submission"
```
