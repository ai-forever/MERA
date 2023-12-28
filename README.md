# MERA

<p align="center">
  <picture>
    <img alt="MERA" src="docs/mera-logo.svg" style="max-width: 100%;">
  </picture>
</p>

<p align="center">
    <a href="https://opensource.org/licenses/MIT">
    <img alt="License" src="https://img.shields.io/badge/License-MIT-yellow.svg">
    </a>
    <a href="https://github.com/ai-forever/MERA/releases">
    <img alt="Release" src="https://img.shields.io/badge/release-v1.0.0-blue">
    </a>

</p>

<h2 align="center">
    <p> MERA (Multimodal Evaluation for Russian-language Architectures) is a new open benchmark for the Russian language for evaluating fundamental models.
</p>
</h2>


## About MERA

MERA benchmark brings together all industry and academic players in one place to study the capabilities of fundamental models, draw attention to AI problems, develop collaboration within the Russian Federation and in the international arena, and create an independent unified system for measuring all current models.

Our contributions of this project are:

- Instruction-based tasks available in 🤗HuggingFace dataset card [link](https://huggingface.co/datasets/ai-forever/MERA).
- LM-Harness evaluation code for models.
- Website of the benchmark with the [Leaderboard](https://mera.a-ai.ru/) and the scoring system inside.
- Baselines of the open models and Human Benchmark.

`v1.0.0`

The MERA benchmark includes 21 text tasks (17 base tasks + 4 diagnostic tasks). See the task-table for a complete list.
        
| Name | Task Name | Task Type | Test Size | N-shots | Metrics |
| --- | --- | --- | --- | --- | --- |
| BPS | bps | Code, Math | 1000 | 2 | acc |
| CheGeKa | chegeka | World Knowledge | 416 | 4 | f1 / em |
| LCS | lcs | Code, Math | 500 | 2 | acc |
| MathLogicQA | mathlogicqa | Math + Logic | 1143 | 5 | acc |
| MultiQ | multiq | Reasoning QA | 900 | 0 | f1 / em |
| PARus | parus | Common Sense | 500 | 0 | acc |
| RCB | rcb | NLI | 438 | 0 | f1_macro / acc |
| ruDetox | rudetox | Ethics | 1000 | 0 | sta, sim, fl, j |
| ruEthics | ruethics | Ethics | 645 | 0 | 5 mcc |
| ruHateSpeech | ruhatespeech | Ethics | 268 | 0 | acc |
| ruHHH | ruhhh | Ethics | 178 | 0 | acc |
| ruHumanEval | ruhumaneval | Math, Code, PLP | 164 | 0 | pass@k |
| ruMMLU | rummlu | Reasoning | 961 | 5 | acc |
| ruModAr | rumodar | Math, Logic | 6000 | 0 | acc |
| ruMultiAr | rumultiar | Math | 1024 | 5 | acc |
| ruOpenBookQA | ruopenbookqa | World Knowledge | 400 | 5 | f1_macro / acc |
| ruTiE | rutie | Reasoning, Dialogue Context, Memory | 430 | 0 | acc |
| ruWorldTree | ruworldtree | World Knowledge | 525 | 5 | f1_macro / acc |
| RWSD | rwsd | Reasoning | 260 | 0 | acc |
| SimpleAr | simplear | Math | 1000 | 5 | acc |
| USE | use | Exam | 900 | 0 | grade_norm |

Our aim is to evaluate all the models:

- in the same scenarios;
- using the same metrics;
- with the same adaptation strategy (e.g., prompting); 
- allowing for controlled and clear comparisons.

**Only united**, with the **support of all the companies** that are creating the foundation models in our country and beyond we could design the fair and transparent leaderboards for the models evaluation. 

*Our team and partners:* 

*SberDevices, Sber AI, Yandex, Skoltech AI, MTS AI, NRU HSE, Russian Academy of Sciences, etc.*

*Powered by [Aliance AI](https://a-ai.ru/)*

## Contents

The repository has the following structure:

- [`examples`](examples/instruction.ipynb) - the examples of loading and using data.
- [`humanbenchmarks`](humanbenchmarks/README.md) - materials and code for human evaluation.
- [`modules`](modules/scoring/README.md) - the examples of scoring scripts that are used on the website for scoring your submission.
- [`lm-evaluation-harness`](lm-evaluation-harness) - a framework for few-shot evaluation of language models.
    

## Submit to MERA

- To see the datasets use the HuggingFace datasets interface. See the example of the datasets in the prepared Jupyter Notebook.
- To run your model on the all datasets please use the code of lm-harness. The result of the code is the archive in ZIP format for the submission.
- Register on the website and submit your the ZIP. The results will be available for you privately in the account.

*The parameters of the generation, prompts and few-shot/zero-shot are fixed. You can vary them for your own purposes. If you want to submit your results on the public leaderboard check that these parameters are the same and please add the logs. We have to be sure that the scenarios for the models evaluation are the same and reproducible.*

We provide the[sample submission](modules/scoring/examples) for you to check the format.

The process of the whole MERA evaluation is described on the Figure:

![evaluation setup](docs/mera.png)

------------------------------------

📌 It’s the first text version of the benchmark. We are to expand and develop it in the future with new tasks and multimodality.

Feel free to ask any questions regarding our work, write on email mera@a-ai.ru. If you have ideas and new tasks feel free to suggest them, it’s important! If you see any bugs, or you know how to make the code better please suggest the fixes via pull-requests and issues in this official github 🤗 We will be glad to get the feedback in any way.
