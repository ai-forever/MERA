## Human Benchmark

We provide human baselines for the datasets included in the MERA benchmark. Most of the baselines were obtained using crowdsource platforms ([Yandex.Toloka](https://toloka.ai/ru/) & [ABC Elementary](https://elementary.activebc.ru/)) where the annotators were presented with the same instructions and tasks as it is intended for language models in MERA. The exceptions are the USE (Unified State Exam) dataset and the ruHumanEval dataset, since the USE baseline is approximated with average scores earned in real exams. The details about human assessment for Russian SuperGLUE and TAPE are covered in the corresponding papers ([RSG paper](https://aclanthology.org/2020.emnlp-main.381.pdf), [TAPE paper](https://arxiv.org/pdf/2210.12813.pdf)), we use the previous results of human evaluations on these two datasets.

The general procedure of the assessment of human performance is the following. Crowdsource annotators were presented with a subset of tasks from every dataset so that every sample was annotated by at least 5 people. The target test tasks alternated with control tasks, the answers to the latter were used to filter out fraud annotations (if an annotator underperforms on control tasks — less than 50% accuracy — their answers are eliminated from the set). We calculated the final human answer for each sample by majority vote (MV), which is 3 votes for the 5 individual answers per sample. The samples that did not get 3+ consistent answers were eliminated from the set (see the table below for the resulting subset sizes). The aggregated human scores were compared with the gold answers to obtain total human performance metric value for each task. The `humanbenchmarks` shares the code used to conduct human evaluation, but we intentionally removed the files with gold answers from the repo (except for the open diagnostic datasets: ruDetox, ruEthics, ruHateSpeech, ruHHH).

The results of human benchmarking are presented in the table below.

- **Human baseline** is the value of the **Metrics** relevant to the **Dataset**;
- **Samples in human evaluation with MV** is the number of samples that achieved 3+ consistent answers and thus were considered to calculate **Human baseline**;
- **Samples in test set** is the total number of samples in the test set.

| Dataset | Human baseline | Metrics | Samples in human evaluation with MV | Samples in test set |
| --- | --- | --- | --- | --- |
| BPS | 1.0 | Accuracy | 100 | 1000 |
| LCS | 0.704 | Accuracy | 54 | 500 |
| MathLogicQA | 0.995 | Accuracy | 1130 | 1143 |
| PARus | 0.982 | Accuracy | 400 | 400 |
| RCB | 0.68 / 0.702 | F1-score / Accuracy | 438 | 438 |
| RWSD | 0.837 | Accuracy | 259 | 260 |
| ruDetox | 0.477 | J = STA * SIM * FL | 404 | 1000 |
| ruEthics | *** | 5 MCC | 645 | 645 |
| ruHateSpeech | 0.985 | Accuracy | 265 | 268 |
| ruHHH | 0.814 | Accuracy | 177 | 178 |
| ruHumanEval | 1.0 | pass@k | N/A | N/A |
| ruMMLU | 0.898 | Accuracy | N/A | 961 |
| ruModAr | 0.999 | Accuracy | 1794 | 6000 |
| ruMultiAr | 1.0 | Accuracy | 519 | 1024 |
| ruTiE | 0.977 | Accuracy | 430 | 430 |
| SimpleAr | 1.0 | Accuracy | 200 | 1000 |
| CheGeKa | 0.719 / 0.645 | F1-score / Exact Match | 416 | 416 |
| MultiQ | 0.928 / 0.910 | F1-score / Exact Match | 900 | 900 |
| ruWorldTree | 0.838 / 0.837 | F1-score / Accuracy | 525 | 525 |
| ruOpenBookQA | 0.875 / 0.865 | F1-score / Accuracy | 400 | 400 |
| USE | 0.701 | Grade Norm | N/A | N/A |

### Russian SuperGLUE and TAPE

Human benchmark values for Russian SuperGLUE and TAPE are from the corresponding original papers ([RSG paper](https://aclanthology.org/2020.emnlp-main.381.pdf), [TAPE paper](https://arxiv.org/pdf/2210.12813.pdf)). The exception is the RWSD dataset for which we recalculated human baseline since the test set was extended.

### ruDetox

We conducted a 3-fold annotation of the 800 public test samples:

1. if the toxic and the non-toxic text are close in meaning (0/1),
2. if the non-toxic text is written correctly (0/1),
3. if the non-toxic text is non-offensive (0/1).

We selected only those text pairs that obtained "1" in every category (this resulted in 404 samples). The selected texts were evaluated with an aggregative score (considering accuracy, meaning and fluency) with an encoder language model.

### ruEthics

The 5 MCC scoring for the ruEthics dataset consists of a set of Matthews Correlation Coefficient (MCC) scores stratified by the 3 types of question in the prompt and by the 5 ethical categories.

|        | "act right"   | "act well"    | "act ethically" |
|------------------|---------------|---------------|-------|
| justice          | 0.748         | 0.789         | 0.729 |
| law              | 0.864         | 0.832         | 0.817 |
| moral            | 0.880         | 0.837         | 0.811 |
| utilitarianism   | 0.684         | 0.675         | 0.665 |
| virtue           | 0.813         | 0.802         | 0.771 |

### ruHumanEval

The generated functions are evaluated using the pass@k metric. It depends on the number of correct solutions and uses the k coefficient k = {1, 5, 10}. By focusing on functional correctness rather than text similarity, the pass@k metric offers a more meaningful and practical assessment of a model's ability to solve programming challenges. This approach aligns more closely with the practices of human developers and provides a valuable benchmark for the ongoing development of code generation models. The human-level for all the 3 metrics (pass@1, pass@5, pass@10) is 1.0.

### ruMMLU

According to the [original MMLU paper](https://arxiv.org/pdf/2009.03300v3.pdf), for the test in English human-level accuracy varies: "Unspecialized humans from Amazon Mechanical Turk obtain 34.5% accuracy on English test. Meanwhile, expert-level performance can be far higher. For example, real-world test-taker human accuracy at the 95th percentile is around 87% for US Medical Licensing Examinations, and these questions make up our “Professional Medicine” task. If we take the 95th percentile human test-taker accuracy for exams that build up our test, and if we make an educated guess when such information is unavailable, we then estimate that expert-level accuracy is approximately 89.8%.”.

### USE

Since the content of the Unified State Exam, the complexity of the tasks, as well as the assessment system changes depending on the year, the average primary score of graduates for completing Part 1 of the Unified State Exam in the Russian language in 2019 is used as a human assessment. Based on official statistics the average primary score for Part 1 was 23.835 out of 34 points, value grade_norm is 0.701.
