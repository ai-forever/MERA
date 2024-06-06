# **ruModAr**

## Task Description

Modified Arithmetic is a mathematical task from [BIG-bench](https://github.com/google/BIG-bench/tree/main/bigbench/benchmark_tasks/modified_arithmetic). The task tests a model's ability to learn new knowledge from context examples and then calculate the results based on new skills.
Each question in each subtask begins with a prompt and five examples of arithmetic expressions with results. The sixth example is incomplete, the model's task is to finish it correctly.

**Keywords:** arithmetic, free response, few-shot, mathematics

### Motivation

Can large language models learn new skills and understand operations from a few examples? This task probes this question with a series of simple few-shot tasks, each involving computing a joint arithmetic function with correctly recognizing a pattern very similar to, yet subtly different from, standard arithmetic operations common in training data.

## Dataset Description

Each subtask (addition, subtraction, multiplication w/o adding `+1` to result) includes 1000 questions. The symbol -> is used instead of = because the last one already has a definite canonical meaning. The symbol -> can mean “=” or “+ 1 = ”. In the end, we got sets for 6 subtasks: addition_control, addition_plus_one, subtraction_control, subtraction_plus_one, multiplication_control, multiplication_plus_one. The arguments of the two-digit subtasks (multiplication_ prefix) are randomly generated from [0, 100), and arguments of the three-digit subtasks (addition_ and subtraction_ prefix) — [0, 1000).

### Data fields

- `instruction` is an instructional prompt specified for the current task;
- `inputs` is five expressions for recognising the pattern, the sixth for calculating by a model;
- `outputs` is the target, the resulted answer for the last expression;
- `meta` is an additional information field:
    - `id` is the id of the example from the dataset;
    - `task_type` is the subtask type.

### Data Instances

Below is an example from the subtask three_digit_addition_plus_one:

```json
{
    "instruction": "В следующих строках символ -> представляет собой одну простую математическую операцию. Определи операцию и вычисли последний пример:\n{inputs}",
    "inputs": "102 + 435 -> 538\n860 + 270 -> 1131\n106 + 71 -> 178\n700 + 20 -> 721\n614 + 121 -> 736\n466 + 214 ->",
    "outputs": "681",
    "meta": {
        "id": 1,
        "task_type": "three_digit_addition_plus_one"
    }
}
```

### Data Splits

The dataset consists of a public test (train split) (`6000` samples) with labeled examples and a closed test set (test split) (`6000` samples) for model evaluation.

### Prompts

5 prompts of varying difficulty were created for this task. Example:

`В следующих строках символ -> представляет собой одну простую математическую операцию. Вычисли последний пример с учетом результатов вычисленных выражений:\n{inputs}`

### Dataset creation

Public test set was taken from the Big-Bench.

Closed test was generated from scratch based on the original methodology of Big-Bench.

## Evaluation

### Metrics

The task is evaluated using the Exact Match (EM). For each example, 1.0 is given for the target sequence that EXACTLY matches the predicted sequence. Else, 0.0. 

### Human Benchmark

The human benchmark is measured on a subset of size 1800 (300 samples per subtask from test set with the original target distribution). Evaluate on one pool (all subtasks) with an overlap of 5 reviewers per task.

The final score is `0.999`.

## References

[1] Brown, T.B., et al. (2020) Language models are few-shot learners. arXiv:2005.14165.
