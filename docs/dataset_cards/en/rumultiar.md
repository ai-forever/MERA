# **ruMultiAr**

## Task Description

Multistep Arithmetic is a mathematical task from [BIG-bench](https://github.com/google/BIG-bench/blob/main/bigbench/benchmark_tasks/multistep_arithmetic/README.md). This task tests a model's ability to solve multistep arithmetic operations composed of addition, subtraction, multiplication, and division. So we can measure the capability of models to think sequentially.

**Keywords:** arithmetic, free response, mathematics, zero-shot

**Authors:** Albina Akhmetgareeva, Pablo Antonio, Moreno Casares

### Motivation

This problem is relatively simple for humans as it is solved step-by-step. Therefore, the tasks aim to check the capability of systems to decompose complex problems into more straightforward steps and plan actions. Moreover, sequential reasoning is one skill within the Fluid Intelligence ability due to the Cattell-Horn-Carroll theory of cognitive capabilities [1]. This test aims to measure precisely that skill.

## Dataset Description

The task is a tree-like arithmetic expression with multiple levels and different content lengths inside the inner-most parenthesis.

### Data Fields

- `instruction` is an instructional prompt specified for the current task;
- `inputs` is the mathematical expression;
- `outputs` is the target, the result of multi-step operations;
- `meta` is an additional information field:
    - `id` is the example id in the dataset.

### Data Instances

Below are examples from the dataset:

```json
{
    "instruction": "Вычисли результат выражения:\n{inputs}",
    "inputs": "((-3) + 5) = ",
    "outputs": "2",
    "meta": {
        "id": 1
    }
}
```


### Data Splits

The dataset consists of a training set (`1039` samples) with labeled examples and a test set (`1024` samples) for model evaluation.

### Prompts

6 prompts of varying difficulty were created for this task. Example:

`"Выполни следующие базовые арифметические операции в правильном порядке, в том числе учитывая порядок скобок, и напиши результат вычисления выражения в виде одного числа:\n{inputs}"`

### Dataset creation

The data in this task is generated using a Python script. The script generates examples by iterating through various configurations with different nesting depths and the number of arguments in parentheses. It filters the examples, considering the following criteria.

The arguments for the task are generated from [-9; 9]. The `random_seed` for the test was selected so that the samples did not overlap with the open set as much as possible.

Both sets were filtered in such a way that:

- target values range from -1000 to 1000;
- target values occurred no more than 10 times in the set split;
- no duplicates occurred;
- for samples with division: taken expressions with integer result.

## Evaluation

### Metrics

The task is evaluated using the Exact Match (EM) For each example, 1 is given for the target sequence EXACTLY matches the predicted sequence. Else, 0. The total score is equal to average sequence-level accuracy.

### Human Benchmark

It is measured on a subset of `600` examples, sampled with varying complexity of operations — ~50 per configuration. Evaluate on one pool (all subtasks) with overlap: 5 reviewers per task.

The final human score is `0.998`.

## Limitations

1. Only numerical answers (e.g., "4") are considered for model evaluation instead of the valid text answer (in this example it is "four").
2. The current task, however, does not allow us to distinguish between a model performing multistep reasoning and a model with access to a calculator / develop tree algorithms / run a script to figure out the answer.

## References

[1] Flanagan, D.P. & Dixon, S.G. (2014) The Cattell-Horn-Carroll theory of cognitive abilities. In C.R. Reynolds, K.J. Vannest and E. Fletcher-Janzen (eds.), Encyclopedia of Special Education. New York: Wiley Online.
