# **SimpleAr**

## Task Description

Simple arithmetic is a mathematical task from [BIG-Bench](https://github.com/google/BIG-bench/tree/main/bigbench/benchmark_tasks/simple_arithmetic). The task itself tests language models' basic arithmetic capabilities by asking them to perform n-digit addition for a range of n.

**Keywords:** arithmetic, example task, free response, mathematics, numerical response, zero-shot

### Motivation

The goal of the task is to analyze the ability of the model to solve simple mathematical addition tasks.

## Dataset Description

### Data Fields

- `instruction` is a string containing instructions for the task and information about the requirements for the model output format;
- `inputs` is the example of arithmetic expression;
- `outputs` is a string containing the correct answer of summation of two numbers;
- `meta` is a dictionary containing meta information:
    - `id` is an integer indicating the index of the example.

### Data Instances

Below is an example from the dataset:

```json
{
    "instruction": "Выполните арифметическую операцию.\n{inputs}",
    "inputs": "901 + 164 = ",
    "outputs": "1065",
    "meta": {
        "id": 679
    }
}
```

### Data Splits

The train set consists of `1000` examples of arithmetic expressions. The test set consists of `1000` examples of arithmetic expressions.

### Prompts

The number of prompts used for the task: 6. The following prompts for the task are used: 

```json
[
    "Вычислите математическое выражение:\n{inputs}",
    "Выполните арифметическую операцию.\n{inputs}",
    "Напишите ответ для математического выражения.\n{inputs}",
    "Сложите два числа:\n{inputs}",
    "Сложите первое и второе слагаемые: {inputs} и напишите ответ.",
    "Выполните арифметическую операцию. В качестве ответа напишите число, которое получается после ее выполнения.\n{inputs}"
]
```

### Dataset Creation

N-digit addition was created for n in the range [1;5] for both train and test sets.

## Evaluation

### Metrics

Accuracy is used for evaluation.

### Human Benchmark

The human benchmark is measured on a subset of size `200` (sampled with the same original distribution). The accuracy for this task is `1.0`.
