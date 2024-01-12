# **BPS**

## Task Description

The balanced sequence is an algorithmic task from [BIG-bench](https://github.com/google/BIG-bench/tree/main/bigbench/benchmark_tasks/cs_algorithms/valid_parentheses). The primary purpose of this task is to measure language models' ability to learn CS algorithmic concepts like stacks, recursion, or dynamic programming.

Each subtask contains a parentheses sequence. The model's goal is to correctly predict whether the sequence is balanced.

An input string is valid if:

1. Open brackets must be closed by the same type of brackets.
2. Open brackets must be closed in the correct order.
3. Every close bracket has a corresponding open bracket of the same type.

**Keywords:** algorithms, numerical response, context length, parantheses, binary answer

**Authors:** Harsh Mehta, Behnam Neyshabur

### Motivation

Algorithms are a way to extrapolate examples and are some of the most concise descriptions of a pattern. In that sense, the ability of language models to learn them is a prominent measure of intelligence.

## Dataset Description

### Data Fields

- `instruction` is a string containing instructions for the task and information about the requirements for the model output format;
- `inputs` is an example of the parentheses sequence;
- `outputs` is a string containing the correct answer: “1” if the parentheses sequence is valid, “0” otherwise;
- `meta` is a dictionary containing meta information:
    - `id` is an integer indicating the index of the example.

### Data Instances

Below is an example from the dataset:

```json
{
    "instruction": "На вход подается последовательность скобок: \"{inputs}\"\nНеобходимо ответить сбалансирована ли данная последовательность. Если последовательность сбалансирована - выведите 1, иначе 0",
    "inputs": "[ ] } { [ ] { ) [ } ) ) { ( ( ( ) ] } {",
    "outputs": "0",
    "meta": {
        "id": 40
    }
}
```

### Data Splits

The train consists of 250 examples, and the test set includes 1000 examples.

### Prompts

8 prompts of varying difficulty were created for this task. Example:

`"Проверьте, сбалансирована ли входная последовательность скобок.\n"{inputs}"\nВыведите 1, если да и 0 в противном случае. Сперва закрывающей скобкой своего типа должна закрываться последняя из открытых скобок, и лишь потом соответствующей закрывающей скобкой может закрываться та, что была открыта перед ней."`.

### Dataset Creation

The parentheses sequences of the length 2, 4, 8, 12, 20 were generated with the following distribution: `{20: 0.336, 12: 0.26, 8: 0.24, 4: 0.14, 2: 0.024}` for the train set and `{20: 0.301, 12: 0.279, 8: 0.273, 4: 0.121, 2: 0.026}` for the test set.

## Evaluation

### Metrics

The task is evaluated using Accuracy.

### Human benchmark

The human benchmark is measured on a subset of size 100 (sampled with the same original distribution). The accuracy for this task is `1.0`.
