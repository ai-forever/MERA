# **LCS**

## Task Description

The longest common subsequence is an algorithmic task from [BIG-Bench](https://github.com/google/BIG-bench/tree/main/bigbench/benchmark_tasks/cs_algorithms/lcs). This problem consists of pairs of strings as input, and language models are expected to predict the length of the longest common subsequence between them correctly.

LCS is a prototypical dynamic programming problem and this task measures the model's ability to capture that approach.

**Keywords:** algorithms, numerical response, context length

**Authors:** Harsh Mehta, Behnam Neyshabur

### Motivation

Recently, large language models have started to do well on simple algorithmic tasks like few-shot arithmetic, so we want to extend this evaluation to more complicated algorithms.

## Dataset Description

### Data Fields

- `instruction` is a string containing instructions for the task and information about the requirements for the model output format;
- `inputs` is an example of two sequences to be compared;
- `outputs` is a string containing the correct answer, the length of the longest common subsequence;
- `meta` is a dictionary containing meta information:
    - `id` is an integer indicating the index of the example.

### Data Instances

Below is an example from the dataset:

```json
{
    "instruction": "Даны две строки: \"{inputs}\"\nОпределите длину их самой длинной общей подпоследовательности.",
    "inputs": "DFHFTUUZTMEGMHNEFPZ IFIGWCNVGEDBBTFDUNHLNNNIAJ",
    "outputs": "5",
    "meta": {
        "id": 186
    }
}
```

### Data Splits

The train set includes `320` examples, and the test set includes `500` examples.

### Prompts

6 prompts of varying difficulty were created for this task. Example:

`"Для двух строк: \"{inputs}\" найдите длину наибольшей общей подпоследовательности. Пересекающиеся символы должны идти в том же порядке, но могут быть разделены другими символами."`.

### Dataset Creation

Sequences of length in the range [4; 32) were generated with a Python script for train and test sets.

## Evaluation

### Metrics

The task is evaluated using Accuracy.

### Human Benchmark

The human benchmark is measured on a subset of size 100 (sampled with the same original distribution). The accuracy for this task is `0.56`.
