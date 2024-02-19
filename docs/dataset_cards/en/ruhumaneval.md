# **ruHumanEval**

## Task Description

Russian HumanEval (ruHumanEval) is the Russian analogue of the original HumanEval dataset, created to evaluate the ability of language models to generate code in the Python programming language to solve simple problems.
The dataset is aimed at measuring the functional correctness of code generation based on information from the function's documentation lines — a text description of the function's operation and several examples of results for different input data.

**Keywords:** PLP, programming, Python

### Motivation

This task tests the ability of models to generate simple Python programs based on a description (condition) in natural language. Since large models have in their training corpus a proportion of texts (programs) written in various programming languages, they are assumed to have the ability to understand and write code for simple tasks.

## Dataset Description

### Data Fields

- `instruction` is a string containing instructions for the task;
- `inputs` is a dictionary that contains the following information:
    - `function` is a line containing the function signature, as well as its docstring in the form of an unwritten function;
    - `tests` is a list of dictionaries that contain input data of test cases for a given task (variants of input data on which the final function code is tested);
- `outputs` is a two-dimensional array of size (n_samples, n_tests), where n_samples is the number of samples required to calculate the pass@k metric, n_tests is the number of test cases in tests; each list in the outputs is the same and contains correct answers to all test cases;
- `meta` is a dictionary containing meta information:
    - `id` is an integer indicating the index of the example;
    - `canonical_solution` is the canonical solution;
    - `entry_point` is the function name.

### Data Instances

Below is an example from the dataset:

```json
{
    "instruction": "На вход подается функция с описанием в виде строки docstring. В соответствии с описанием вам необходимо реализовать функцию на основе шаблона:\n{function}",
    "inputs": {
        "function": "
                    def greatest_common_divisor(a: int, b: int) -> int:
                        '''Верните наибольший общий делитель двух целых чисел a и b.
                        Примеры:
                            greatest_common_divisor(3, 5)
                            1
                            greatest_common_divisor(25, 15)
                            5
                        '''
            ",
        "tests": [{"a": 3, "b": 7}, {"a": 10, "b": 15}, {"a": 49, "b": 14}, {"a": 144, "b": 60}]
    },
    "outputs": [1, 5, 7, 12],
    "meta": {
        "id": 666,
        "canonical_solution": "
                def query_gcd(a: int, b: int) -> int:
                        return a if b == 0 else query_gcd(b, a % b)
                    return query_gcd(a, b)",
        "entry_point": "greatest_common_divisor"
    }
}
```

### Data Splits

The training set contains `164` tasks with test cases and answers from the original dataset. The test set contains `164` tasks with closed answers specially collected by authors for this benchmark. For the test set, we provide only test cases without outputs and solutions.

### Prompts

For this task 10 prompts of varying difficulty were created. Example:

`"На вход подается функция с описанием в виде строки docstring. В соответствии с описанием вам необходимо реализовать функцию на основе шаблона:\n{function}"`.

### Dataset Creation

The training set was translated into Russian from the dataset [openai_humaneval](https://huggingface.co/datasets/openai_humaneval). We corrected typos in the docstring and canonical solutions and made the corrections described in [2]. The test set was manually collected from open source and filtered to avoid data leakage during training.

## Evaluation

### Metrics

The model is evaluated using the `pass@k` metric, which is computed as follows:

$$ pass@k:=\mathbb{E}_{problems}\left[1-\frac{\binom{n-c}{k}}{\binom{n}{k}}\right] $$

Notation: *n* is the total number of generated solution options, *c* is the number of solutions that are correct, *k* is the selected indicator, how many options are taken into account.

To calculate `pass@k`, `n ≥ k` solutions are generated for each problem and are run through test cases (we use n = 200 and k ≤ 100 and an average of 10 test cases per problem). Then, the number of the correct solutions is calculated (`c ≤ n`). The solution is considered to be correct if it passes all test cases. That means the result of running solutions on test cases should be equal to the correct answers (outputs) for one problem. Such an evaluation process yields an unbiased score.

### Human evaluation
Dataset includes algorithmic problems that require knowledge of the Python programming language, which is too complex skill for an average annotator. All problems have strict solutions, therefore all human evaluation metrics are taken as `1.0`.

## References

[1] Chen, Mark, et al. "Evaluating large language models trained on code." arXiv preprint arXiv:2107.03374 (2021).

[2] Jiawei Liu, Chunqiu Steven Xia, Yuyao Wang, Lingming Zhang Is Your Code Generated by ChatGPT Really Correct? Rigorous Evaluation of Large Language Models for Code Generation.
