# **PARus**

## Task Description

The choice of Plausible Alternatives for the Russian language (PARus) evaluation provides researchers with a tool for assessing progress in open-domain commonsense causal reasoning.

Each question in PARus is composed of a premise and two alternatives, where the task is to select the alternative that more plausibly has a causal relation with the premise. The correct alternative is randomized, so the expected randomly guessing performance is 50%. The dataset was first proposed in [Russian SuperGLUE](https://russiansuperglue.com/tasks/task_info/PARus) [1] and is an analog of the English [COPA](https://people.ict.usc.edu/~gordon/copa.html) [2] dataset that was constructed as a translation of the English COPA dataset from [SuperGLUE](https://super.gluebenchmark.com/tasks) [3] and edited by professional editors. The data split from COPA is retained.

**Keywords:** reasoning, commonsense, causality, commonsense causal reasoning

**Authors:** Shavrina Tatiana, Fenogenova Alena, Emelyanov Anton, Shevelev Denis, Artemova Ekaterina, Malykh Valentin, Mikhailov Vladislav, Tikhonova Maria, Evlampiev Andrey

### Motivation

The dataset tests the models’ ability to identify cause-and-effect relationships in the text and draw conclusions based on them. The dataset is first presented from the [RussianSuperGLUE](https://russiansuperglue.com/tasks/task_info/PARus) leaderboard, and it’s one of the sets for which there is still a significant gap between model and human estimates.

## Dataset Description

### Data Fields

Each dataset sample represents a `premise` and two `options` for continuing situations depending on the task tag: cause or effect.

- `instruction` is a prompt specified for the task, selected from different pools for cause and effect;
- `inputs` is a dictionary containing the following input information:
    - `premise` is a text situation;
    - `choice1` is the first option;
    - `choice2` is the second option;
- `outputs` are string values "1" or "2";
- `meta` is meta-information about the task:
    - `task` is a task class: cause or effect;
    - `id` is the id of the example from the dataset.

### Data Instances

Below is an example from the dataset:

```
{
    "instruction": "Дано описание ситуации:\n'{premise}'\nи два фрагмента текста:\n1. {choice1}\n2. {choice2}\nОпредели, какой из двух фрагментов является следствием описанной ситуации? Ответь одной цифрой 1 или 2, ничего не добавляя.",
    "inputs": {
        "premise": "Власти пообещали сохранить в тайне личность жертвы преступления.",
        "choice1": "Жертва изо всех сил пыталась вспомнить подробности преступления.",
        "choice2": "Они скрывали имя жертвы от общественности."
    },
    "outputs": "2",
    "meta": {
        "task": "effect",
        "id": 72
    }
}
```

### Data Splits

The dataset consists of `400` train samples, `100` dev samples, and `500` private test samples. The number of sentences in the whole set is `1000`. The number of tokens is 5.4 · 10^3.

### Prompts

We prepare 10 different prompts of various difficulty for this task. Prompts are presented separately for the `cause` and for the `effect`, e.g.:

For cause: `"Дано описание ситуации:\n'{premise}'\nи два фрагмента текста:\n1. {choice1}\n2. {choice2}\nОпредели, какой из двух фрагментов является причиной описанной ситуации? Ответь одной цифрой 1 или 2, ничего не добавляя."`.

For effect: `"Дано описание ситуации:\n'{premise}'\nи два фрагмента текста:\n1. {choice1}\n2. {choice2}\nОпредели, какой из двух фрагментов является следствием описанной ситуации? Ответь одной цифрой 1 или 2, ничего не добавляя."`.

### Dataset Creation

The dataset was taken initially from the RussianSuperGLUE set and reformed in an instructions format. All examples for the original set from RussianSuperGLUE were collected from open news sources and literary magazines, then manually cross-checked and supplemented by human evaluation on Yandex.Toloka.

Please, be careful! [PArsed RUssian Sentences](https://parus-proj.github.io/PaRuS/parus_pipe.html) is not the same dataset. It’s not a part of the Russian SuperGLUE.

## Evaluation

### Metrics

The metric for this task is Accuracy.

### Human Benchmark

Human-level score is measured on a test set with Yandex.Toloka project with the overlap of 3 reviewers per task. The Accuracy score is `0.982`.

## References

[1] Original COPA paper: Roemmele, M., Bejan, C., and Gordon, A. (2011) Choice of Plausible Alternatives: An Evaluation of Commonsense Causal Reasoning. AAAI Spring Symposium on Logical Formalizations of Commonsense Reasoning, Stanford University, March 21-23, 2011.

[2] Wang A. et al. Superglue: A stickier benchmark for general-purpose language understanding systems //Advances in Neural Information Processing Systems. – 2019. – С. 3261-3275.

[3] Tatiana Shavrina, Alena Fenogenova, Emelyanov Anton, Denis Shevelev, Ekaterina Artemova, Valentin Malykh, Vladislav Mikhailov, Maria Tikhonova, Andrey Chertok, and Andrey Evlampiev. 2020. RussianSuperGLUE: A Russian Language Understanding Evaluation Benchmark. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 4717–4726, Online. Association for Computational Linguistics.
