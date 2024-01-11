# **RCB**

## Task Description

The Russian Commitment Bank is a corpus of naturally occurring discourses whose final sentence contains a clause-embedding predicate under an entailment canceling operator (question, modal, negation, antecedent of conditional). It was first introduced in the [Russian SuperGLUE](https://russiansuperglue.com/tasks/task_info/RCB) benchmark [1].

**Keywords:** Reasoning, Common Sense, Causality, Textual Entailment

**Authors:** Shavrina Tatiana, Fenogenova Alena, Emelyanov Anton, Shevelev Denis, Artemova Ekaterina, Malykh Valentin, Mikhailov Vladislav, Tikhonova Maria, Evlampiev Andrey

### Motivation

The dataset allows to evaluate how well the models solve a logical text entailment. The dataset is constructed in such a way as to take into account discoursive characteristics. This dataset in the Russian SuperGLUE benchmark is one of the few for which there is still a significant gap between model and human estimates.

## Dataset Description

### Data Fields

Each dataset sample represents some text situation:

- `instruction` is an instructional prompt specified for the current task;
- `inputs` is a dictionary containing the following input information:
    - `premise` is a text situation;
    - `hypothesis` is a text of the hypothesis for which it is necessary to define whether it can be inferred from the hypothesis or not;
- `outputs` are the results: can be the following string values: 1 — hypothesis follows from the situation, 2 — hypothesis contradicts the situation, or 3 — hypothesis is neutral;
- `meta` is meta-information about the task:
    - `genre` is where the text was taken from;
    - `verb` is the action by which the texts were selected;
    - `negation` is the flag;
    - `id` is the id of the example from the dataset.

### Data Instances

Below is an example from the dataset:

```json
{
    "instruction": "Приведено описание ситуации и гипотеза. Ситуация: \"{premise}\" Гипотеза: \"{hypothesis}\". Определи отношение гипотезы к ситуации, выбери один из трех вариантов: 1 — гипотеза следует из ситуации, 2 — гипотеза противоречит ситуации, 3 — гипотеза независима от ситуации. В ответ напиши только цифру 1, 2 или 3, больше ничего не добавляй.",
    "inputs": {
        "premise": "Сумма ущерба составила одну тысячу рублей. Уточняется, что на место происшествия выехала следственная группа, которая установила личность злоумышленника. Им оказался местный житель, ранее судимый за подобное правонарушение.",
        "hypothesis": "Ранее местный житель совершал подобное правонарушение."
    },
    "outputs": "1",
    "meta": {
        "verb": "судить",
        "negation": "no_negation",
        "genre": "kp",
        "id": 0
    }
}
```

### Data Splits

The dataset contains 438 training samples, 220 validation samples, and 438 test samples. The number of sentences for the entire set is 2715, and the total number of tokens is 3.7 · 10^3.

### Prompts

We prepare 10 different prompts of various difficulties for this task.

An example of the prompt is given below:

`"Ситуация: \"{premise}\" Гипотеза: \"{hypothesis}\". Определи логическое отношение гипотезы к ситуации, возможен один из трех вариантов: 1 — гипотеза следует из ситуации, 2 — гипотеза противоречит ситуации, 3 — гипотеза независима от ситуации. В ответ напиши только цифру 1, 2 или 3, больше ничего не добавляй."`.

### Dataset creation

The dataset is an instruction-based version of the Russian SuperGLUE benchmark RCB. The set was filtered out of Taiga (news, literature domains) [4] with several rules and the extracted passages were manually post-processed. Final labeling was conducted by three of the authors. The original dataset corresponds to CommitmentBank dataset [2, 3].

## Evaluation

### Metrics

The metrics are Accuracy and Average Macro F1.

### Human Benchmark

Human Benchmark was measured on a test set with Yandex.Toloka project with the overlap of 3 reviewers per task.

Average Macro F1 and Accuracy results are `0.68` / `0.702`, respectively.

## References

[1] Tatiana Shavrina, Alena Fenogenova, Emelyanov Anton, Denis Shevelev, Ekaterina Artemova, Valentin Malykh, Vladislav Mikhailov, Maria Tikhonova, Andrey Chertok, and Andrey Evlampiev. 2020. RussianSuperGLUE: A Russian Language Understanding Evaluation Benchmark. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 4717–4726, Online. Association for Computational Linguistics.

[2] Marie-Catherine de Marneffe, Mandy Simons, and Judith Tonhauser (2019). The CommitmentBank: Investigating projection in naturally occurring discourse. Proceedings of Sinn und Bedeutung 23.

[3] Wang A. et al. Superglue: A stickier benchmark for general-purpose language understanding systems //Advances in Neural Information Processing Systems. – 2019. – С. 3261-3275.

[4] Shavrina, Tatiana, and Olga Shapovalova. "To the methodology of corpus construction for machine learning:“Taiga” syntax tree corpus and parser." Proceedings of “CORPORA-2017” International Conference. 2017.
