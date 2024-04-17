# **RWSD**

## Task Description

Russian Winograd Schema Dataset (RWSD), or the Winograd schema, is a task in which each example contains a sentence with two selected phrases. The task is to define whether they are used in the same sense or not. The schema takes its name from a well-known example by Terry Winograd.

The set would then be presented as a challenge for AI programs like the Turing test. The strengths of the challenge are that it is clear-cut, in that the answer to each schema is a binary choice; vivid, in that it is evident to non-experts that a program that fails to get the correct answers has severe gaps in its understanding; and difficult, in that it is far beyond the current state of the art.

**Keywords:** Logic and Reasoning, World Knowledge, Common Sense

**Authors:** Shavrina Tatiana, Fenogenova Alena, Emelyanov Anton, Shevelev Denis, Artemova Ekaterina, Malykh Valentin, Mikhailov Vladislav, Tikhonova Maria,  Evlampiev Andrey

### Motivation

A Winograd schema is a pair of sentences that differ in only one or two. The dataset will test the models' ability to identify and resolve syntactic ambiguities using logic and knowledge about the world—the classic standard set by Terry Winograd [1,2]. The dataset was first introduced in [the Russian SuperGLUE](https://russiansuperglue.com/tasks/task_info/RWSD) benchmark [3], and it's one of the sets for which there is still a significant gap between model and human estimates.

## Dataset Description

### Data Fields

- `instruction` is instructions with the description of the task;
- `inputs` is a dictionary containing the following input information:
    - `text` is the initial situation, usually a sentence that contains some syntactic ambiguity;
    - `span1_index` and `span_text` are a span and a text representing an object indication in the text situation (referent);
    - `span2_index` and `span2_text` are (anaphors) a span and a text representing a pronoun (or another word) that you need to understand which object it refers to;
- `outputs` is a string containing the correct answer text ("Yes" or "No");
- `meta` is a dictionary containing meta-information about the dataset:
    - `id` is an integer, the unique number of a sample.

### Data Instances

Below is an example from the dataset:

```json
{
    "instruction": "Дан небольшой текст: \"{text}\"\nОбъект из текста: \"{span1_text}\"\nТекстовый фрагмент, который может относиться к двум или нескольким объектам в тексте, включая указанный: \"{span2_text}\"\nНужно ответить, относится ли фрагмент к названному объекту. Ответь Да, если относится, или Нет.",
    "inputs": {
        "text": "Женя поблагодарила Сашу за помощь, которую она оказала.",
        "span1_index": 2,
        "span1_text": "Сашу",
        "span2_index": 6,
        "span2_text": "она оказала"
     },
    "outputs": "Да",
    "meta": {
        "id": 11
    }
}
```

### Data Splits

The dataset includes 606 training, 204 validation, and 260 test examples.

### Prompts

We prepare 10 different prompts of various difficulty for this task.

An example of the prompt is given below:

`"Перед тобой текст: \"{text}\"\nОпираясь на текст, скажи, относится ли местоимение во фрагменте текста \"{span2_text}\" к объекту фрагмента \"{span1_text}\"? В качестве ответа выдай одно слово: Да, если относится, или Нет, если не относится. Напиши только правильный ответ без дополнительных объяснений."`.

### Dataset creation

The set was created based on the Russian SuperGLUE [3] dataset, and the test part was verified and augmented to preserve the class balance: 130 examples for each class. All examples for the original set from Russian SuperGLUE have been converted to the instructional format.

## Evaluation

### Metrics

The metric used for the evaluation of this task is Accuracy.

### Human Benchmark

Human assessment was carried out using the Yandex.Toloka platform with annotator overlap equal to 5. The final human Accuracy is `0.835`.

## References

[1] Levesque, H. J., Davis, E., & Morgenstern, L. (2012). The winograd schema challenge. In 13th International Conference on the Principles of Knowledge Representation and Reasoning, KR 2012 (pp. 552-561). (Proceedings of the International Conference on Knowledge Representation and Reasoning). Institute of Electrical and Electronics Engineers Inc.

[2] Wang A. et al. Superglue: A stickier benchmark for general-purpose language understanding systems //Advances in Neural Information Processing Systems. – 2019. – С. 3261-3275.

[3] Tatiana Shavrina, Alena Fenogenova, Emelyanov Anton, Denis Shevelev, Ekaterina Artemova, Valentin Malykh, Vladislav Mikhailov, Maria Tikhonova, Andrey Chertok, and Andrey Evlampiev. 2020. RussianSuperGLUE: A Russian Language Understanding Evaluation Benchmark. In *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, pages 4717–4726, Online. Association for Computational Linguistics.
