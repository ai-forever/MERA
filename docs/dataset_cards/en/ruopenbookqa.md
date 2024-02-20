# **ruOpenBookQA**

## Task Description

RuOpenBookQA is a QA dataset with multiple-choice elementary-level science questions that probe understanding of 1k+ core science facts. The dataset is built with automatic translation of the original English dataset [1]. and manual validation by a few authors; a test set was created from scratch. The set is a part of the [TAPE](https://tape-benchmark.com/) benchmark [2] that was redesigned to an instruction-based format and filtered.

**Keywords:** Logic, World Knowledge, Common Sense

**Authors:** Ekaterina Taktasheva, Tatiana Shavrina, Alena Fenogenova, Denis Shevelev, Nadezhda Katricheva, Maria Tikhonova, Albina Akhmetgareeva, Oleg Zinkevich, Anastasiia Bashmakova, Svetlana Iordanskaia, Alena Spiridonova, Valentina Kurenshchikova, Ekaterina Artemova, Vladislav Mikhailov

### Motivation

RuOpenBookQA is mainly based on the work [1]. The original OpenBookQA is a new kind of question-answering dataset modeled after open-book exams for assessing human understanding of a subject. It consists of 5957 multiple-choice elementary-level science questions, which probe the understanding of a small “book” of 1326 core science facts and the application of these facts to novel situations. Answering OpenBookQA questions requires additional broad common knowledge not contained in the book. The questions, by design, are answered incorrectly by both a retrieval-based algorithm and a word co-occurrence algorithm. The Russian version of the set is much smaller but covers the topics representative of the Russian language.

## Dataset Description

### Data Fields

- `meta` is a dictionary containing meta-information about the dataset:
    - `id` is the unique number of a sample;
- `instruction` is an instructional prompt specified for the current task;
- `inputs` is a dictionary containing the following input information:
    - `text` is the question of the test;
    - `option_a` is the option A;
    - `option_b` is the option B;
    - `option_c` is the option C;
    - `option_d` is the option D;
- `outputs` are the results, can be the following string values: "A", "B", "C", "D".

### Data Instances

Below is an example from the dataset:

```
{
    "instruction": "{text}\nA. {option_a}\nB. {option_b}\nC. {option_c}\nD. {option_d}\nКакой ответ является правильным? В качестве ответа запишите только букву верного варианта: A, B, C или D без дополнительных объяснений.\nОтвет: ",
    "inputs": {
        "text": "Что вращается вокруг своей оси?",
        "option_a": "океаны",
        "option_b": "ветры",
        "option_c": "шар голубой",
        "option_d": "люди"
    },
    "outputs": "C",
    "meta": {
        "id": "14-167"
    }
}
```

### Data Splits

The number of training and test samples in the dataset is 2338 and 400, respectively.

### Prompts

We prepared ten different prompts of various difficulties for this task.

Examples of the prompt are given below:

`"{text}\nA. {option_a}\nB. {option_b}\nC. {option_c}\nD. {option_d}\nКакой ответ является правильным? В качестве ответа запишите только букву верного варианта: A, B, C или D без дополнительных объяснений.\nОтвет:"`,

`"Опираясь на логику и общеизвестные факты, ответьте на вопрос: {text}\nA) {option_a}\nB) {option_b}\nC) {option_c}\nD) {option_d}\nВ качестве ответа запишите только букву верного варианта: A, B, C или D без дополнительных объяснений.\nОтвет:"`.

### Dataset Creation

The questions are taken from the original OpenBookQA dataset, created via multi-stage crowdsourcing and partial expert filtering. The dataset mainly consists of automatic translation of the English OpenBookQA and human validation and correction. The samples that are part of the BIG-Bench set were excluded from the TAPE version of the dataset and rewritten in instruction-based format.

## Evaluation

### Metrics

The dataset is evaluated using Average Macro F1 and Accuracy.

### Human Benchmark

Human Benchmark was measured on a test set with Yandex.Toloka project with the overlap of 3 reviewers per task.

Results for Average Macro F1 and Accuracy are `0.875` / `0.865`, respectively.

## References

[1] Todor Mihaylov, Peter Clark, Tushar Khot, and Ashish Sabharwal. 2018. Can a Suit of Armor Conduct Electricity? A New Dataset for Open Book Question Answering. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pages 2381–2391, Brussels, Belgium. Association for Computational Linguistics.

[2] Taktasheva Ekaterina et al. "TAPE: Assessing Few-shot Russian Language Understanding." Findings of the Association for Computational Linguistics: EMNLP 2022.
