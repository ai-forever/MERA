# **CheGeKa**

## Task Description

CheGeKa is a Jeopardy!-like the Russian QA dataset collected from the official Russian quiz database ChGK and belongs to the open-domain question-answering group of tasks. The dataset was created based on the [corresponding dataset](https://tape-benchmark.com/datasets.html#chegeka) from the TAPE benchmark [1].

**Keywords:** Reasoning, World Knowledge, Logic, Question-Answering, Open-Domain QA

**Authors:** Ekaterina Taktasheva, Tatiana Shavrina, Alena Fenogenova, Denis Shevelev, Nadezhda Katricheva, Maria Tikhonova, Albina Akhmetgareeva, Oleg Zinkevich, Anastasiia Bashmakova, Svetlana Iordanskaia, Alena Spiridonova, Valentina Kurenshchikova, Ekaterina Artemova, Vladislav Mikhailov

### Motivation

The task can be considered the most challenging in terms of reasoning, knowledge, and logic, as the task implies the QA pairs with a free response form (no answer choices); however, a long chain of causal relationships between facts and associations forms the correct answer.

## Dataset Description

### Data Fields

- `meta` is a dictionary containing meta-information about the example:
    - `id` is the task ID;
    - `author` is the author of the question;
    - `tour name` is the name of the game in which the question was used;
    - `tour_link` is a link to the game in which the question was used (None for the test set);
- `instruction` is an instructional prompt specified for the current task;
- `inputs` is a dictionary containing the following input information:
    - `text` is a text fragment with a question from the game “What? Where? When?";
    - `topic` is a string containing the category of the question;
- `outputs` is a string containing the correct answer to the question.

### Data Instances

Each instance in the dataset contains an instruction, a question, the topic of the question, the correct answer, and all the meta-information. Below is an example from the dataset:

```json
{
    "instruction": "Вы участвуете в викторине “Что? Где? Когда?”. Внимательно прочитайте вопрос из категории \"{topic}\" и ответьте на него.\nВопрос: {text}\nВ качестве ответа запишите только ваш вариант без дополнительных объяснений.\nОтвет:",
    "inputs": {
        "text": "В корриде, кроме быка, он тоже играет одну из главных ролей.",
        "topic": "\"ТОР\""
    },
    "outputs": "Тореадор",
    "meta": {
        "id": 7571,
        "author": "Максим Стасюк",
        "tour_name": "Своя игра. ШДК им. Рабиндраната Дебендранатовича Тагора",
        "tour_link": "https://db.chgk.info/tour/tagor02"
    }
}
```

### Data Splits

The dataset consists of 29376 training examples (train set) and 416 test examples (test set).

### Prompts

We use 10 different prompts written in natural language for this task. An example of the prompt is given below:

`"Вы участвуете в викторине “Что? Где? Когда?”. Категория вопроса: {topic}\nВнимательно прочитайте вопрос и ответьте на него: {text}\nОтвет:"`.

### Dataset Creation

The dataset was created using the corresponding dataset from the TAPE benchmark [1], which is, in turn, based on the original corpus of the CheGeKa game introduced in [2].

## Evaluation

### Metrics

The dataset is evaluated via two metrics: F1-score and Exact Match (EM).

### Human Benchmark

Human Benchmark was measured on a test set with Yandex.Toloka project with the overlap of 3 reviewers per task.

The F1-score / Exact Match results are `0.719` / `0.645`, respectively.

## References

[1] Taktasheva, Ekaterina, et al. "TAPE: Assessing Few-shot Russian Language Understanding." Findings of the Association for Computational Linguistics: EMNLP 2022. 2022.

[2] Mikhalkova, Elena, and Alexander A. Khlyupin. "Russian Jeopardy! Data Set for Question-Answering Systems." Proceedings of the Thirteenth Language Resources and Evaluation Conference. 2022.
