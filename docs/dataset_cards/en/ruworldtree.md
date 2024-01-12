# **ruWorldTree**

## Task Description

RuWorldTree is a QA dataset with multiple-choice elementary-level science questions that evaluate the understanding of core science facts. The set is created based on the original English WorldTree dataset that provides a corpus of explanation graphs for elementary science questions. The set is a part of the TAPE benchmark that was redesigned to an instruction-based format and filtered.

**Keywords:** Logic, Reasoning, World Knowledge, Facts

**Authors:** Ekaterina Taktasheva, Tatiana Shavrina, Alena Fenogenova, Denis Shevelev, Nadezhda Katricheva, Maria Tikhonova, Albina Akhmetgareeva, Oleg Zinkevich, Anastasiia Bashmakova, Svetlana Iordanskaia, Alena Spiridonova, Valentina Kurenshchikova, Ekaterina Artemova, Vladislav Mikhailov

### Motivation

The WorldTree design was initially proposed in [1]. The WorldTree dataset starts the triad of the Reasoning and Knowledge tasks. The data includes the corpus of factoid utterances of various kinds, complex factoid questions, and a corresponding causal chain of facts from the corpus, resulting in a correct answer. The Russian RuWorldTree is an analog of WorldTree and is a part of the [TAPE](https://tape-benchmark.com/) benchmark [2] that was redesigned to instruction format and filtered.

## Dataset Description

### Data Fields

- `meta` is meta-information about the task:
    - `id` is the original task id from the TAPE benchmark;
    - `exam_name` is information about the source exam;
    - `school_grade` is the difficulty level;
    - `knowledge_type` is the type of knowledge one needs to solve the task;
- `instruction` is the instructional prompt specified for the current task;
- `inputs` is a dictionary containing the following input information:
    - `question` is the question of the test;
    - `option_a` is the option A;
    - `option_b` is the option B;
    - `option_c` is the option C;
    - `option_d` is the option D;
- `outputs` is the results, can be the following string values: "A", "B", "C", "D".

### Data Instances

Below is an example from the dataset:

```json
{
    "instruction": "{text}\nA. {option_a}\nB. {option_b}\nC. {option_c}\nD. {option_d}\nКакой ответ является правильным? В качестве ответа запишите только букву верного варианта: A, B, C или D без дополнительных объяснений.\nОтвет: ",
    "inputs": {
        "question": "Какие из следующих структур развиваются у лягушки, когда она превращается из головастика во взрослую лягушку?",
        "option_a": "глаза",
        "option_b": "сердце",
        "option_c": "легкие",
        "option_d": "хвост"
    },
    "outputs": "C",
    "meta": {
        "id": 5,
        "exam_name": "MCAS",
        "school_grade": 5,
        "knowledge_type": "PROCESS"
    }
}
```

### Data Splits

The number of training and the test examples is 115 and 525, respectively.

### Prompts

We prepared ten different prompts of various difficulties for this task.

Examples of the prompt are given below:

`"{question}\nA. {option_a}\nB. {option_b}\nC. {option_c}\nD. {option_d}\nВыберите ответ из списка.\nОтвет:"`,

`"Опираясь на логику и общеизвестные факты, ответьте на вопрос: {question}\nA) {option_a}\nB) {option_b}\nC) {option_c}\nD) {option_d}\nОтвет:"`.

### Dataset Creation

The questions for the dataset are taken from the original WorldTree dataset, which was sourced from the AI2 Science Questions V2 corpus, consisting of both standardized exam questions from 12 US states, and the AI2 Science Questions Mercury dataset, a set of questions licensed from a student assessment entity. The dataset mainly consists of automatic translation of the English WorldTree Corpus and human validation and correction. The samples that are part of the Big-Bench set were excluded from the TAPE version of the dataset and rewritten in instruction-based format.

## Evaluation

### Metrics

The dataset is evaluated using Average Macro F1 and Accuracy.

### Human Benchmark

Human Benchmark was measured on a test set with Yandex.Toloka project with overlap: 3 reviewers per task.

Results for Average Macro F1 and Accuracy are `0.838` / `0.837`, respectively.

## References

[1] Peter Jansen, Elizabeth Wainwright, Steven Marmorstein, and Clayton Morrison. 2018. WorldTree: A Corpus of Explanation Graphs for Elementary Science Questions supporting Multi-hop Inference. In Proceedings of the Eleventh International Conference on Language Resources and Evaluation (LREC 2018), Miyazaki, Japan. European Language Resources Association (ELRA).

[2] Taktasheva, Ekaterina, et al. "TAPE: Assessing Few-shot Russian Language Understanding." Findings of the Association for Computational Linguistics: EMNLP 2022.
