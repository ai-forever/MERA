# **ruTiE**

## Task Description

Turing-test Interview Emulation (ruTiE) — is a Russian-language test for the simulation of the Turing test. The dataset simulates a coherent dialogue with the subject, where the subject is asked a set of questions on various topics, and the subject needs to choose the most correct of two answer options for each question. The topics of the questions cover different categories on different aspects of the Turing test. The questions imply that the subject (model) fully remembers the context of the dialogue and may have a reference to the previous parts. The peculiarity is that the answers are not necessarily presented in a purely binary format when only one is correct and the second one is false. It is necessary to process both answers and choose the one closer to the correct answer, further complicating the solution and introducing an additional step of reasoning.

**Keywords:** memory, context, logic, knowledge about the world, common sense

### Motivation

The first version of the dataset is a full-fledged long dialogue, during which the model answers a number of interrelated (or not) questions.
The dataset explores:

1. The length of the model's context and memory. To do this, the dataset has special metadata fields indicating whether the question is contextual. If the question is independent and can be asked in the exact wording with the same answer options without reducing the possibility of answering correctly, then the metadata of the question in the use_context field is False; if the question is based on the context of the previous conversation and cannot be fully understood and interpreted without this context, then in the metadata use_context field is True.
2. To an initial extent — the capabilities of models in several categories of the direction of thinking that are necessary **to solve the emulation of the Turing Test (the categories are selected to develop any subsequent dataset of this type, taking into account the default possibility of their identification):**
    - `sentiment` (emotional coloring);
    - `intent` (the intentions of the participants in the dialogue or the characters described in the question);
    - `style` (the style of the text; for example, it belongs to the clerical style, certain authors' style, etc.);
    - `humor` (the presence of humor, the ability to determine how funny the text is);
    - `irony` (irony and its detection);
    - `facts` (factual accuracy, honesty);
    - `profanity` (profane/obscene vocabulary);
    - `adult_content` (adult content);
    - `text_metrics` (simple symbolic/mathematical operations, count the number of letters, consonants, vowels, voiced, deaf, count words with the letter "o", solve the simplest mathematical example given in the text or digital form, etc.);
    - `language_structure` (ability to perceive word forms and structural-formative relations in a sentence: inflections, text consistency, spelling/syntax, etc.);
    - `topic_modelling` (ability to determine the subject of the text);
    - `multilanguage` (cross-lingual and multilingual tasks);
    - `algorithmic_transformations`  (different texts-shifters, sorting characters, adding/removing parts, duplications, and so on).

3. The ability of the model to distinguish between the basic classes of problems that are necessary to solve the emulation of the Turing test (they make up the dataset):
    - `world` (knowledge about the world);
    - `math` (symbolic calculations, mathematics, logic);
    - `memory` (activation of the directed long-term memory function of the model, including some information and a question in memory, extracting some information from long-term memory);
    - `reasoning` (conclusions, causal relationships);
    - `strings` (operations with strings: anagrams, sub-sequence counting, etc.);
    - `spell` (questions related to spelling and the composition of words);
    - `games and rules` (the ability to handle systems based on rules: games, including chess problems, traffic rules, puzzles, and similar systems);
    - `sound` (text questions on sound modality and audio form of words, sounds, accents, rhyme, and audio on text);
    - `shape` (questions on associative connections, “awareness” of the forms of the real world through symbolic systems and graphic objects);
    - `lexis` (knowledge of the language system, linguistic knowledge, word formation: hyperonyms/hyponyms, kinship terms, etc.);
    - `emotion` (emotion recognition);
    - `ethics` (ethical tasks);
    - `trap` (trick questions, contextual or logical-linguistic traps leading to the wrong answer, knocking off the course of the dialogue).

## Dataset Description

### Data Fields

- `instruction` is a string containing instructions for the task;
- `inputs` is a dictionary that contains the following information:
    - `question` is the question;
    - `choice1` is a possible answer `1`;
    - `choice2` is a possible answer `2`;
- `outputs` is the answer information, possible options: `1` or `2`;
- `meta` is a dictionary containing meta information about the dataset:
    - `dialog_id` is the dialogue id (from zero);
    - `question_id` is the serial id of the question in the dialogue;
    - `category` is the question category;
    - `use_context` is True if one needs context to answer the question (else False);
    - `turing_imitation`is the simulation class.

### Data Instances

One complete example of a task is one dialogue. Formally, the dialogue looks like this:

```json
[
    {
        "instruction": "Вам дан диалог, в котором необходимо продолжить реплики. Учитывая контекст диалога, и два варианта ответа на реплику (вопрос) ответьте на последний вопрос.\n{context}\n{question}\n1. {choice1}\n2. {choice2}\nКакой ответ наиболее правильный?",
        "inputs": {
            "question": "Сколько ног у человека?",
            "choice1": "Две",
            "choice2": "Четыре"
        },
        "outputs": "1",
        "meta": {
            "dialog_id": 0,
            "question_id": 0,
            "category": ["world"],
            "use_context": false,
            "turing_imitation": ["facts"]
        }
    },
    {
        "instruction": "Вам дан диалог, в котором необходимо продолжить реплики. Учитывая предыдущий контекст диалога, и два варианта ответа на вопрос ответьте на последний.\n{context}\n{question}\n1) {choice1}\n2) {choice2}\nКакой ответ наиболее правильный?",
        "inputs": {
            "question": "А у муравья?",
            "choice1": "Две",
            "choice2": "Шесть"
        },
        "outputs": "2",
        "meta": {
            "dialog_id": 0,
            "question_id": 1,
            "category": ["world", "memory"],
            "use_context": true,
            "turing_imitation": ["facts"]
        }
    }
]
```

To run the model on the dataset, you need to consistently submit replies by `question_id` one after another and add the model's response to the context in the `context` field of the instruction.

For example:

- Take the dialog `dialog_id=0`.
- Submit questions to the model consistently by `question_id` and get the result.
- The `context` field on the first question is an empty string, with each subsequent question of the dialog, `{question}\nОтвет:` is written in the `context` field and the answer from the previous replies, the answer is written in the form of text, which is taken from the answer option from the fields `choice1` or `choice2`. So, the instruction for the second reply of the dialogue, if we answered the first question that a Person has four legs (choice 2), looks like this:

    ```markdown
    Вам дан диалог, в котором необходимо продолжить реплики. Учитывая предыдущий контекст диалога, и два варианта ответа на вопрос ответьте на последний.
    Сколько ног у человека?
    Четыре
    {question}
    1) {choice1}
    2) {choice2}
    Какой ответ наиболее правильный?
    ```

- Next, it is necessary to substitute by analogy the question and answer options of the following ordinal example from the dataset and send them to the model:

    ```markdown
    Вам дан диалог, в котором необходимо продолжить реплики. Учитывая предыдущий контекст диалога, и два варианта ответа на вопрос ответьте на последний.
    Сколько ног у человека?
    Четыре
    А у муравья?
    1) Две
    2) Шесть
    Какой ответ наиболее правильный?
    ```

- And so forth until the end of the dialogue.

**Please follow the sequence of replies! Strictly by `question_id`; otherwise the entire dataset will be solved incorrectly.**

### Data Splits

The first version of the dataset consists of only one long dialogue of length 430 for the training public set, and one dialogue of length 430 for the test dataset.

### Prompts

The instruction (prompt) is sent to the entire dataset, and not to each replica. Several different prompts were selected, such as:

"Вам дан диалог, в котором необходимо продолжить реплики. Учитывая контекст диалога, и два варианта ответа на реплику (вопрос) ответьте на последний вопрос.\n{context}\n{question}\n1. {choice1}\n2. {choice2}\n
Какой ответ наиболее правильный?".

### Dataset Creation

The dataset was collected manually by annotators and then validated.

## Evaluation

### Metrics

The dataset is a full-fledged long dialogue, with binary tasks on various topics. The closed test set is one such dialogue, the quality of which is considered to be the Accuracy metric, the average for the dialogue.

### Human benchmark

Accuracy for this task is `0.977`.

## Limitations

There is no balance of classes by meta-categories. The dataset will be updated with new dialogues in the future.

## References

[1] Pinar Saygin, A., Cicekli, I. & Akman, V. Turing Test: 50 Years Later. *Minds and Machines* 10, 463–518 (2000).

[2] Stanford Encyclopedia of Philosophy. "The Turing Test.".
