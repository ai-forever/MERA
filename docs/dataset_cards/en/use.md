# **USE**

## Task Description

The dataset comprises tasks on the "The Russian language" subject from the Unified State Exam. The Unified State Exam (USE) is a form of mandatory state final exam for graduates of Russian schools. The content of the exam may vary depending on the year. In this article, the tasks from the 2019 exam are used.

### Motivation

Analyze the ability of the model to solve the tasks from the exam on the subject of “The Russian language", as well as output the answer in a pre-defined format. This exam aims to test proficiency in the norms of the modern Russian language and the ability to analyze information from texts.

## Dataset Description

The exam consists of two parts. Part 1 contains 26 tasks with a short answer. Part 2 consists of essay writing. In this article, the tasks of Part 1 will be analyzed.

Each task is designed to measure proficiency in the specific elements of the Russian language. Thus, the elements of the Russian language tested in the Unified State Exam are:

- proficiency in the norms of the modern Russian language — orthoepic (stress placement) (task 4); vocabulary and speech (tasks 3, 5, 6, 24); grammar (morphology and syntax) (tasks 7, 8); knowledge of the basic rules of Russian spelling (tasks 9-15) and punctuation (tasks 16-21)
- proficiency in the text analysis (tasks 1–3, 22–26);
- description and narration in Russian (tasks 1, 24, 26).

The exam consists of the following types of short answer tasks:

- **text** — open-question task that requires writing down a self-formulated correct answer (tasks 2, 4-7, 13, 14, 24)
- **multiple_choice** — task that requires to choose one or more correct answers from the given answer options. (tasks 1, 3, 8-12, 15-23, 25);
- **matching** — task to match objects in the text with answer options (task 26).

In the original exam, in task 8, the student must match two lists: a list with grammatical errors and a list with sentences in which they are made. As part of our benchmark, this task was divided into several tasks of the multiple_choice type, in which each error represents a separate task. Thus, from a given list of sentences, it is necessary to find a sentence in which a particular grammatical error is made.

In our dataset, **multiple_choice** type tasks are divided into three more subtypes:

- **based_on_text** — there is text and a question to it with answer options.
- **options_within_text** — there is text and numbers in it; a participant needs to select the correct options from these numbers.
- **independent_options** — there is a task and answer options.

Answers to tasks in Part 1 are recorded on the answer form as a number, a word (several words), or a sequence of numbers written without spaces, commas, and other additional marks.

The benchmark defines the following requirements for the model response format:

- for tasks of the **multiple_choice** and **matching** types, the response is a string containing a number or sequence of numbers, separated by commas without spaces;
- for tasks of the **text** type, the answer is a string containing a word or several words without spaces, commas or other additional characters.

### Data Fields

- `instruction` is a string containing instructions for the task and information about the requirements for the model output format;
- `inputs` is a dictionary containing model input data:
    - `task` is a string containing the text of the question;
    - `text` is a string containing text related to the question;
    - `choices` is a string containing options for answering the question;
    - `additional_text` is a string containing additional text required to complete the task;
- `outputs` is a string containing the correct answers;
- `meta` is a dictionary containing meta-information necessary for calculating metrics:
    - `id` is an integer indicating the number of the example from the dataset;
    - `id_task` is a string indicating the number of the task from the variant;
    - `variant` is an integer indicating the exam option;
    - `score` is an integer containing the maximum score that can be obtained for correct execution;
    - `type` is a string containing information about the type of task.

For some keys from the inputs field, the values are empty strings if this information is not used to solve the task.

### Data Instances

Example from the dataset for *text* task:

```json
{
	"instruction": "Прочитайте задание и выполните его. Ответом к заданию является слово или несколько слов без пробелов, запятых и других дополнительных символов.\nЗадание: {task}\n{text}\nОтвет: ",
	"inputs": {
		"task": "Отредактируйте предложение: исправьте лексическую ошибку, исключив лишнее слово. Выпишите это слово (пару слов).",
		"text": "Внезапный холодный мороз повредил урожай салата.",
		"choices": "",
		"additional_text": ""
	},
	"outputs": "холодный",
	"meta": {
		"id_task": "6",
		"variant": 25,
		"score": 1,
		"type": "text",
		"id": 740
	}
}
```

Example from the dataset for *matching* task:

```json
{
	"instruction": "Прочитайте текст и выполните задание по тексту.\nТекст: {text}\nЗадание: {task}\nРецензии: {additional_text}\nСписок терминов:\n{choices}\nВ ответе запишите цифры через запятую без пробелов в порядке, соответствующем буквам АБВГ.\nОтвет: ",
	"inputs": {
		"task": "Прочитайте фрагмент рецензии, составленной на основе приведённого выше текста. В этом фрагменте рассматриваются языковые особенности текста. Некоторые термины, использованные в рецензии, пропущены. Пропуск в рецензии обозначен как «_________». Вставьте на места пропусков (А, Б, В, Г) цифры, соответствующие номеру термина из списка.",
		"additional_text": "«Каждая строчка, каждое слово Дмитрия Шеварова пронизаны искренним уважением к личности Пушкина. Эмоциональное, неравнодушное отношение автора выражено с помощью та кого синтаксического средства, как (А)_________ (предложения 7, 17), а также лексических — (Б)_________ («подлец», «пошляк», «сплетник») и (В)_________ («честь и имя» в предложениях 18—19), (Г)_________ («звон... стали в слове...», в предложении 3, «разряд... силы» в предложении 8, «слово... отливалось в свинец» в предложении 13) придают особую образность тексту Д. Шеварова».",
		"text": "(1)В письме к жене 18 мая 1836 года Пушкин удивлялся: откуда взялись эти благоразумные молодые люди, «которым плюют в глаза, а они утираются» вместо того, чтобы защитить свою честь? (2)Иногда кажется, что мы вышли из шинелей именно этих людей. (3)Звон упругой стали более не слышится нам в слове честь.\n (4)Откроем словарь Даля, чтобы вспомнить, во имя чего ставилась на карту жизнь, полная великих надежд и гениальных замыслов. (5) Итак, «честь — внутреннее нравственное достоинство человека, доблесть, честность, благородство души и чистая совесть». (6) И тут же примеры: «Человек незапятнанной чести. По чести... Уверяю вас честью. Поступок, несовместимый с честью... Знал бы ты честь... Поле чести... Честь моя требует крови...».\n (7)Дуэль! (8)Только этот разряд убийственной силы мог стремительно восстановить нравственное равновесие. (9)Подлец знал, что его подлость может быть наказана не взиманием штрафа через год по приговору суда, а сегодня вечером. (10)Самое позднее — завтра утром. (11)Пошляк не говорил двусмысленностей вслух, остерегаясь немедленного возмездия. (12)Сплетник вынужден был осторожничать.(13)В грозном свете дуэльных правил слово быстро отливалось в свинец.\n (14)А как же Пушкин? (15) Какая непоправимая и бессмысленная гибель... (16)Да, непоправимая, но не бессмысленная. (17)Да, «невольник чести», но ведь чести! (18)3а год до дуэли Пушкин писал графу Репнину: «Как дворянин и отец семейства, я должен блюсти честь и имя, которое оставлю моим детям». (19) Вот и всё, что остаётся детям: честь и имя. (20)Всё остальное им не нужно, всё остальное — неважно. (21)Очевидно, нам ещё многое предстоит пережить и передумать, чтобы вернуться к пониманию этой истины.\n(По Д. Шеварову)",
		"choices": "1) метафоры\n2) сравнительный оборот\n3) гипербола\n4) эмоционально-оценочные слова\n5) эпитеты\n6) риторический вопрос\n7) вопросно-ответная форма изложения\n8) лексический повтор\n9) риторическое восклицание"
	},
	"outputs": "4,9,2,8",
	"meta": {
		"id_task": "26",
		"variant": 3,
		"score": 4,
		"type": "matching",
		"id": 866
	}
}
```

Example from the dataset for *multiple_choice_based_on_text* task:
```
{
	"instruction": "Прочитайте текст и выполните задание по тексту. Ответом к заданию является число или последовательность чисел, перечисленных через запятую без пробелов.\nТекст: {text}\nЗадание: {task}\nВарианты ответа:\n{choices}\nОтвет: ",
	"inputs": {
		"task": ".Прочитайте фрагмент словарной статьи, в которой приводятся значения слова СОБСТВЕННЫЙ. Определите значение, в котором это слово употреблено в первом (1) предложении текста. Выпишите цифру, соответствующую этому значению в приведённом фрагменте словарной статьи",
		"text": "(1) Растущий оброк и барщина тормозили развитие собственного хозяйства крестьян. (2) Частые неурожаи обрекали сельских тружеников на полуголодное существование. (3) <…> усиление эксплуатации крепостных крестьян обусловливало застой и рутинность производительных сил в деревне.СОБСТВЕННЫЙ",
		"choices": "1. Принадлежащий кому-чему-н. по праву собственности.\n2. Свой, личный. Видеть собственными глазами. В собственные руки.\n3. Находящийся в непосредственном ведении, распоряжении, подчинении кого-чего-н. С. корреспондент.\n4. Буквальный, настоящий. В. собственном смысле слова\n5. Свойственный только чему-н., без посторонних добавлений",
		"additional_text": ""
	},
	"outputs": "2",
	"meta": {
		"id_task": "3",
		"variant": 23,
		"score": 1,
		"type": "multiple_choice_based_on_text",
		"id": 53
    }
}
```

Example from the dataset for *multiple_choice_options_within_text* task:

```json
{
	"instruction": "Прочитайте текст задания и выполните его указания. Ответом к заданию является число или последовательность чисел, перечисленных через запятую без пробелов.\nЗадание: {task}\nТекст: {text}\nОтвет: ",
	"inputs": {
		"task": "Укажите все цифры, на месте которых пишется НН.",
		"text": "Пират, облитый серебря(1)ым лу(2)ым светом, долго стоял на пороге и напряжё(3)о слушал",
		"choices": "",
		"additional_text": ""
	},
	"outputs": "2,3",
	"meta": {
		"id_task": "15",
		"variant": 17,
		"score": 1,
		"type": "multiple_choice_options_within_text",
		"id": 137
	}
}
```

Example from the dataset for *multiple_choice_independent_options* task:

```json
{
	"instruction": "Прочитайте текст задания и выполните его указания. Ответом к заданию является число или последовательность чисел, перечисленных через запятую без пробелов.\nЗадание: {task}\nВарианты ответа:\n{choices}\nОтвет: ",
	"inputs": {
		"task": "Укажите варианты ответов, в которых в обоих словах одного ряда пропущена одна и та же буква.Запишите номера ответов.",
		"choices": "1) невид..мый, разгон..шься\n2) отрасл..вой, мах..нький\n3) груш..вый, нищ..та\n4) леч..щий, молч..щий\n5) ткан..вый, лист..к",
		"text": "",
		"additional_text": ""
	},
	"outputs": "1,3",
	"meta": {
		"id_task": "12",
		"variant": 26,
		"score": 1,
		"type": "multiple_choice_independent_options",
		"id": 592
	}
}
```

Since task 8 was divided into 5 separate tasks, for this task the `id_task` field also contains information about the number of the question within this task, for example, `id_task` contains the value `8_1`.

### Data Splits

Train set consists of 110 incomplete versions of exam tests. In total, it included `2622` tasks: 94 tasks of the **matching** type, 1815 tasks of the **multiple_choice** type, 713 tasks of the **text** type.

Dev set consists of 30 complete versions of exam tests. In total, it included `900` tasks: 30 tasks of the **matching** type, 630 tasks of the **multiple_choice** type, 240 tasks of the **text** type.

Test set consists of 30 complete versions of exam tests. In total, it included `900` tasks: 30 tasks of the **matching** type, 630 tasks of the **multiple_choice** type, 240 tasks of the **text** type.

### Prompts
Number of prompts per sub-tasks multiplied by the number of sub-tasks 3x5. Example for sub-task:
```json
{
    "multiple_choice": {
        "based_on_text": [
            "Прочитайте текст и выполните задание по тексту. Ответом к заданию является число или последовательность чисел, перечисленных через запятую без пробелов.\nТекст: {text}\nЗадание: {task}\nВарианты ответа:\n{choices}\nОтвет:"
        ],
        "options_within_text": [
            "Прочитайте текст задания и выполните его указания. Ответом к заданию является число или последовательность чисел, перечисленных через запятую без пробелов.\nЗадание: {task}\nТекст: {text}\nОтвет:"
        ],
        "independent_options": [
            "Прочитайте текст задания и выполните его указания. Ответом к заданию является число или последовательность чисел, перечисленных через запятую без пробелов.\nЗадание: {task}\nВарианты ответа:\n{choices}\nОтвет:"
        ]
    },
    "text": [
        "Прочитайте задание и выполните его. Ответом к заданию является слово или несколько слов без пробелов, запятых и других дополнительных символов в нижнем регистре.\nЗадание: {task}\n{text}\nОтвет:"
    ],
    "matching": [
        "Прочитайте текст и выполните задание по тексту.\nТекст: {text}\nЗадание: {task}\nРецензии: {additional_text}\nСписок терминов:\n{choices}\nВ ответе запишите цифры через запятую без пробелов в порядке, соответствующем буквам АБВГ.\nОтвет:"
    ]
}
```

### Dataset Creation

Examples for train and dev sets were collected from open sources with examples of tasks from the Unified State Exam in the Russian language.

For the closed test, experts prepared 30 unique exam options based on the same methodological standard.

1. https://rus-ege.sdamgia.ru/
2. https://yandex.ru/tutor/

## Evaluation

### Metrics

For the text and multiple_choice tasks from the test sample, for which the answer is a string containing several words or a string containing a sequence of numbers, all possible combinations of these words and numbers are used when calculating metrics. For these tasks from the train and dev sets, only one answer combination is presented.

**Grading System**

- For correct completion of tasks 1–7, 8–15, 17–25, the examinee receives 1 point. For an incorrect answer or lack thereof, 0 points are given.
- For completing task 16, you can score from 0 to 2 points. The answer that contains all the numbers from the standard and no other numbers is considered correct. 1 point is given if: one of the numbers indicated in the answer does not correspond to the standard; one of the numbers specified in the answer template is missing. In all other cases, 0 points are given.
- For completing task 26, you can score from 0 to 4 points. The answer that contains all the numbers from the standard and no other numbers is considered correct. For each correctly indicated number corresponding to a number from the list, the examinee receives 1 point.

**Final Metric**

The final primary score is calculated as the sum of points for all tasks of the option. The maximum number of primary points for Part 1 of the exam is 34.

The final metric `grade_norm` is the average normalized primary score across all versions, where normalization is done by dividing the final primary score by the maximum possible number of points (i.e. 34).

The calculation of the final primary score, as well as the final `grade_norm` metric, is carried out only for the validation and test parts of the dataset, which consist of full exam versions of the USE.

### Human Benchmark

The tasks from the 2019 exam are used. Since the content of the exam, the complexity of the tasks, as well as the assessment system changes depending on the year, the average primary score of graduates for completing Part 1 of the Unified State Exam in the Russian language in 2019 is used as a human assessment.

Based on [official statistics](https://doc.fipi.ru/ege/analiticheskie-i-metodicheskie-materialy/2019/russkiy_yazyk_2019.pdf) the average primary score for Part 1 was `23.835` out of 34 points, value `grade_norm` was `0.701`.
