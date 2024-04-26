"""
The USE dataset: Unified State Exam in the Russian language.

The Unified State Exam (USE) in the Russian language dataset includes tasks similar to the
tasks from the Unified State Exam in the Russian language. The USE in the Russian
language is the mandatory form of graduation examination in Russian schools. The exam
is based on questions from the school curriculum. Tasks of the 'multiple_choice' type
require the choice of the correct answers from the proposed list. For tasks of the
'text' type, the answer is a word or a combination of words. Tasks of the 'matching'
type involve setting up correspondences between two lists.

Homepage: https://mera.a-ai.ru/
"""

from benchmark_tasks.custom_task import MERATask
from benchmark_tasks.use.utils import get_scores, overall_score


class USE(MERATask):
    VERSION = 0
    DATASET_NAME = "use"

    def __init__(
        self,
        data_dir=None,
        cache_dir=None,
        download_mode=None,
        config=None,
    ) -> None:
        super().__init__(data_dir, cache_dir, download_mode, config)
        self._config.generation_kwargs["until"] = [".", "\n"]

    def doc_to_text_without_instruction(self, doc):
        task_type = doc["meta"]["type"]
        if task_type in {"text", "multiple_choice_options_within_text"}:
            prompt = "Задание: {task}\n{text}\nОтвет:".format(**doc["inputs"])
        elif task_type == "matching":
            prompt = "Задание: {task}\nТекст: {text}\nРецензии: {additional_text}\nСписок терминов:\n{choices}\nОтвет:".format(
                **doc["inputs"]
            )
        elif task_type == "multiple_choice_based_on_text":
            prompt = "Задание: {task}\nТекст: {text}\nВарианты ответа:\n{choices}\nОтвет:".format(
                **doc["inputs"]
            )
        elif task_type == "multiple_choice_independent_options":
            prompt = "Задание: {task}\nВарианты ответа:\n{choices}\nОтвет:".format(
                **doc["inputs"]
            )
        return prompt

    def fewshot_examples(self, doc, k, rnd):
        docs = list(self.fewshot_docs())
        docs = [
            one_doc
            for one_doc in docs
            if one_doc["meta"]["id_task"] == doc["meta"]["id_task"]
        ]
        return rnd.sample(docs, k + 1)

    def process_results(self, doc, results):
        variant = doc["meta"]["variant"]
        if len(doc["outputs"]) > 0:
            id_task = doc["meta"]["id_task"]
            task_type = doc["meta"]["type"]
            answer = doc["outputs"]
            prediction = results[0]
            score = get_scores(task_type, id_task, answer, prediction)
            return {"grade_norm": (score, variant)}
        return {"grade_norm": (0.0, variant)}

    def aggregation(self):
        return {"grade_norm": overall_score}

    def higher_is_better(self):
        return {"grade_norm": True}
