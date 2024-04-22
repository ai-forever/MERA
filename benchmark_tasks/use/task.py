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
import re

from benchmark_tasks.custom_task import MERATask
from lm_eval.api.instance import Instance
from lm_eval.api.metrics import mean
from lm_eval.utils import positional_deprecated


class USE(MERATask):
    VERSION = 0
    DATASET_NAME = "use"

    OUTPUT_TYPE = "generate_until"

    def __init__(
        self,
        data_dir=None,
        cache_dir=None,
        download_mode=None,
        config=None,
    ) -> None:
        super().__init__(data_dir, cache_dir, download_mode, config)
        self._config.generation_kwargs["until"] = [".", "\n"]

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def training_docs(self):
        if self._training_docs is None:
            self._training_docs = list(map(self._process_doc, self.dataset["train"]))
        return self._training_docs

    def validation_docs(self):
        if self.has_validation_docs():  # TODO: find when this check is important
            return map(self._process_doc, self.dataset["validation"])
        return []

    def test_docs(self):
        # random shuffling with fixed seed, same as MERA 1.1.0
        self.rnd.seed(42)
        docs = list(map(self._process_doc, self.dataset["test"]))
        self.rnd.shuffle(docs)
        return docs

    def _process_doc(self, doc):
        task_type = doc["meta"]["type"]

        if task_type in {"text", "multiple_choice_options_within_text"}:
            prompt = (
                doc["instruction"]
                .format(
                    task=doc["inputs"]["task"],
                    text=doc["inputs"]["text"],
                )
                .strip()
            )

        elif task_type == "matching":
            prompt = (
                doc["instruction"]
                .format(
                    task=doc["inputs"]["task"],
                    text=doc["inputs"]["text"],
                    additional_text=doc["inputs"]["additional_text"],
                    choices=doc["inputs"]["choices"],
                )
                .strip()
            )

        elif task_type == "multiple_choice_based_on_text":
            prompt = (
                doc["instruction"]
                .format(
                    task=doc["inputs"]["task"],
                    text=doc["inputs"]["text"],
                    choices=doc["inputs"]["choices"],
                )
                .strip()
            )

        elif task_type == "multiple_choice_independent_options":
            prompt = (
                doc["instruction"]
                .format(
                    task=doc["inputs"]["task"],
                    choices=doc["inputs"]["choices"],
                )
                .strip()
            )

        out_doc = {
            "meta": {"id": doc["meta"]["id"]},
            "query": prompt,
            "answers": doc["outputs"],
            "task_type": task_type,
            "id_task": doc["meta"]["id_task"],
            "variant": doc["meta"]["variant"],
        }
        return out_doc

    def doc_to_text(self, doc):
        return doc["query"]

    def doc_to_target(self, doc):
        return " " + doc["answers"]

    def construct_requests(self, doc, ctx, **kwargs):
        ans = Instance(
            request_type="generate_until",
            doc=doc,
            arguments=(ctx, self.config["generation_kwargs"]),
            idx=0,
            **kwargs,
        )
        return ans

    def process_results(self, doc, results):
        id_task = doc["id_task"]
        task_type = doc["task_type"]
        variant = doc["variant"]

        answer = doc["answers"]
        prediction = results[0]

        score = self._get_scores(task_type, id_task, answer, prediction)

        return {"grade_norm": (score, variant)}

    def aggregation(self):
        return {"grade_norm": self._overall_score}

    def higher_is_better(self):
        return {"grade_norm": True}

    def fewshot_examples(self, id_task, k: int, rnd):
        if self._training_docs is None:
            self._training_docs = list(self.training_docs())

        training_docs = [
            training_doc
            for training_doc in self._training_docs
            if training_doc["id_task"] == id_task
        ]
        return rnd.sample(training_docs, k)

    @positional_deprecated
    def fewshot_context(
        self,
        doc,
        num_fewshot,
        rnd=None,
        description=None,
    ):
        if rnd is None:
            rnd = self.rnd
        assert (
            rnd is not None
        ), "A `random.Random` generator argument must be provided to `rnd`"

        description = description + self.config.fewshot_delimiter if description else ""

        if num_fewshot == 0:
            labeled_examples = ""
        else:
            assert (
                self.has_training_docs()
            ), "The task must have a training set for few-shot learning."

            id_task = doc["id_task"]
            fewshotex = self.fewshot_examples(id_task=id_task, k=num_fewshot, rnd=rnd)

            labeled_examples = (
                self.config.fewshot_delimiter.join(
                    [
                        self.doc_to_text(doc) + self.doc_to_target(doc)
                        for doc in fewshotex
                    ]
                )
                + self.config.fewshot_delimiter
            )

        example = self.doc_to_text(doc)
        return description + labeled_examples + example

    def _multiple_choice_score(
        self, answer: str, prediction: str, is_task16=False
    ) -> int:
        pred = prediction.split(",")
        ans = answer.split(",")
        if is_task16:
            while len(pred) < len(ans):
                pred.append(-1)
            return max(
                0,
                len(set.intersection(set(ans), set(pred))) - len(pred) + len(ans),
            )
        else:
            ans = set(ans)
            pred = set(pred)
            return int(len(set.intersection(ans, pred)) == len(ans) == len(pred))

    def _matching_score(self, answer: str, prediction: str) -> int:
        pred = prediction.split(",")
        ans = answer.split(",")
        score = 0
        if len(ans) != len(pred):
            print(
                'Format Error: The prediction must contain a string of 4 numbers separated by ","'
            )
            return 0
        for idx, num in enumerate(ans):
            if num == pred[idx]:
                score += 1
        return score

    def _text_score(self, answer: str, prediction: str) -> int:
        pred = re.sub(r"[\d+\W+]", "", prediction).lower()
        ans = answer.split(",")
        if pred in ans:
            return 1
        return 0

    def _get_scores(self, task_type, id_task, answer, prediction):
        if task_type == "matching":
            score = self._matching_score(answer, prediction)
        elif task_type == "text":
            score = self._text_score(answer, prediction)
        else:
            is_task16 = False
            if id_task == "16":
                is_task16 = True
            score = self._multiple_choice_score(answer, prediction, is_task16)
        return score

    def _overall_score(self, items, max_grade_point=34):
        overall_scores = {}
        for item in items:
            score, variant = item[0], item[1]
            if variant not in overall_scores:
                overall_scores[variant] = 0
            overall_scores[variant] += score

        average_overall_score = mean(
            [score / max_grade_point for score in overall_scores.values()]
        )
        return average_overall_score
