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

import numpy as np
import re

from lm_eval.base import Task, rf


class USE(Task):
    VERSION = 0
    DATASET_NAME = "use"

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
        if self.has_validation_docs():
            return map(self._process_doc, self.dataset["validation"])

    def test_docs(self):
        return map(self._process_doc, self.dataset["test"])

    @classmethod
    def _process_doc(cls, doc):
        task_type = doc["meta"]["type"]

        if task_type == "text":
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

        elif task_type == "multiple_choice_options_within_text":
            prompt = (
                doc["instruction"]
                .format(
                    task=doc["inputs"]["task"],
                    text=doc["inputs"]["text"],
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

    def fewshot_context(self, doc, num_fewshot, provide_description=None, rnd=None, description=None):
        assert rnd is not None, "A `random.Random` generator argument must be provided to `rnd`"
        assert not provide_description, (
            "The `provide_description` arg will be removed in future versions. To prepend "
            "a custom description to the context, supply the corresponding string via the "
            "`description` arg."
        )
        if provide_description is not None:
            # nudge people to not specify it at all
            print(
                "WARNING: provide_description is deprecated and will be removed in a future version in favor of description_dict"
            )

        description = description + "\n\n" if description else ""

        if num_fewshot == 0:
            labeled_examples = ""
        else:
            assert self.has_training_docs(), "The task must have a training set for few-shot learning."

            id_task = doc["id_task"]
            fewshotex = self.fewshot_examples(id_task=id_task, k=num_fewshot, rnd=rnd)

            labeled_examples = (
                "\n\n".join([self.doc_to_text(doc) + self.doc_to_target(doc) for doc in fewshotex]) + "\n\n"
            )

        example = self.doc_to_text(doc)
        return description + labeled_examples + example

    def fewshot_examples(self, id_task, k, rnd):
        if self._training_docs is None:
            self._training_docs = list(self.training_docs())

        training_docs = [training_doc for training_doc in self._training_docs if training_doc["id_task"] == id_task]
        return rnd.sample(training_docs, k)

    def doc_to_text(self, doc):
        return doc["query"]

    def doc_to_target(self, doc):
        return " " + doc["answers"]

    def should_decontaminate(self):
        return True

    def doc_to_decontamination_query(self, doc):
        return doc["query"]

    def construct_requests(self, doc, ctx):
        """Uses RequestFactory to construct Requests and returns an iterable of
        Requests which will be sent to the LM.

        :param doc:
            The document as returned from training_docs, validation_docs, or test_docs.
        :param ctx: str
            The context string, generated by fewshot_context. This includes the natural
            language description, as well as the few shot examples, and the question
            part of the document for `doc`.
        """
        return rf.greedy_until(ctx, {"until": [".", "\n"]})

    def process_results(self, doc, results):
        id_task = doc["id_task"]
        task_type = doc["task_type"]
        variant = doc["variant"]

        answer = doc["answers"]
        prediction = results[0]

        score = self.get_scores(task_type, id_task, answer, prediction)

        return {"grade_norm": (score, variant)}

    def multiple_choice_score(self, answer: str, prediction: str, is_task16=False) -> int:
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

    def matching_score(self, answer: str, prediction: str) -> int:
        pred = prediction.split(",")
        ans = answer.split(",")
        score = 0
        if len(ans) != len(pred):
            print('Format Error: The prediction must contain a string of 4 numbers separated by ","')
            return 0
        for idx, num in enumerate(ans):
            if num == pred[idx]:
                score += 1
        return score

    def text_score(self, answer: str, prediction: str) -> int:
        pred = re.sub(r"[\d+\W+]", "", prediction).lower()
        ans = answer.split(",")
        if pred in ans:
            return 1
        return 0

    def get_scores(self, task_type, id_task, answer, prediction):
        if task_type == "matching":
            score = self.matching_score(answer, prediction)
        elif task_type == "text":
            score = self.text_score(answer, prediction)
        else:
            is_task16 = False
            if id_task == "16":
                is_task16 = True
            score = self.multiple_choice_score(answer, prediction, is_task16)
        return score

    def overall_score(self, items, max_grade_point=34):
        overall_scores = {}
        for item in items:
            score, variant = item[0], item[1]
            if variant not in overall_scores:
                overall_scores[variant] = 0
            overall_scores[variant] += score

        average_overall_score = np.mean([score / max_grade_point for score in overall_scores.values()])
        return average_overall_score

    def higher_is_better(self):
        return {
            "grade_norm": True,
        }

    def aggregation(self):
        return {
            "grade_norm": self.overall_score,
        }
