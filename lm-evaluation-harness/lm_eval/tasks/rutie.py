"""
The Turing-test Interview Emulation (RuTie) dataset.

Turing-test Interview Emulation (RuTie) is a Russian-language test for simulation of the Turing test. The dataset imitates a coherent dialogue with the subject, where a set of questions on various topics is asked, and it is necessary to choose the most correct of two answer options for each question. The dataset checks the various categories, including string operations, world knowledge, lexic, ethics, math, and many more. The dialogue context and memory of the models is a special focus of the dataset.

Homepage: https://mera.a-ai.ru/
"""

import numpy as np
from lm_eval.base import rf, Task
from lm_eval.metrics import mean


class RUTIE(Task):
    VERSION = 0
    DATASET_NAME = "rutie"

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def training_docs(self):
        if self.has_training_docs():
            if self._training_docs is None:
                self._training_docs = sorted(list(self.dataset["train"]), key=lambda x: x["meta"]["question_id"])
            return self._training_docs

    def test_docs(self):
        if self.has_test_docs():
            return sorted(list(self.dataset["test"]), key=lambda x: x["meta"]["question_id"])

    def record_answer(self, idx: int, answer: int):
        if hasattr(self, "_labels_track"):
            self._labels_track[idx] = answer
        else:
            self._labels_track = dict()
            self._labels_track[idx] = answer

    def access_track(self):
        return self._labels_track if hasattr(self, "_labels_track") else {}

    def doc_to_text(self, doc):
        query_id = doc["meta"]["question_id"]
        if query_id == 0:
            instruction = doc["instruction"]
            inputs = doc["inputs"]
            inputs["context"] = ""
            query = instruction.format(**inputs)
            query = query.replace("\n\n", "\n")
            return query + "\n" + "Ответ:"
        instruction = self.test_docs()[0]["instruction"]
        inputs = doc["inputs"]
        context = [
            "{question}\n1. {choice1}\n2. {choice2}".format(
                **{
                    "question": elem["inputs"]["question"],
                    "choice1": elem["inputs"]["choice1"],
                    "choice2": elem["inputs"]["choice2"],
                }
            )
            + f"\nОтвет: {self.access_track()[i]}"
            for i, elem in enumerate(self.test_docs()[:query_id])
        ]
        inputs["context"] = "\n".join(context)
        result_call = instruction.format(**inputs) + "\n" + "Ответ:"
        return result_call

    def doc_to_target(self, doc):
        target = doc["outputs"]
        return " " + target

    def construct_requests(self, doc, ctx):
        ll_choice1, _ = rf.loglikelihood(ctx, " 1")
        ll_choice2, _ = rf.loglikelihood(ctx, " 2")
        return ll_choice1, ll_choice2

    def process_results(self, doc, results):
        gold = {"1": 0, "2": 1}[doc["outputs"]]
        pred = np.argmax(results)
        return {
            "acc": pred == gold,
        }

    def aggregation(self):
        return {"acc": mean}

    def higher_is_better(self):
        return {"acc": True}
