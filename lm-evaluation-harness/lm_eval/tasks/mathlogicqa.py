"""
The MathLogicQA dataset.

MathLogicQA is a QA dataset with multiple-choice math questions consisting systems
of equations, proportional relationships, and comparisons.

Homepage: https://mera.a-ai.ru/
"""

import numpy as np
from lm_eval.metrics import mean

from lm_eval.base import MultipleChoiceTask


class MathLogicQA(MultipleChoiceTask):
    VERSION = 0
    DATASET_NAME = "mathlogicqa"

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def training_docs(self):
        if self._training_docs is None:
            self._training_docs = list(map(self._process_doc, self.dataset["train"]))
        return self._training_docs

    def test_docs(self):
        return map(self._process_doc, self.dataset["test"])

    def doc_to_target(self, doc):
        if isinstance(doc["gold"], int):
            gold = doc["choices"][doc["gold"]]
        else:
            gold = ""
        return " " + gold

    def _process_doc(self, doc):
        choices = ["A", "B", "C", "D"]
        if doc["outputs"]:
            gold = choices.index(doc["outputs"])
        else:
            gold = ""

        prompt = (
            doc["instruction"]
            .format(
                text=doc["inputs"]["text"],
                option_a=doc["inputs"]["option_a"],
                option_b=doc["inputs"]["option_b"],
                option_c=doc["inputs"]["option_c"],
                option_d=doc["inputs"]["option_d"],
            )
            .strip()
        )

        doc["query"] = prompt
        doc["choices"] = choices
        doc["gold"] = gold
        return doc

    def doc_to_text(self, doc):
        return doc["query"]

    def doc_to_text_without_instruction(self, doc):
        prompt = "{text}\nA) {option_a}\nB) {option_b}\nC) {option_c}\nD) {option_d}\nОтвет:".format(
            **doc["inputs"]
        ).strip()
        return prompt

    def should_decontaminate(self):
        return True

    def doc_to_decontamination_query(self, doc):
        return doc["query"]

    def process_results(self, doc, results):
        gold = doc["gold"]
        acc = 1.0 if np.argmax(results) == gold else 0.0
        return {
            "acc": acc,
        }

    def higher_is_better(self):
        return {
            "acc": True,
        }

    def aggregation(self):
        return {
            "acc": mean,
        }
