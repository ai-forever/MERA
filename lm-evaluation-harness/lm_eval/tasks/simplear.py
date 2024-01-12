"""
The Simple arithmetic (SimpleAr) dataset.

Simple arithmetic - a mathematical task based on the set from Bigbench. The task itself
tests language models' basic arithmetic capabilities by asking them to perform n-digit
addition for a range of n.

Homepage: https://mera.a-ai.ru/
"""

from lm_eval.base import Task, rf
from lm_eval.metrics import mean


class SimpleAr(Task):
    VERSION = 0
    DATASET_NAME = "simplear"

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def training_docs(self):
        if self.has_training_docs():
            if self._training_docs is None:
                self._training_docs = list(self.dataset["train"])
            return self._training_docs

    def test_docs(self):
        if self.has_test_docs():
            return list(self.dataset["test"])

    def doc_to_text(self, doc):
        instruction = doc["instruction"]
        inputs = doc["inputs"]
        return instruction.format(inputs=inputs).strip()

    def doc_to_text_without_instruction(self, doc):
        inputs = doc["inputs"]
        return inputs.strip()

    def doc_to_target(self, doc):
        target = doc["outputs"]
        return " " + target

    def construct_requests(self, doc, ctx):
        completion = rf.greedy_until(ctx, {"until": ["\n"]})
        return completion

    def process_results(self, doc, results):
        completion = results[0]
        completion1 = str(completion).strip()
        out = str(doc["outputs"])
        if completion1 == out:
            res = 1
        else:
            res = 0
        return {"acc": res}

    def aggregation(self):
        return {"acc": mean}

    def higher_is_better(self):
        return {"acc": True}
