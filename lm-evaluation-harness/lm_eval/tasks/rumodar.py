"""
The Russian Modified Arithmetic (ruModAr) dataset.

Russian Modified Arithmetic is a mathematical task from Bigbench.
Each question in each subtask begins with a prompt and five examples of arithmetic
expressions with results. The sixth example is incomplete, the model's task is to
finish it correctly.

Homepage: https://mera.a-ai.ru/
"""

from lm_eval.base import Task, rf
from lm_eval.metrics import mean


class ruModAr(Task):
    VERSION = 0
    DATASET_NAME = "rumodar"

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
