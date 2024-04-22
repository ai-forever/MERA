"""
The Balanced Parentheses Sequence (BPS) dataset.

The Balanced Parentheses Sequence (BPS) is an algorithmic task from BIG-bench.
The primary purpose of this task is to measure language models' ability to learn CS
algorithmic concepts like stacks, recursion, or dynamic programming.

Homepage: https://mera.a-ai.ru/
"""

import re

from numpy import argmax

from benchmark_tasks.custom_task import MERATask
from lm_eval.api.instance import Instance
from lm_eval.api.metrics import mean


class BPS(MERATask):
    VERSION = 0
    DATASET_NAME = "bps"

    OUTPUT_TYPE = "loglikelihood"

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
        return []

    def _custom_format(self, input_str, **kwargs):
        pattern = r"\{([^}]*)\}"
        matches = re.finditer(pattern, input_str)
        for match in matches:
            placeholder = match.group(0)
            key = match.group(1)
            if key in kwargs and kwargs[key]:
                input_str = input_str.replace(placeholder, str(kwargs[key]))
        return input_str

    def doc_to_text(self, doc):
        prompt = self._custom_format(doc["instruction"], inputs=doc["inputs"])
        return prompt

    def doc_to_text_without_instruction(self, doc):
        inputs = doc["inputs"]
        return inputs.strip()

    def doc_to_target(self, doc):
        return " " + doc["outputs"]

    def construct_requests(self, doc, ctx, **kwargs):
        choices = [0, 1]
        return [
            Instance(
                request_type=self.OUTPUT_TYPE,
                doc=doc,
                arguments=(ctx, " {}".format(choice)),
                idx=idx,
                **kwargs,
            )
            for idx, choice in enumerate(choices)
        ]

    def process_results(self, doc, results):
        if len(doc["outputs"]) > 0:
            results = [
                res[0] for res in results
            ]  # only retain loglikelihoods, discard is_greedy
            gold = int(doc["outputs"])

            acc = 1.0 if argmax(results) == gold else 0.0

            return {"acc": acc}
        return {"acc": 0.0}  # if no label provided (test answers are secret)

    def aggregation(self):
        return {"acc": mean}

    def higher_is_better(self):
        return {"acc": True}
