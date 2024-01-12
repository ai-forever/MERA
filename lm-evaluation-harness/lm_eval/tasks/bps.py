"""
The Balanced Parentheses Sequence (BPS) dataset.

The Balanced Parentheses Sequence (BPS) is an algorithmic task from BIG-bench.
The primary purpose of this task is to measure language models' ability to learn CS
algorithmic concepts like stacks, recursion, or dynamic programming.

Homepage: https://mera.a-ai.ru/
"""
import re

from lm_eval.base import Task, rf
from lm_eval.metrics import mean


class BPS(Task):
    VERSION = 0
    DATASET_NAME = "bps"

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

    def doc_to_target(self, doc):
        return " " + doc["outputs"]

    def custom_format(self, input_str, **kwargs):
        pattern = r"\{([^}]*)\}"
        matches = re.finditer(pattern, input_str)
        for match in matches:
            placeholder = match.group(0)
            key = match.group(1)
            if key in kwargs and kwargs[key]:
                input_str = input_str.replace(placeholder, str(kwargs[key]))
        return input_str

    def doc_to_text(self, doc):
        prompt = self.custom_format(doc["instruction"], inputs=doc["inputs"])
        return prompt

    def doc_to_text_without_instruction(self, doc):
        inputs = doc["inputs"]
        return inputs.strip()

    def should_decontaminate(self):
        return True

    def doc_to_decontamination_query(self, doc):
        return doc["inputs"]

    def construct_requests(self, doc, ctx):
        ll_yes, _ = rf.loglikelihood(ctx, " 1")
        ll_no, _ = rf.loglikelihood(ctx, " 0")

        return ll_yes, ll_no

    def process_results(self, doc, results):
        ll_yes, ll_no = results
        ans = int(doc["outputs"])
        gold = True if ans == 1 else False

        acc = 1.0 if (ll_yes > ll_no) == gold else 0.0

        return {"acc": acc}

    def aggregation(self):
        return {"acc": mean}

    def higher_is_better(self):
        return {"acc": True}
