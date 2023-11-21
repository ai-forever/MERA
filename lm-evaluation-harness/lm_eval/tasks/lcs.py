"""
The Largest Common Subsequence (LCS) dataset.

The longest common subsequence (LCS) is an algorithmic task from Bigbench. This problem
consists of pairs of strings as input, and language models are expected to correctly
predict the length of the longest common subsequence between the strings.

Homepage: https://mera.a-ai.ru/
"""

import inspect

from lm_eval.base import Task, rf
from lm_eval.metrics import mean
from numpy import argmax


class LCS(Task):
    VERSION = 0
    DATASET_NAME = "lcs"

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

    def doc_to_text(self, doc):
        prompt = doc["instruction"].format(inputs=doc["inputs"]).strip()
        return prompt

    def doc_to_text_without_instruction(self, doc):
        inputs = doc["inputs"]
        return inputs.strip()

    def should_decontaminate(self):
        return True

    def doc_to_decontamination_query(self, doc):
        return doc["inputs"]

    def construct_requests(self, doc, ctx):
        ll_0 = rf.loglikelihood(ctx, " 0")
        ll_1 = rf.loglikelihood(ctx, " 1")
        ll_2 = rf.loglikelihood(ctx, " 2")
        ll_3 = rf.loglikelihood(ctx, " 3")
        ll_4 = rf.loglikelihood(ctx, " 4")
        ll_5 = rf.loglikelihood(ctx, " 5")
        ll_6 = rf.loglikelihood(ctx, " 6")
        ll_7 = rf.loglikelihood(ctx, " 7")
        ll_8 = rf.loglikelihood(ctx, " 8")
        ll_9 = rf.loglikelihood(ctx, " 9")
        return [ll_0, ll_1, ll_2, ll_3, ll_4, ll_5, ll_6, ll_7, ll_8, ll_9]

    def process_results(self, doc, results):
        ll_0, ll_1, ll_2, ll_3, ll_4, ll_5, ll_6, ll_7, ll_8, ll_9 = results
        gold = int(doc["outputs"])
        max_arg = argmax([ll_0, ll_1, ll_2, ll_3, ll_4, ll_5, ll_6, ll_7, ll_8, ll_9])
        acc = max_arg == gold

        return {"acc": acc}

    def aggregation(self):
        return {
            "acc": mean,
        }

    def higher_is_better(self):
        return {"acc": True}
