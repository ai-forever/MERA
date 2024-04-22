"""
The Largest Common Subsequence (LCS) dataset.

The longest common subsequence (LCS) is an algorithmic task from Bigbench. This problem
consists of pairs of strings as input, and language models are expected to correctly
predict the length of the longest common subsequence between the strings.

Homepage: https://mera.a-ai.ru/
"""

from numpy import argmax

from benchmark_tasks.custom_task import MERATask
from lm_eval.api.instance import Instance
from lm_eval.api.metrics import mean


class LCS(MERATask):
    VERSION = 0
    DATASET_NAME = "lcs"

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
                self._training_docs = list(self.dataset["public_test"])
            return self._training_docs
        return []

    def doc_to_text(self, doc):
        prompt = doc["instruction"].format(inputs=doc["inputs"]).strip()
        return prompt

    def doc_to_text_without_instruction(self, doc):
        inputs = doc["inputs"]
        return inputs.strip()

    def doc_to_target(self, doc):
        return " " + doc["outputs"]

    def construct_requests(self, doc, ctx, **kwargs):
        ll_0 = Instance(
            request_type=self.OUTPUT_TYPE,
            doc=doc,
            arguments=(ctx, " 0"),
            idx=0,
            **kwargs,
        )
        ll_1 = Instance(
            request_type=self.OUTPUT_TYPE,
            doc=doc,
            arguments=(ctx, " 1"),
            idx=1,
            **kwargs,
        )
        ll_2 = Instance(
            request_type=self.OUTPUT_TYPE,
            doc=doc,
            arguments=(ctx, " 2"),
            idx=2,
            **kwargs,
        )
        ll_3 = Instance(
            request_type=self.OUTPUT_TYPE,
            doc=doc,
            arguments=(ctx, " 3"),
            idx=3,
            **kwargs,
        )
        ll_4 = Instance(
            request_type=self.OUTPUT_TYPE,
            doc=doc,
            arguments=(ctx, " 4"),
            idx=4,
            **kwargs,
        )
        ll_5 = Instance(
            request_type=self.OUTPUT_TYPE,
            doc=doc,
            arguments=(ctx, " 5"),
            idx=5,
            **kwargs,
        )
        ll_6 = Instance(
            request_type=self.OUTPUT_TYPE,
            doc=doc,
            arguments=(ctx, " 6"),
            idx=6,
            **kwargs,
        )
        ll_7 = Instance(
            request_type=self.OUTPUT_TYPE,
            doc=doc,
            arguments=(ctx, " 7"),
            idx=7,
            **kwargs,
        )
        ll_8 = Instance(
            request_type=self.OUTPUT_TYPE,
            doc=doc,
            arguments=(ctx, " 8"),
            idx=8,
            **kwargs,
        )
        ll_9 = Instance(
            request_type=self.OUTPUT_TYPE,
            doc=doc,
            arguments=(ctx, " 9"),
            idx=9,
            **kwargs,
        )
        return [ll_0, ll_1, ll_2, ll_3, ll_4, ll_5, ll_6, ll_7, ll_8, ll_9]

    def process_results(self, doc, results):
        if len(doc["outputs"]) > 0:
            results = [
                res[0] for res in results
            ]  # only retain loglikelihoods, discard is_greedy
            ll_0, ll_1, ll_2, ll_3, ll_4, ll_5, ll_6, ll_7, ll_8, ll_9 = results
            gold = int(doc["outputs"])
            max_arg = argmax(
                [ll_0, ll_1, ll_2, ll_3, ll_4, ll_5, ll_6, ll_7, ll_8, ll_9]
            )
            acc = max_arg == gold
            return {"acc": acc}
        return {"acc": 0.0}  # if no label provided (test answers are secret)

    def aggregation(self):
        return {"acc": mean}

    def higher_is_better(self):
        return {"acc": True}
