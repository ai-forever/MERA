"""
The Russian HateSpeech (ruHateSpeech) dataset.

The Russian HateSpeech dataset. The goal of the task is to predict whether the answer
to the toxic comment, which offends a particular group of people, is also toxic towards
this target group of people. This dataset is diagnostic and is not used in the overall
model evaluation. It is intended to identify the model's ethical bias and analyze
the possibility of its safe application. Any statements that appeared in the dataset
are used as negative examples of phenomena from which users should be protected.
These statements are recorded in the dataset only for the purpose of analyzing
the models' ability to avoid such speech. They are not intended to offend anyone
in any possible way.

Homepage: https://mera.a-ai.ru/
"""

import inspect

from lm_eval.metrics import mean
from lm_eval.base import Task, rf


class RuHateSpeech(Task):
    VERSION = 0
    DATASET_NAME = "ruhatespeech"

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def training_docs(self):
        raise NotImplementedError("This dataset has no training docs")

    def validation_docs(self):
        raise NotImplementedError("This dataset has no validation docs")

    def test_docs(self):
        if self.has_test_docs():
            return list(map(self._process_doc, self.dataset["test"]))

    def _process_doc(self, doc):
        return {
            "meta": {"id": doc["meta"]["id"]},
            "query": doc["instruction"].format(**doc["inputs"]),
            "gold": doc["outputs"],
        }

    def doc_to_text(self, doc):
        return doc["query"]

    def doc_to_target(self, doc):
        return " " + doc["gold"]

    def construct_requests(self, doc, ctx):
        ll_first, _ = rf.loglikelihood(ctx, " 1")
        ll_second, _ = rf.loglikelihood(ctx, " 2")
        return ll_first, ll_second

    def process_results(self, doc, results):
        ll_1, ll_2 = results
        pred = "1" if ll_1 > ll_2 else "2"
        return {"acc": float(pred == doc["gold"])}

    def aggregation(self):
        return {"acc": mean}

    def higher_is_better(self):
        return {"acc": True}
