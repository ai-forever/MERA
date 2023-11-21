"""
RussianSuperGLUE: A Russian Language Understanding Evaluation Benchmark
https://arxiv.org/abs/2010.15925

Modern universal language models and transformers such as BERT, ELMo, XLNet, RoBERTa
and others need to be properly compared and evaluated.
In the last year, new models and methods for pretraining and transfer learning have
driven striking performance improvements across a range of language understanding tasks.

We offer testing methodology based on tasks, typically proposed for “strong AI” — logic,
commonsense, reasoning. Adhering to the GLUE and SuperGLUE methodology,
we present a set of test tasks for general language understanding and leaderboard models.

For the first time a complete test for Russian language was developed,
which is similar to its English analog. Many datasets were composed for the first time,
and a leaderboard of models for the Russian language with comparable results is also presented.

Homepage: https://russiansuperglue.com
"""

import inspect
import numpy as np

from lm_eval.base import rf, Task
from lm_eval.metrics import mean, f1_score_multiclass


class RCB(Task):
    VERSION = 0
    DATASET_NAME = "rcb"

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def training_docs(self):
        if self.has_training_docs():
            if self._training_docs is None:
                self._training_docs = list(self.dataset["train"])
            return self._training_docs

    def validation_docs(self):
        if self.has_validation_docs():
            return self.dataset["validation"]

    def test_docs(self):
        if self.has_test_docs():
            return self.dataset["test"]

    def doc_to_text(self, doc):
        instruction = doc["instruction"]
        inputs = doc["inputs"]
        return instruction.format(**inputs) + "\n" + "Ответ:"

    def doc_to_target(self, doc):
        target = doc["outputs"]
        return " " + target

    def construct_requests(self, doc, ctx):
        ll_entailment, _ = rf.loglikelihood(ctx, " 1")
        ll_contradiction, _ = rf.loglikelihood(ctx, " 2")
        ll_neutral, _ = rf.loglikelihood(ctx, " 3")
        return ll_entailment, ll_contradiction, ll_neutral

    def process_results(self, doc, results):
        gold = {"1": 0, "2": 1, "3": 2}[doc["outputs"]]
        pred = np.argmax(results)
        return {"acc": pred == gold, "f1": (gold, pred)}

    def aggregation(self):
        return {"acc": mean, "f1": f1_score_multiclass}

    def higher_is_better(self):
        return {"acc": True, "f1": True}


class PARus(Task):
    VERSION = 0
    DATASET_NAME = "parus"

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def training_docs(self):
        if self.has_training_docs():
            if self._training_docs is None:
                self._training_docs = list(self.dataset["train"])
            return self._training_docs

    def validation_docs(self):
        if self.has_validation_docs():
            return self.dataset["validation"]

    def test_docs(self):
        if self.has_test_docs():
            return self.dataset["test"]

    def doc_to_text(self, doc):
        instruction = doc["instruction"]
        inputs = doc["inputs"]
        return instruction.format(**inputs) + "\nОтвет:"

    def doc_to_target(self, doc):
        target = doc["outputs"]
        return " " + target

    def construct_requests(self, doc, ctx):
        ll_choice_1, _ = rf.loglikelihood(ctx, " 1")
        ll_choice_2, _ = rf.loglikelihood(ctx, " 2")
        return ll_choice_1, ll_choice_2

    def process_results(self, doc, results):
        gold = {"1": 0, "2": 1}[doc["outputs"]]
        pred = np.argmax(results)
        return {"acc": pred == gold}

    def aggregation(self):
        return {"acc": mean}

    def higher_is_better(self):
        return {"acc": True}


class RWSD(Task):
    VERSION = 0
    DATASET_NAME = "rwsd"

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def training_docs(self):
        if self.has_training_docs():
            if self._training_docs is None:
                self._training_docs = list(self.dataset["train"])
            return self._training_docs

    def validation_docs(self):
        if self.has_validation_docs():
            return self.dataset["validation"]

    def test_docs(self):
        if self.has_test_docs():
            return self.dataset["test"]

    def doc_to_text(self, doc):
        instruction = doc["instruction"]
        inputs = doc["inputs"]
        return instruction.format(**inputs) + "\nОтвет:"

    def doc_to_target(self, doc):
        target = doc["outputs"]
        return " " + target

    def construct_requests(self, doc, ctx):
        ll_choice_yes, _ = rf.loglikelihood(ctx, " Да")
        ll_choice_no, _ = rf.loglikelihood(ctx, " Нет")
        return ll_choice_yes, ll_choice_no

    def process_results(self, doc, results):
        pred = np.argmax(results)
        output = {"Да": 0, "Нет": 1}[doc["outputs"]]
        return {"acc": pred == output}

    def aggregation(self):
        return {"acc": mean}

    def higher_is_better(self):
        return {"acc": True}
