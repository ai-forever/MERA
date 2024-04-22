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

from numpy import argmax

from benchmark_tasks.custom_metrics import f1_score_multiclass_macro
from benchmark_tasks.custom_task import MERATask
from lm_eval.api.instance import Instance
from lm_eval.api.metrics import mean


class RSGTask(MERATask):
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
        return []

    def validation_docs(self):
        if self.has_validation_docs():
            return self.dataset["validation"]
        return []

    def doc_to_text(self, doc):
        instruction = doc["instruction"]
        inputs = doc["inputs"]
        return instruction.format(**inputs) + "\nОтвет:"

    def doc_to_target(self, doc):
        target = doc["outputs"]
        return " " + target

    def aggregation(self):
        return {"acc": mean}

    def higher_is_better(self):
        return {"acc": True}


class RCB(RSGTask):
    VERSION = 0
    DATASET_NAME = "rcb"

    OUTPUT_TYPE = "loglikelihood"

    def construct_requests(self, doc, ctx, **kwargs):
        ll_entailment = Instance(
            request_type=self.OUTPUT_TYPE,
            doc=doc,
            arguments=(ctx, " 1"),
            idx=0,
            **kwargs,
        )
        ll_contradiction = Instance(
            request_type=self.OUTPUT_TYPE,
            doc=doc,
            arguments=(ctx, " 2"),
            idx=1,
            **kwargs,
        )
        ll_neutral = Instance(
            request_type=self.OUTPUT_TYPE,
            doc=doc,
            arguments=(ctx, " 3"),
            idx=2,
            **kwargs,
        )
        return [ll_entailment, ll_contradiction, ll_neutral]

    def process_results(self, doc, results):
        if len(doc["outputs"]) > 0:
            results = [
                res[0] for res in results
            ]  # only retain loglikelihoods, discard is_greedy
            gold = {"1": 0, "2": 1, "3": 2}[doc["outputs"]]
            pred = argmax(results)
            return {"acc": pred == gold, "f1_macro": (gold, pred)}
        return {
            "acc": 0.0,
            "f1_macro": (1, 0),
        }  # if no label provided (test answers are secret)

    def aggregation(self):
        return {"acc": mean, "f1_macro": f1_score_multiclass_macro}

    def higher_is_better(self):
        return {"acc": True, "f1_macro": True}


class PARus(RSGTask):
    VERSION = 0
    DATASET_NAME = "parus"

    OUTPUT_TYPE = "loglikelihood"

    def construct_requests(self, doc, ctx, **kwargs):
        ll_choice_1 = Instance(
            request_type=self.OUTPUT_TYPE,
            doc=doc,
            arguments=(ctx, " 1"),
            idx=0,
            **kwargs,
        )
        ll_choice_2 = Instance(
            request_type=self.OUTPUT_TYPE,
            doc=doc,
            arguments=(ctx, " 2"),
            idx=1,
            **kwargs,
        )
        return [ll_choice_1, ll_choice_2]

    def process_results(self, doc, results):
        if len(doc["outputs"]) > 0:
            results = [
                res[0] for res in results
            ]  # only retain loglikelihoods, discard is_greedy
            gold = {"1": 0, "2": 1}[doc["outputs"]]
            pred = argmax(results)
            return {"acc": pred == gold}
        return {"acc": 0.0}  # if no label provided (test answers are secret)


class RWSD(RSGTask):
    VERSION = 0
    DATASET_NAME = "rwsd"

    OUTPUT_TYPE = "loglikelihood"

    def construct_requests(self, doc, ctx, **kwargs):
        ll_yes = Instance(
            request_type=self.OUTPUT_TYPE,
            doc=doc,
            arguments=(ctx, " Да"),
            idx=0,
            **kwargs,
        )
        ll_no = Instance(
            request_type=self.OUTPUT_TYPE,
            doc=doc,
            arguments=(ctx, " Нет"),
            idx=1,
            **kwargs,
        )
        return [ll_yes, ll_no]

    def process_results(self, doc, results):
        if len(doc["outputs"]) > 0:
            results = [
                res[0] for res in results
            ]  # only retain loglikelihoods, discard is_greedy
            pred = argmax(results)
            output = {"Да": 0, "Нет": 1}[doc["outputs"]]
            return {"acc": pred == output}
        return {"acc": 0.0}  # if no label provided (test answers are secret)
