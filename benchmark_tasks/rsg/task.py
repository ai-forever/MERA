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
from benchmark_tasks.custom_task import MultipleChoiceMERATask
from lm_eval.api.metrics import mean


class RSGTask(MultipleChoiceMERATask):
    def doc_to_text(self, doc):
        prompt = doc["instruction"].format(**doc["inputs"]) + "\nОтвет:"
        return prompt


class RCB(RSGTask):
    VERSION = 0
    DATASET_NAME = "rcb"

    CHOICES = ["1", "2", "3"]

    def doc_to_text_without_instruction(self, doc):
        prompt = "Ситуация: {premise}\nГипотеза: {hypothesis}\nОтвет:".format(
            **doc["inputs"]
        )
        return prompt

    def process_results(self, doc, results):
        if len(doc["outputs"]) > 0:
            results = [
                res[0] for res in results
            ]  # only retain loglikelihoods, discard is_greedy
            gold = doc["gold"]
            pred = argmax(results)
            return {"acc": float(pred == gold), "f1_macro": (gold, pred)}
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

    CHOICES = ["1", "2"]

    def doc_to_text_without_instruction(self, doc):
        prompt = "Ситуация: {premise}\nФрагмент 1: {choice1}\nФрагмент 2: {choice2}\nОтвет:".format(
            **doc["inputs"]
        )
        return prompt


class RWSD(RSGTask):
    VERSION = 0
    DATASET_NAME = "rwsd"

    CHOICES = ["Да", "Нет"]

    def doc_to_text_without_instruction(self, doc):
        prompt = "Текст: {text}\nФрагмент 1: {span1_text}\nФрагмент 2: {span2_text}\nОтвет:".format(
            **doc["inputs"]
        )
        return prompt
