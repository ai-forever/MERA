"""
TAPE: Assessing Few-shot Russian Language Understanding
https://arxiv.org/pdf/2210.12813.pdf

TAPE (Text Attack and Perturbation Evaluation) is a novel benchmark for few-shot
Russian language understanding evaluation that includes six complex NLU tasks, covering
multi-hop reasoning, ethical concepts, logic and commonsense knowledge.

Homepage: https://tape-benchmark.com/
"""
import transformers.data.metrics.squad_metrics as squad_metrics
from numpy import argmax

from benchmark_tasks.custom_metrics import f1_score_multiclass_macro
from benchmark_tasks.custom_task import MERATask, MultipleChoiceMERATask
from lm_eval.api.metrics import mean, metric_max_over_ground_truths


class GenerativeTAPETask(MERATask):
    VERSION = 0

    def __init__(
        self,
        data_dir=None,
        cache_dir=None,
        download_mode=None,
        config=None,
    ) -> None:
        super().__init__(data_dir, cache_dir, download_mode, config)
        self._config.generation_kwargs["until"] = ["."]

    def aggregation(self):
        return {"f1": mean, "em": mean}

    def higher_is_better(self):
        return {"f1": True, "em": True}


class CheGeKa(GenerativeTAPETask):
    DATASET_NAME = "chegeka"

    def __init__(
        self,
        data_dir=None,
        cache_dir=None,
        download_mode=None,
        config=None,
    ) -> None:
        super().__init__(data_dir, cache_dir, download_mode, config)

        # fix default number of few-shots
        self._config.num_fewshot = 4

    def doc_to_text_without_instruction(self, doc):
        prompt = 'Категория "{topic}"\nВопрос: {text}\nОтвет:'.format(**doc["inputs"])
        return prompt.strip()

    def process_results(self, doc, results):
        if len(doc["outputs"]) > 0:
            gold_label_set = doc["outputs"].split(";")
            pred = results[0]

            f1 = metric_max_over_ground_truths(
                squad_metrics.compute_f1, pred, gold_label_set
            )
            em = metric_max_over_ground_truths(
                squad_metrics.compute_exact, pred, gold_label_set
            )

            return {"f1": f1, "em": em}
        return {"f1": 0.0, "em": 0.0}  # if no label provided (test answers are secret)


class MultiQ(GenerativeTAPETask):
    DATASET_NAME = "multiq"

    def doc_to_text_without_instruction(self, doc):
        prompt = "Вопрос: {question}\nТекст 1: {support_text}\nТекст 2: {text}\nОтвет:".format(
            **doc["inputs"]
        )
        return prompt.strip()

    def doc_to_target(self, doc):
        target = doc["outputs"][0]["segment"]
        return " " + target

    def process_results(self, doc, results):
        if len(doc["outputs"]) > 0:
            gold_label_set = [answer["segment"] for answer in doc["outputs"]]
            pred = results[0]

            f1 = metric_max_over_ground_truths(
                squad_metrics.compute_f1, pred, gold_label_set
            )
            em = metric_max_over_ground_truths(
                squad_metrics.compute_exact, pred, gold_label_set
            )

            return {"f1": f1, "em": em}
        return {"f1": 0, "em": 0}  # if no label provided (test answers are secret)


class MultipleChoiceTAPETask(MultipleChoiceMERATask):
    VERSION = 0

    CHOICES = ["A", "B", "C", "D"]

    def __init__(
        self,
        data_dir=None,
        cache_dir=None,
        download_mode=None,
        config=None,
    ) -> None:
        super().__init__(data_dir, cache_dir, download_mode, config)

        # fix default number of few-shots
        self._config.num_fewshot = 5

    def doc_to_text_without_instruction(self, doc):
        prompt = "{question}\nA) {option_a}\nB) {option_b}\nC) {option_c}\nD) {option_d}\nОтвет:".format(
            **doc["inputs"]
        )
        return prompt.strip()

    def process_results(self, doc, results):
        if len(doc["outputs"]) > 0:
            results = [
                res[0] for res in results
            ]  # only retain loglikelihoods, discard is_greedy
            gold = doc["gold"]
            pred = argmax(results)
            return {"acc": pred == gold, "f1_macro": (gold, pred)}
        return {
            "acc": 0,
            "f1_macro": (1, 0),
        }  # if no label provided (test answers are secret)

    def aggregation(self):
        return {"acc": mean, "f1_macro": f1_score_multiclass_macro}

    def higher_is_better(self):
        return {"acc": True, "f1_macro": True}


class RuWorldTree(MultipleChoiceTAPETask):
    DATASET_NAME = "ruworldtree"


class RuOpenBookQA(MultipleChoiceTAPETask):
    DATASET_NAME = "ruopenbookqa"
