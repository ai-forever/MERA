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
from benchmark_tasks.custom_task import MERATask
from lm_eval.api.instance import Instance
from lm_eval.api.metrics import mean, metric_max_over_ground_truths


class TapeTask(MERATask):
    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True


class CheGeKa(TapeTask):
    VERSION = 0
    DATASET_NAME = "chegeka"

    OUTPUT_TYPE = "generate_until"

    def __init__(
        self,
        data_dir=None,
        cache_dir=None,
        download_mode=None,
        config=None,
    ) -> None:
        super().__init__(data_dir, cache_dir, download_mode, config)
        self._config.generation_kwargs["until"] = ["."]

    def training_docs(self):
        if self.has_training_docs():
            if self._training_docs is None:
                self._training_docs = list(self.dataset["train"])
            return self._training_docs
        return []

    def doc_to_text(self, doc):
        prompt = doc["instruction"].format(**doc["inputs"])
        return prompt.strip()

    def doc_to_text_without_instruction(self, doc):
        prompt = 'Категория "{topic}"\nВопрос: {text}\nОтвет:'.format(**doc["inputs"])
        return prompt.strip()

    def doc_to_target(self, doc):
        return " " + doc["outputs"]

    def construct_requests(self, doc, ctx, **kwargs):
        return Instance(
            request_type=self.OUTPUT_TYPE,
            doc=doc,
            arguments=(ctx, self.config["generation_kwargs"]),
            idx=0,
            **kwargs,
        )

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

    def aggregation(self):
        return {"f1": mean, "em": mean}

    def higher_is_better(self):
        return {"f1": True, "em": True}


class MultiQ(TapeTask):
    VERSION = 0
    DATASET_NAME = "multiq"

    OUTPUT_TYPE = "generate_until"

    def __init__(
        self,
        data_dir=None,
        cache_dir=None,
        download_mode=None,
        config=None,
    ) -> None:
        super().__init__(data_dir, cache_dir, download_mode, config)
        self._config.generation_kwargs["until"] = ["."]

    def training_docs(self):
        if self.has_training_docs():
            if self._training_docs is None:
                self._training_docs = list(self.dataset["train"])
            return self._training_docs
        return []

    def doc_to_text(self, doc):
        prompt = doc["instruction"].format(**doc["inputs"])
        return prompt.strip()

    def doc_to_target(self, doc):
        return " " + doc["outputs"][0]["segment"]

    def construct_requests(self, doc, ctx, **kwargs):
        return Instance(
            request_type="generate_until",
            doc=doc,
            arguments=(ctx, self.config["generation_kwargs"]),
            idx=0,
            **kwargs,
        )

    def process_results(self, doc, results):
        # - Pick the maximum likelihood prediction entity
        # - Evaluate the accuracy and token F1 PER EXAMPLE
        # - Average over all examples
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

    def higher_is_better(self):
        return {"f1": True, "em": True}

    def aggregation(self):
        return {"f1": mean, "em": mean}


class RuWorldTree(TapeTask):
    VERSION = 0
    DATASET_NAME = "ruworldtree"

    OUTPUT_TYPE = "loglikelihood"

    def training_docs(self):
        if self.has_training_docs():
            if self._training_docs is None:
                self._training_docs = list(
                    map(self._process_doc, self.dataset["train"])
                )
            return self._training_docs
        return []

    def test_docs(self):
        if self.has_test_docs():
            # random shuffling with fixed seed, same as MERA 1.1.0
            self.rnd.seed(42)
            docs = list(map(self._process_doc, self.dataset["test"]))
            self.rnd.shuffle(docs)
            return docs
        return []

    def _process_doc(self, doc):
        query = doc["instruction"].format(**doc["inputs"]).strip()
        choices = list("ABCD")
        if doc["outputs"]:
            gold = choices.index(doc["outputs"])
        else:
            gold = ""

        doc["query"] = query
        doc["choices"] = choices
        doc["gold"] = gold
        return doc

    def doc_to_text(self, doc):
        return doc["query"]

    def doc_to_text_without_instruction(self, doc):
        prompt = "{question}\nA) {option_a}\nB) {option_b}\nC) {option_c}\nD) {option_d}\nОтвет:".format(
            **doc["inputs"]
        )
        return prompt.strip()

    def doc_to_target(self, doc):
        if isinstance(doc["gold"], int):
            gold = doc["choices"][doc["gold"]]
        else:
            gold = ""
        return " " + gold

    def construct_requests(self, doc, ctx, **kwargs):
        return [
            Instance(
                request_type=self.OUTPUT_TYPE,
                doc=doc,
                arguments=(ctx, " {}".format(choice)),
                idx=i,
                **kwargs,
            )
            for i, choice in enumerate(doc["choices"])
        ]

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

    def higher_is_better(self):
        return {"acc": True, "f1_macro": True}

    def aggregation(self):
        return {"acc": mean, "f1_macro": f1_score_multiclass_macro}


class RuOpenBookQA(TapeTask):
    VERSION = 0
    DATASET_NAME = "ruopenbookqa"

    OUTPUT_TYPE = "loglikelihood"

    def training_docs(self):
        if self.has_training_docs():
            if self._training_docs is None:
                self._training_docs = list(
                    map(self._process_doc, self.dataset["train"])
                )
            return self._training_docs
        return []

    def test_docs(self):
        if self.has_test_docs():
            # random shuffling with fixed seed, same as MERA 1.1.0
            self.rnd.seed(42)
            docs = list(map(self._process_doc, self.dataset["test"]))
            self.rnd.shuffle(docs)
            return docs
        return []

    def _process_doc(self, doc):
        query = doc["instruction"].format(**doc["inputs"]).strip()
        choices = list("ABCD")
        if doc["outputs"]:
            gold = choices.index(doc["outputs"])
        else:
            gold = ""

        doc["query"] = query
        doc["choices"] = choices
        doc["gold"] = gold
        return doc

    def doc_to_text(self, doc):
        return doc["query"]

    def doc_to_text_without_instruction(self, doc):
        prompt = "{question}\nA) {option_a}\nB) {option_b}\nC) {option_c}\nD) {option_d}\nОтвет:".format(
            **doc["inputs"]
        )
        return prompt.strip()

    def doc_to_target(self, doc):
        if isinstance(doc["gold"], int):
            gold = doc["choices"][doc["gold"]]
        else:
            gold = ""
        return " " + gold

    def construct_requests(self, doc, ctx, **kwargs):
        return [
            Instance(
                request_type=self.OUTPUT_TYPE,
                doc=doc,
                arguments=(ctx, " {}".format(choice)),
                idx=i,
                **kwargs,
            )
            for i, choice in enumerate(doc["choices"])
        ]

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

    def higher_is_better(self):
        return {"acc": True, "f1_macro": True}

    def aggregation(self):
        return {"acc": mean, "f1_macro": f1_score_multiclass_macro}
