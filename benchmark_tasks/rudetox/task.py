"""
The RuDetox Diagnostic (ruDetox) dataset.

RuDetox Diagnostic is a part of RuDetox - a parallel corpus for text detoxification.
Given a sentence written in a toxic style, the model is asked to rewrite it in a polite
style preserving original meaning and fluency.

Homepage: https://mera.a-ai.ru/
"""

from benchmark_tasks.custom_task import MERATask
from benchmark_tasks.rudetox.utils import ruDetoxScoring
from lm_eval.api.instance import Instance
from lm_eval.api.metrics import mean
from lm_eval.filters import build_filter_ensemble


class ruDetox(MERATask):
    VERSION = 0
    DATASET_NAME = "rudetox"

    OUTPUT_TYPE = "generate_until"

    def __init__(
        self,
        data_dir=None,
        cache_dir=None,
        download_mode=None,
        config=None,
    ) -> None:
        super().__init__(data_dir, cache_dir, download_mode, config)
        self._config.generation_kwargs["until"] = ["\n"]
        self._filters = [build_filter_ensemble("scoring", [[ruDetoxScoring, None]])]

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

    def doc_to_text(self, doc):
        prompt = doc["instruction"].format(toxic_comment=doc["inputs"])
        return prompt.strip()

    def doc_to_text_without_instruction(self, doc):
        prompt = (
            "Токсичный вариант текста: {text}\nНетоксичный вариант этого текст:".format(
                text=doc["inputs"]
            )
        )
        return prompt.strip()

    def doc_to_target(self, doc):
        target = doc["outputs"]
        return " " + target

    def construct_requests(self, doc, ctx, **kwargs):
        ans = Instance(
            request_type=self.OUTPUT_TYPE,
            doc=doc,
            arguments=(ctx, self.config["generation_kwargs"]),
            idx=0,
            **kwargs,
        )
        return ans

    def process_results(self, doc, results):
        if len(doc["outputs"]) > 0:
            metrics = results[0]
            sta, sim, fl, j, _ = metrics
            return {"j": j, "sta": sta, "sim": sim, "fl": fl}
        return {
            "j": 0.0,
            "sta": 0.0,
            "sim": 0.0,
            "fl": 0.0,
        }  # if no label provided (test answers are secret)

    def aggregation(self):
        return {"j": mean, "sta": mean, "sim": mean, "fl": mean}

    def higher_is_better(self):
        return {"j": True, "sta": True, "sim": True, "fl": True}
