"""
The Simple arithmetic (SimpleAr) dataset.

Simple arithmetic - a mathematical task based on the set from Bigbench. The task itself
tests language models' basic arithmetic capabilities by asking them to perform n-digit
addition for a range of n.

Homepage: https://mera.a-ai.ru/
"""
from benchmark_tasks.custom_task import MERATask
from lm_eval.api.metrics import mean


class SimpleAr(MERATask):
    VERSION = 0
    DATASET_NAME = "simplear"

    def __init__(
        self,
        data_dir=None,
        cache_dir=None,
        download_mode=None,
        config=None,
    ) -> None:
        super().__init__(data_dir, cache_dir, download_mode, config)
        self._config.generation_kwargs["until"] = ["\n"]

        # fix default number of few-shots
        self._config.num_fewshot = 5

    def doc_to_text_without_instruction(self, doc):
        prompt = doc["inputs"]
        return prompt

    def process_results(self, doc, results):
        if len(doc["outputs"]) > 0:
            completion = str(results[0])
            out = str(doc["outputs"])
            acc = float(completion == out)
            return {"acc": acc}
        return {"acc": 0}  # if no label provided (test answers are secret)

    def aggregation(self):
        return {"acc": mean}

    def higher_is_better(self):
        return {"acc": True}
