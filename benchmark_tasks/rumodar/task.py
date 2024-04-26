"""
The Russian Modified Arithmetic (ruModAr) dataset.

Russian Modified Arithmetic is a mathematical task from Bigbench.
Each question in each subtask begins with a prompt and five examples of arithmetic
expressions with results. The sixth example is incomplete, the model's task is to
finish it correctly.

Homepage: https://mera.a-ai.ru/
"""
from benchmark_tasks.custom_task import MERATask
from lm_eval.api.metrics import mean


class ruModAr(MERATask):
    VERSION = 0
    DATASET_NAME = "rumodar"

    def __init__(
        self,
        data_dir=None,
        cache_dir=None,
        download_mode=None,
        config=None,
    ) -> None:
        super().__init__(data_dir, cache_dir, download_mode, config)
        self._config.generation_kwargs["until"] = ["\n"]
    
    def training_docs(self):
        # need to check for tasks that do not have train docs
        if self.has_training_docs():
            if self._training_docs is None:
                self._training_docs = list(self.dataset["public_test"])
            return self._training_docs
        return []

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
