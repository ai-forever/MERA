"""
The Russian Multistep Arithmetic (ruMultiAr) dataset.

Russian Multistep Arithmetic is a mathematical task from Bigbench. This task tests
a model's ability to solve multistep arithmetic operations composed of addition,
subtraction, multiplication, and division. So we can measure the capability of models
to think sequentially.

Homepage: https://mera.a-ai.ru/
"""
from benchmark_tasks.custom_task import MERATask
from lm_eval.api.instance import Instance
from lm_eval.api.metrics import mean


class RuMultiAr(MERATask):
    VERSION = 0
    DATASET_NAME = "rumultiar"

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

    def doc_to_text_without_instruction(self, doc):
        inputs = doc["inputs"]
        return inputs.strip()

    def doc_to_text(self, doc):
        prompt = doc["instruction"].format(inputs=doc["inputs"])
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
        completion = results[0]
        completion1 = str(completion).strip()
        if len(doc["outputs"]) > 0:
            out = str(doc["outputs"])
            acc = int(completion1 == out)
            return {"acc": acc}
        return {"acc": 0}  # if no label provided (test answers are secret)

    def aggregation(self):
        return {"acc": mean}

    def higher_is_better(self):
        return {"acc": True}
