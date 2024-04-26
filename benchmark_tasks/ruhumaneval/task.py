"""
ruHumanEval
"""
from benchmark_tasks.custom_task import MERATask
from benchmark_tasks.ruhumaneval.utils import (
    check_solution,
    compute_pass_k,
    ruHumanEvalScoring,
)
from lm_eval.api.metrics import mean
from lm_eval.filters import build_filter_ensemble


class ruHumanEval(MERATask):
    VERSION = 0
    DATASET_NAME = "ruhumaneval"

    def __init__(
        self,
        data_dir=None,
        cache_dir=None,
        download_mode=None,
        config=None,
    ) -> None:
        super().__init__(data_dir, cache_dir, download_mode, config)
        self._config.generation_kwargs["until"] = [
            "\nclass",
            "\ndef",
            "\n#",
            "\nif",
            "\nprint",
        ]
        self._config.repeats = 10  # call model 10 times for each request
        self._filters = [build_filter_ensemble("scoring", [[ruHumanEvalScoring, None]])]
    
    def training_docs(self):
        # need to check for tasks that do not have train docs
        if self.has_training_docs():
            if self._training_docs is None:
                self._training_docs = list(self.dataset["public_test"])
            return self._training_docs
        return []

    def doc_to_text(self, doc):
        prompt = doc["instruction"].format(function=doc["inputs"]["function"])
        return prompt.strip()

    def doc_to_text_without_instruction(self, doc):
        prompt = "Функция:\n{function}".format(function=doc["inputs"]["function"])
        return prompt.strip()

    def doc_to_target(self, doc):
        target = doc["meta"]["canonical_solution"]
        return target

    def process_results(self, doc, results):
        if len(doc["outputs"]) > 0:
            output_results = []
            code_outputs = results[0]
            for generation in code_outputs:
                score = check_solution(doc["outputs"], generation)
                if score:
                    output_results.extend([1])
                else:
                    output_results.extend([0])

            total, correct = len(output_results), sum(output_results)

            pass_1 = compute_pass_k(total, correct, 1)
            pass_5 = compute_pass_k(total, correct, 5)
            pass_10 = compute_pass_k(total, correct, 10)
            return {"pass@1": pass_1, "pass@5": pass_5, "pass@10": pass_10}
        return {
            "pass@1": 0.0,
            "pass@5": 0.0,
            "pass@10": 0.0,
        }  # if no label provided (test answers are secret)

    def aggregation(self):
        return {"pass@1": mean, "pass@5": mean, "pass@10": mean}

    def higher_is_better(self):
        return {"pass@1": True, "pass@5": True, "pass@10": True}
