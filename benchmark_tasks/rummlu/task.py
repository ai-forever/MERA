"""
The Russian MMLU (ruMMLU) dataset.

Russian analogue of the MMLU dataset based on the English version. The dataset consists
of tasks with four possible answers, only one of which is correct.

Homepage: https://mera.a-ai.ru/
"""

from numpy import argmax

from benchmark_tasks.custom_task import MultipleChoiceMERATask
from lm_eval.api.metrics import mean


SUBJECTS = [
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_medicine",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "econometrics",
    "electrical_engineering",
    "elementary_mathematics",
    "formal_logic",
    "global_facts",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_european_history",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_mathematics",
    "high_school_microeconomics",
    "high_school_physics",
    "high_school_psychology",
    "high_school_statistics",
    "high_school_us_history",
    "high_school_world_history",
    "human_aging",
    "human_sexuality",
    "international_law",
    "jurisprudence",
    "logical_fallacies",
    "machine_learning",
    "management",
    "marketing",
    "medical_genetics",
    "miscellaneous",
    "moral_disputes",
    "moral_scenarios",
    "nutrition",
    "philosophy",
    "prehistory",
    "professional_accounting",
    "professional_law",
    "professional_medicine",
    "professional_psychology",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy",
    "virology",
    "world_religions",
]


class RuMMLU(MultipleChoiceMERATask):
    VERSION = 0
    DATASET_NAME = "rummlu"

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
    
    def training_docs(self):
        if self.has_training_docs():
            if self._training_docs is None:
                self._training_docs = list(map(self.process_doc, self.dataset["public_test"]))
            return self._training_docs
        return []

    def doc_to_text_without_instruction(self, doc):
        prompt = "{text}\nA) {option_a}\nB) {option_b}\nC) {option_c}\nD) {option_d}\nОтвет:".format(
            **doc["inputs"]
        )
        return prompt.strip()

    def fewshot_examples(self, doc, k, rnd):
        docs = list(self.fewshot_docs())
        # need only subset of docs with specific domain
        docs = [
            one_doc
            for one_doc in docs
            if one_doc["meta"]["domain"] == doc["meta"]["domain"]
        ]
        return rnd.sample(docs, k + 1)

    def process_results(self, doc, results):
        domain = doc["meta"]["domain"]
        if len(doc["outputs"]) > 0:
            results = [
                res[0] for res in results
            ]  # only retain loglikelihoods, discard is_greedy
            gold = doc["gold"]  # index of target in doc["choices"] array
            pred = argmax(results)  # index of prediction in doc["choices"] array
            acc = float(pred == gold)
            return {"acc": acc, f"acc_{domain}": acc}
        else:
            return {"acc": 0.0, f"acc_{domain}": 0.0}

    def aggregation(self):
        metrics = {f"acc_{sub}": mean for sub in SUBJECTS}
        metrics["acc"] = mean
        return metrics

    def higher_is_better(self):
        metrics = {f"acc_{sub}": True for sub in SUBJECTS}
        metrics["acc"] = True
        return metrics
