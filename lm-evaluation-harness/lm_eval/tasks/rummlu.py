"""
The Russian MMLU (ruMMLU) dataset.

Russian analogue of the MMLU dataset based on the English version. The dataset consists
of tasks with four possible answers, only one of which is correct.

Homepage: https://mera.a-ai.ru/
"""

import numpy as np

from lm_eval.metrics import mean
from lm_eval.base import MultipleChoiceTask


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


class RuMMLU(MultipleChoiceTask):
    VERSION = 0
    DATASET_NAME = "rummlu"

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def test_docs(self):
        return map(self._process_doc, self.dataset["test"])

    def training_docs(self):
        if self._training_docs is None:
            self._training_docs = list(map(self._process_doc, self.dataset["train"]))
        return self._training_docs

    def doc_to_target(self, doc):
        if isinstance(doc["gold"], int):
            gold = doc["choices"][doc["gold"]]
        else:
            gold = ""
        return " " + gold

    def _process_doc(self, doc):
        choices = ["A", "B", "C", "D"]
        if doc["outputs"]:
            gold = choices.index(doc["outputs"])
        else:
            gold = ""

        prompt = (
            doc["instruction"]
            .format(
                text=doc["inputs"]["text"],
                option_a=doc["inputs"]["option_a"],
                option_b=doc["inputs"]["option_b"],
                option_c=doc["inputs"]["option_c"],
                option_d=doc["inputs"]["option_d"],
                subject=doc["inputs"]["subject"],
            )
            .strip()
        )

        doc["query"] = prompt
        doc["choices"] = choices
        doc["gold"] = gold
        return doc

    def doc_to_text(self, doc):
        return doc["query"]

    def doc_to_text_without_instruction(self, doc):
        prompt = "{text}\nA) {option_a}\nB) {option_b}\nC) {option_c}\nD) {option_d}\nОтвет:".format(
            **doc["inputs"]
        ).strip()
        return prompt

    def should_decontaminate(self):
        return True

    def doc_to_decontamination_query(self, doc):
        return doc["query"]

    def fewshot_context(self, doc, num_fewshot, provide_description=None, rnd=None, description=None):
        assert rnd is not None, "A `random.Random` generator argument must be provided to `rnd`"
        assert not provide_description, (
            "The `provide_description` arg will be removed in future versions. To prepend "
            "a custom description to the context, supply the corresponding string via the "
            "`description` arg."
        )
        if provide_description is not None:
            # nudge people to not specify it at all
            print(
                "WARNING: provide_description is deprecated and will be removed in a future version in favor of description_dict"
            )

        description = description + "\n\n" if description else ""

        if num_fewshot == 0:
            labeled_examples = ""
            example = self.doc_to_text(doc)
        else:
            assert self.has_training_docs(), "The task must have a training set for few-shot learning."

            domain = doc["meta"]["domain"]
            fewshotex = self.fewshot_examples(domain=domain, k=num_fewshot, rnd=rnd)

            labeled_examples = ""
            for idx_shot, shot_doc in enumerate(fewshotex):
                if idx_shot == 0:
                    shot = self.doc_to_text(shot_doc) + self.doc_to_target(shot_doc)
                else:
                    shot = self.doc_to_text_without_instruction(shot_doc) + self.doc_to_target(shot_doc)

                labeled_examples += shot
                labeled_examples += "\n\n"

            example = self.doc_to_text_without_instruction(doc)

        return description + labeled_examples + example

    def fewshot_examples(self, domain, k, rnd):
        if self._training_docs is None:
            self._training_docs = list(self.training_docs())

        training_docs = [
            training_doc for training_doc in self._training_docs if training_doc["meta"]["domain"] == domain
        ]

        return rnd.sample(training_docs, k)

    def process_results(self, doc, results):
        gold = doc["gold"]
        acc = 1.0 if np.argmax(results) == gold else 0.0
        sub = doc["meta"]["domain"]
        return {
            "acc": acc,
            f"acc_{sub}": acc,
        }

    def higher_is_better(self):
        metrics = {f"acc_{sub}": True for sub in SUBJECTS}
        metrics["acc"] = True
        return metrics

    def aggregation(self):
        metrics = {f"acc_{sub}": mean for sub in SUBJECTS}
        metrics["acc"] = mean
        return metrics
