"""
The Russian MMLU (ruMMLU) dataset.

Russian analogue of the MMLU dataset based on the English version. The dataset consists
of tasks with four possible answers, only one of which is correct.

Homepage: https://mera.a-ai.ru/
"""
from numpy import argmax

from benchmark_tasks.custom_task import MultipleChoiceMERATask
from lm_eval.api.metrics import mean
from lm_eval.utils import positional_deprecated


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

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def training_docs(self):
        if self._training_docs is None:
            self._training_docs = list(map(self._process_doc, self.dataset["public_test"]))
        return self._training_docs

    def test_docs(self):
        # random shuffling with fixed seed, same as MERA 1.1.0
        self.rnd.seed(42)
        docs = list(map(self._process_doc, self.dataset["test"]))
        self.rnd.shuffle(docs)
        return docs

    def doc_to_text(self, doc):
        return doc["query"]

    def doc_to_text_without_instruction(self, doc):
        # no strip for greedy
        prompt = "{text}\nA) {option_a}\nB) {option_b}\nC) {option_c}\nD) {option_d}\nОтвет:".format(
            **doc["inputs"]
        )
        return prompt.strip()

    def doc_to_target(self, doc: dict) -> str:
        if isinstance(doc["gold"], int):
            target = doc["choices"][doc["gold"]]
        else:
            target = ""
        return " " + target

    def _process_doc(self, doc):
        choices = ["A", "B", "C", "D"]
        if doc["outputs"]:
            gold = choices.index(doc["outputs"])
        else:
            gold = ""

        prompt = (  # TODO: maybe convert to .format(**doc["inputs"])
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

    def process_results(self, doc, results):
        gold = doc["gold"]
        if isinstance(gold, int):
            results = [
                res[0] for res in results
            ]  # only retain loglikelihoods, discard is_greedy
            acc = 1.0 if argmax(results) == gold else 0.0
        else:  # if no label provided (test answers are secret)
            acc = 0.0
        sub = doc["meta"]["domain"]
        return {"acc": acc, f"acc_{sub}": acc}

    def aggregation(self):
        metrics = {f"acc_{sub}": mean for sub in SUBJECTS}
        metrics["acc"] = mean
        return metrics

    def higher_is_better(self):
        metrics = {f"acc_{sub}": True for sub in SUBJECTS}
        metrics["acc"] = True
        return metrics

    def fewshot_examples(self, domain, k: int, rnd):
        if self._training_docs is None:
            self._training_docs = list(self.training_docs())

        training_docs = [
            training_doc
            for training_doc in self._training_docs
            if training_doc["meta"]["domain"] == domain
        ]
        return rnd.sample(training_docs, k)

    @positional_deprecated
    def fewshot_context(
        self,
        doc,
        num_fewshot,
        rnd=None,
        description=None,
    ):
        if rnd is None:
            rnd = self.rnd
        assert (
            rnd is not None
        ), "A `random.Random` generator argument must be provided to `rnd`"

        description = description + self.config.fewshot_delimiter if description else ""

        if num_fewshot == 0:
            labeled_examples = ""
            example = self.doc_to_text(doc)  # MERA 1.1.0 fewshot ideas
        else:
            assert (
                self.has_training_docs()
            ), "The task must have a training set for few-shot learning."

            domain = doc["meta"]["domain"]
            fewshotex = self.fewshot_examples(domain=domain, k=num_fewshot, rnd=rnd)

            ### MERA 1.1.0 fewshot idea
            labeled_examples = ""
            for idx_shot, shot_doc in enumerate(fewshotex):
                if idx_shot == 0:
                    shot = self.doc_to_text(shot_doc) + self.doc_to_target(shot_doc)
                else:
                    shot = self.doc_to_text_without_instruction(
                        shot_doc
                    ) + self.doc_to_target(shot_doc)
                labeled_examples += shot
                labeled_examples += self.config.fewshot_delimiter

            example = self.doc_to_text_without_instruction(doc)
            ### End of MERA 1.1.0 fewshot idea

        return description + labeled_examples + example
