"""
The MathLogicQA dataset.

MathLogicQA is a QA dataset with multiple-choice math questions consisting systems
of equations, proportional relationships, and comparisons.

Homepage: https://mera.a-ai.ru/
"""

from numpy import argmax

from benchmark_tasks.custom_task import MultipleChoiceMERATask
from lm_eval.api.metrics import mean


class MathLogicQA(MultipleChoiceMERATask):
    VERSION = 0
    DATASET_NAME = "mathlogicqa"

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def training_docs(self):
        if self._training_docs is None:
            self._training_docs = list(map(self._process_doc, self.dataset["train"]))
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

    # def doc_to_target(self, doc):
    #     target = doc["outputs"]
    #     return " " + target
    # def construct_requests(self, doc, ctx):
    #     return rf.greedy_until(ctx, {"until": ["\n"]})

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
        return {"acc": acc}

    # def process_results(self, doc, results):
    #     answer = results[0].strip()
    #     if answer == doc["outputs"]:
    #         res = 1
    #     else:
    #         res = 0
    #     return {"acc": res}

    def aggregation(self):
        return {"acc": mean}

    def higher_is_better(self):
        return {"acc": True}
