import os
import random
from typing import Iterable, List, Optional, Tuple

import numpy as np

from lm_eval.api.instance import Instance
from lm_eval.api.metrics import mean
from lm_eval.api.task import Task
from lm_eval.filters import build_filter_ensemble
from lm_eval.utils import positional_deprecated


BENCHMARK_STORAGE: Optional[str] = "ai-forever/MERA"
DATASETS_DIR: str = "datasource"


class MERATask(Task):
    OUTPUT_TYPE = "generate_until"
    DATASET_NAME = ""

    def __init__(
        self,
        data_dir=None,
        cache_dir=None,
        download_mode=None,
        config=None,
    ) -> None:
        self.DATASET_PATH = BENCHMARK_STORAGE or os.path.abspath(
            os.path.join(DATASETS_DIR, self.DATASET_NAME, f"{self.DATASET_NAME}.py")
        )
        super().__init__(data_dir, cache_dir, download_mode, config)
        self._config.output_type = self.OUTPUT_TYPE
        self.rnd = random.Random(42)
        self._filters = [
            build_filter_ensemble(
                "metrics", [["remove_whitespace", None], ["take_first", None]]
            )
        ]

        # fix default number of few-shots
        self._config.num_fewshot = 0

    # almost all tasks have train and test sets only
    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def download(
        self, data_dir=None, cache_dir=None, download_mode="force_redownload"
    ) -> None:
        super().download(data_dir, cache_dir, download_mode)

    def training_docs(self):
        # need to check for tasks that do not have train docs
        if self.has_training_docs():
            if self._training_docs is None:
                self._training_docs = list(self.dataset["train"])
            return self._training_docs
        return []

    def test_docs(self):
        if self.has_test_docs():
            # random shuffling with fixed seed, same as MERA 1.1.0
            self.rnd.seed(42)
            docs = list(self.dataset["test"])
            self.rnd.shuffle(docs)
            return docs
        return []

    def doc_to_text(self, doc):
        if isinstance(doc["inputs"], dict):
            prompt = doc["instruction"].format(**doc["inputs"]).strip()
        else:
            prompt = doc["instruction"].format(inputs=doc["inputs"]).strip()
        return prompt

    def doc_to_target(self, doc: dict) -> str:
        target = doc["outputs"]
        return " " + target

    # all tasks should override this func
    def doc_to_text_without_instruction(self, doc):
        raise NotImplementedError(
            "Task {task} should have `doc_to_text_without_instruction` method!".format(
                task=self.DATASET_NAME
            )
        )

    def construct_requests(self, doc, ctx, **kwargs):
        return Instance(
            request_type=self.OUTPUT_TYPE,
            doc=doc,
            arguments=(ctx, self.config["generation_kwargs"]),
            idx=0,
            **kwargs,
        )

    def fewshot_examples(self, doc, k, rnd):
        docs = list(self.fewshot_docs())
        return rnd.sample(docs, k + 1)

    # custom MERA fewshot construction strategy
    @positional_deprecated
    def fewshot_context(
        self,
        doc,
        num_fewshot,
        rnd=None,
        description=None,
    ):
        if rnd is None:
            if self.rnd is not None:
                rnd = self.rnd
            else:
                raise ValueError(
                    "A `random.Random` generator argument must be provided to `rnd`"
                )

        description = (
            description + self.config.fewshot_delimiter if description else ""
        )  # MERA 1.1.0 fewshot ideas

        if num_fewshot == 0:
            labeled_examples = ""
            # only the example with instruction
            example = self.doc_to_text(doc)  # MERA 1.1.0 fewshot ideas
        else:
            # select (num_fewshot+1) docs from fewshot_docs (train or val or test)
            fewshotex = self.fewshot_examples(doc=doc, k=num_fewshot, rnd=rnd)

            # get rid of the doc that's the one we're evaluating, if it's in the fewshot
            # when select from test for test samples, or for train when pick from train
            fewshotex = [x for x in fewshotex if x != doc][:num_fewshot]

            ### MERA 1.1.0 fewshot idea
            labeled_examples = ""
            for idx_shot, shot_doc in enumerate(fewshotex):
                if idx_shot == 0:
                    # first shot with instruction
                    shot = self.doc_to_text(shot_doc) + self.doc_to_target(shot_doc)
                else:
                    # all others with no instruction
                    shot = self.doc_to_text_without_instruction(
                        shot_doc
                    ) + self.doc_to_target(shot_doc)
                labeled_examples += shot
                labeled_examples += self.config.fewshot_delimiter

            example = self.doc_to_text_without_instruction(doc)
            ### End of MERA 1.1.0 fewshot idea

        return description + labeled_examples + example


class MultipleChoiceMERATask(MERATask):
    OUTPUT_TYPE = "loglikelihood"
    CHOICES = []

    def __init__(
        self,
        data_dir=None,
        cache_dir=None,
        download_mode=None,
        config=None,
    ) -> None:
        super().__init__(data_dir, cache_dir, download_mode, config)
        # only reduce dim of resps
        self._filters = [build_filter_ensemble("metrics", [["take_first", None]])]
        if not getattr(self, "CHOICES", False):
            raise AttributeError(
                "Provide `CHOICES` attribute for {name} task!".format(
                    name=self.DATASET_NAME
                )
            )

    # all multiple choice tasks should have process_doc func
    # at least it adds doc["choices"] and doc["gold"]: int
    def process_doc(self, doc: dict) -> dict:
        if doc["outputs"]:
            gold = self.CHOICES.index(doc["outputs"])
        else:
            gold = ""
        doc["choices"] = self.CHOICES
        doc["gold"] = gold
        return doc

    def training_docs(self):
        if self.has_training_docs():
            if self._training_docs is None:
                self._training_docs = list(map(self.process_doc, self.dataset["train"]))
            return self._training_docs
        return []

    def test_docs(self):
        if self.has_test_docs():
            # random shuffling with fixed seed, same as MERA 1.1.0
            self.rnd.seed(42)
            docs = list(map(self.process_doc, self.dataset["test"]))
            self.rnd.shuffle(docs)
            return docs

    def construct_requests(self, doc: dict, ctx: str, **kwargs) -> List[Instance]:
        return [
            Instance(
                request_type="loglikelihood",
                doc=doc,
                arguments=(ctx, " {}".format(choice)),
                idx=i,
                **kwargs,
            )
            for i, choice in enumerate(doc["choices"])
        ]

    # default task uses only accuracy_score
    def process_results(self, doc: dict, results: Iterable[Tuple[float, bool]]) -> dict:
        # for diagnostic tasks or scoring on train set
        if len(doc["outputs"]) > 0:
            results = [
                res[0] for res in results
            ]  # only retain loglikelihoods, discard is_greedy
            gold = doc["gold"]  # index of target in doc["choices"] array
            pred = np.argmax(results)  # index of prediction in doc["choices"] array
            acc = float(pred == gold)
            return {
                "acc": acc,
            }
        # otherwise, test targets are secret
        else:
            return {
                "acc": 0.0,
            }

    def higher_is_better(self) -> dict:
        return {
            "acc": True,
        }

    def aggregation(self) -> dict:
        return {
            "acc": mean,
        }
