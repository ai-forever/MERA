import os
import random
from typing import Iterable, List, Optional, Tuple

import numpy as np

from lm_eval.api.instance import Instance
from lm_eval.api.metrics import mean
from lm_eval.api.task import Task
from lm_eval.utils import positional_deprecated


BENCHMARK_STORAGE: Optional[str] = "ai-forever/MERA"
DATASETS_DIR: str = "datasource"


class MERATask(Task):
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

    def download(
        self, data_dir=None, cache_dir=None, download_mode="force_redownload"
    ) -> None:
        super().download(data_dir, cache_dir, download_mode)

    def test_docs(self):
        if self.has_test_docs():
            # random shuffling with fixed seed, same as MERA 1.1.0
            self.rnd.seed(42)
            docs = list(self.dataset["test"])
            self.rnd.shuffle(docs)
            return docs
        return []

    def doc_to_text_without_instruction(self, doc):
        return self.doc_to_text(doc)

    @positional_deprecated
    def fewshot_context(
        self,
        doc,
        num_fewshot,
        rnd=None,
        description=None,
    ):
        """Returns a fewshot context string that is made up of a prepended description
        (if provided), the `num_fewshot` number of examples, and an appended prompt example.

        :param doc: str
            The document as returned from training_docs, validation_docs, or test_docs.
        :param num_fewshot: int
            The number of fewshot examples to provide in the returned context string.
        :param rnd: random.Random
            The pseudo-random number generator used to randomly sample examples.
            WARNING: This is currently a required arg although it's optionalized with a default `None`.
        :param description: str
            The task's description that will be prepended to the fewshot examples.
        :returns: str
            The fewshot context.
        """
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
            example = self.doc_to_text(doc)  # MERA 1.1.0 fewshot ideas
        else:
            # for sets with no training docs, draw from other set *but ensure no overlap with current doc*
            if self.has_training_docs():
                fewshotex = self.fewshot_examples(k=num_fewshot, rnd=rnd)
            else:
                if self._fewshot_docs is None:
                    self._fewshot_docs = list(
                        self.validation_docs()
                        if self.has_validation_docs()
                        else self.test_docs()
                    )

                fewshotex = rnd.sample(self._fewshot_docs, num_fewshot + 1)

                # get rid of the doc that's the one we're evaluating, if it's in the fewshot
                fewshotex = [x for x in fewshotex if x != doc][:num_fewshot]

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


class MultipleChoiceMERATask(MERATask):
    OUTPUT_TYPE = "loglikelihood"

    def doc_to_target(self, doc: dict) -> str:
        # TODO do we need if isinstance(doc["gold"], int)
        return " " + doc["choices"][doc["gold"]]

    def construct_requests(self, doc: dict, ctx: str, **kwargs) -> List[Instance]:
        # TODO: add mutual info here?
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

    def process_results(self, doc: dict, results: Iterable[Tuple[float, bool]]) -> dict:
        results = [
            res[0] for res in results
        ]  # only retain loglikelihoods, discard is_greedy TODO: do we need is_greedy anywhere?
        gold = doc["gold"]

        acc = 1.0 if np.argmax(results) == gold else 0.0
        completion_len = np.array([float(len(i)) for i in doc["choices"]])
        acc_norm = 1.0 if np.argmax(results / completion_len) == gold else 0.0

        return {
            "acc": acc,
            "acc_norm": acc_norm,
        }

    def higher_is_better(self) -> dict:
        return {
            "acc": True,
            "acc_norm": True,
        }

    def aggregation(self) -> dict:
        return {
            "acc": mean,
            "acc_norm": mean,
        }
