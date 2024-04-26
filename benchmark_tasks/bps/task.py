"""
The Balanced Parentheses Sequence (BPS) dataset.

The Balanced Parentheses Sequence (BPS) is an algorithmic task from BIG-bench.
The primary purpose of this task is to measure language models' ability to learn CS
algorithmic concepts like stacks, recursion, or dynamic programming.

Homepage: https://mera.a-ai.ru/
"""

import re

from benchmark_tasks.custom_task import MultipleChoiceMERATask


class BPS(MultipleChoiceMERATask):
    VERSION = 0
    DATASET_NAME = "bps"
    CHOICES = ["0", "1"]
    PATTERN = re.compile(r"\{([^}]*)\}")

    def __init__(
        self,
        data_dir=None,
        cache_dir=None,
        download_mode=None,
        config=None,
    ) -> None:
        super().__init__(data_dir, cache_dir, download_mode, config)

        # fix default number of few-shots
        self._config.num_fewshot = 2

    def _custom_format(self, input_str, **kwargs):
        """
        Fills only fields mentioned in kwargs of instruction.
        Ignore any other rooms created by {} brackets that are
        part of some instructions and not supposed to be filled.
        """
        matches = self.PATTERN.finditer(input_str)
        for match in matches:
            placeholder = match.group(0)
            key = match.group(1)
            if key in kwargs and kwargs[key]:
                input_str = input_str.replace(placeholder, str(kwargs[key]))
        return input_str

    def doc_to_text(self, doc):
        prompt = self._custom_format(doc["instruction"], inputs=doc["inputs"])
        return prompt

    def doc_to_text_without_instruction(self, doc):
        prompt = "Последовательность: {inputs}\nОтвет:".format(inputs=doc["inputs"])
        return prompt
