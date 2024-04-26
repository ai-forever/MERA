"""
The Largest Common Subsequence (LCS) dataset.

The longest common subsequence (LCS) is an algorithmic task from Bigbench. This problem
consists of pairs of strings as input, and language models are expected to correctly
predict the length of the longest common subsequence between the strings.

Homepage: https://mera.a-ai.ru/
"""


from benchmark_tasks.custom_task import MultipleChoiceMERATask


class LCS(MultipleChoiceMERATask):
    VERSION = 0
    DATASET_NAME = "lcs"

    CHOICES = list(map(str, range(10)))

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
    
    def training_docs(self):
        if self.has_training_docs():
            if self._training_docs is None:
                self._training_docs = list(map(self.process_doc, self.dataset["public_test"]))
            return self._training_docs
        return []

    def doc_to_text_without_instruction(self, doc):
        inputs = "Строки: {inputs}\nОтвет:".format(inputs=doc["inputs"])
        return inputs.strip()
