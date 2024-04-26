"""
The MathLogicQA dataset.

MathLogicQA is a QA dataset with multiple-choice math questions consisting systems
of equations, proportional relationships, and comparisons.

Homepage: https://mera.a-ai.ru/
"""


from benchmark_tasks.custom_task import MultipleChoiceMERATask


class MathLogicQA(MultipleChoiceMERATask):
    VERSION = 0
    DATASET_NAME = "mathlogicqa"
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

    def doc_to_text_without_instruction(self, doc):
        prompt = "{text}\nA) {option_a}\nB) {option_b}\nC) {option_c}\nD) {option_d}\nОтвет:".format(
            **doc["inputs"]
        )
        return prompt.strip()
