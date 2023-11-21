import random

from lm_eval.base import LM


class DummyLM(LM):
    def __init__(self):
        pass

    @classmethod
    def create_from_arg_string(cls, arg_string, additional_config=None):
        return cls()

    def loglikelihood(self, requests):
        res = []

        for _ in requests:
            res.append((-random.random(), False))

        return res

    def greedy_until(self, requests, task=None, num_generation=1):
        res = []

        for ctx, _ in requests:
            if task == "humaneval":
                ans = ["lol" for i in range(num_generation)]
                res += [ans]
            else:
                res.append("lol")
            assert ctx.strip() != ""

        return res

    def loglikelihood_rolling(self, requests):
        res = []

        for _ in requests:
            res.append(-random.random())

        return res
