import os
from tqdm import tqdm
from typing import List, Optional, Tuple
import copy

from lm_eval import utils
from lm_eval.base import LM, CacheHook
from lm_eval.utils import retry_on_specific_exceptions


def get_result(response) -> Tuple[float, bool]:
    """Process results from OpenAI API response.

    :param response: dict
        OpenAI API Response
    :param ctxlen: int
        Length of context (so we can slice them away and only keep the predictions)
    :return:
        continuation_logprobs: np.array
            Log probabilities of continuation tokens
        is_greedy: bool
            whether argmax matches given continuation exactly
    """
    logprobs = response.logprobs.token_logprobs
    for i in range(len(logprobs)):
        if logprobs[i] is not None:
            continuation_logprobs = sum(logprobs[i:])
            break

    # TODO: logic for greedy via passing context length
    is_greedy = False

    return continuation_logprobs, is_greedy


def oa_completion(**kwargs):
    """Query OpenAI API for completion.

    Retry with back-off until they respond
    """
    import openai

    def _exception_callback(e: Exception, sleep_time: float) -> None:
        import traceback

        traceback.print_exc()

    @retry_on_specific_exceptions(
        on_exceptions=[openai.OpenAIError],
        max_retries=None,  # retry forever, consider changing
        on_exception_callback=_exception_callback,
    )
    def completion():
        return openai.completions.create(**kwargs)

    return completion()


class OpenaiCompletionsLM(LM):
    REQ_CHUNK_SIZE = 20
    _DEFAULT_MAX_LENGTH = 2048

    def __init__(
        self,
        model: str,
        truncate: bool = False,
        max_gen_toks: int = 256,
        batch_size: int = 1,
        seed: int = 1234,
        max_length: Optional[int] = None,
    ) -> None:
        """

        :param engine: str
            OpenAI API engine (e.g. gpt-3.5-turbo-instruct)
        :param truncate: bool
            Truncate input if too long (if False and input is too long, throw error)
        """
        super().__init__()
        self.seed = seed
        try:
            import openai
            import tiktoken
        except ModuleNotFoundError:
            raise Exception(
                "attempted to use 'openai' LM type, but package `openai` or `tiktoken` are not installed. "
                "please install these via `pip install lm-eval[openai]` or `pip install -e .[openai]`"
            )
        self.model = model
        self.tokenizer = tiktoken.encoding_for_model(self.model)
        self.vocab_size = self.tokenizer.n_vocab
        self.truncate = truncate
        self.end_of_text_token_id = self.tokenizer.eot_token
        self._max_gen_toks = max_gen_toks
        self._max_length = max_length
        self._rank = 0
        self._world_size = 1
        self._batch_size = batch_size
        self.cache_hook = CacheHook(None)

        # Read from environment variable OPENAI_API_KEY
        openai.api_key = os.environ["OPENAI_API_KEY"]

    @property
    def eot_token_id(self):
        return self.end_of_text_token_id

    @property
    def max_length(self) -> int:
        if self._max_length:
            return self._max_length
        else:
            return self._DEFAULT_MAX_LENGTH

    @property
    def max_gen_toks(self) -> int:
        return self._max_gen_toks

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def device(self):
        # Isn't used because we override _loglikelihood_tokens
        raise NotImplementedError()

    def tok_encode(self, string: str) -> List[int]:
        return self.tokenizer.encode(string)

    def tok_decode(self, tokens: List[int]) -> str:
        return self.tokenizer.decode(tokens)

    def loglikelihood(self, requests, task=None) -> List[Tuple[float, bool]]:
        new_reqs = []
        for context, continuation in requests:
            ctx = self.tok_encode(context + continuation)
            new_reqs.append(((context, continuation), ctx))

        return self._loglikelihood_tokens(new_reqs)

    def _loglikelihood_tokens(self, requests, disable_tqdm: bool = False) -> List[Tuple[float, bool]]:
        res = []

        def _collate(x):
            # this doesn't efficiently handle last-token differences yet, but those are kinda annoying because
            # it's not guaranteed that the 100 or so logprobs we get to see actually contain all the continuations
            # we care about, and so we need some kind of backup for when it isn't
            toks = x[1]
            return -len(toks), tuple(toks)

        re_ord = utils.Reorderer(requests, _collate)
        reordered_requests = re_ord.get_reordered()

        for chunk in utils.chunks(
            tqdm(reordered_requests, disable=disable_tqdm),
            n=int(self.batch_size) if self.batch_size != "auto" and str(self.batch_size).isdigit() else 1,
        ):
            inps = []
            logs = []

            for cache_key, ctx in chunk:
                # max_length+1 because the API takes up to 2049 tokens, including the first context token
                inp = ctx[-(self.max_length + 1) :]
                inps.append(inp)

                logs_ctx = {}
                logs_ctx["inp_len"] = len(inp)
                logs.append(logs_ctx)

            response = oa_completion(
                model=self.model,
                prompt=inps,
                echo=True,
                max_tokens=0,
                temperature=0.0,
                logprobs=10,
                seed=self.seed,
            )

            for resp, (cache_key, ctx), lg in zip(response.choices, chunk, logs):
                answer = get_result(resp)

                res.extend([[answer, lg]])

                # partial caching
                if cache_key is not None:
                    self.cache_hook.add_partial("loglikelihood", cache_key, answer)
        return re_ord.get_original(res)

    def greedy_until(self, requests, task=None, num_generation=1) -> List[str]:
        if not requests:
            return []
        res = []

        def _collate(x):
            toks = self.tok_encode(x[0])
            return len(toks), x[0]

        re_ord = utils.Reorderer(requests, _collate)

        def sameuntil_chunks(xs, size):
            ret = []
            lastuntil = xs[0][1]
            for x in xs:
                if len(ret) >= size or x[1] != lastuntil:
                    yield ret, lastuntil
                    ret = []
                    lastuntil = x[1]
                ret.append(x)

            if ret:
                yield ret, lastuntil

        # todo: more intelligent batching for heterogeneous `until`
        override_bs = 1 if task == "humaneval" else 0
        for chunk, request_args in tqdm(
            list(
                sameuntil_chunks(
                    re_ord.get_reordered(),
                    int(self.batch_size) if not override_bs and str(self.batch_size).isdigit() else override_bs,
                )
            )
        ):
            inps = []
            logs = []
            self._max_gen_toks = request_args.pop("max_gen_toks", self.max_gen_toks)
            for context, _ in chunk:
                context_enc = self.tok_encode(context)
                inp = context_enc[-(self.max_length - self.max_gen_toks) :]
                inps.append(inp)
                logs_ctx = {"inp_len": len(inp), "max_gen_toks": self._max_gen_toks}
                logs.extend([logs_ctx])

            # store initial until list in case it is truncated for model
            until_init = request_args.pop("until", ["<|endoftext|>"])
            if isinstance(until_init, str):
                until_init = [until_init]
            until = copy.copy(until_init)
            if task == "humaneval":
                # optimal temp for k=10 for codex
                request_args["temperature"] = request_args.get("temperature", 0.6)
                if isinstance(until, list) and len(until) > 4:
                    # openai API does not allow more than 3 stops in an array
                    # so remove the shortest stops 
                    until.remove(sorted(until, key=len)[0])
                    until.remove(sorted(until, key=len)[0])
            else:
                request_args["temperature"] = request_args.get("temperature", 0)

            for elem in logs:
                elem["temp"] = request_args["temperature"]
                elem["until"] = until

            if task == "humaneval":
                response = []
                for it in range(num_generation):
                    candidate = oa_completion(
                        model=self.model,
                        prompt=inps,
                        max_tokens=self.max_gen_toks,
                        stop=until,
                        # ensure diversity of generations
                        seed=self.seed + it,
                        **request_args,
                    )
                    s = getattr(candidate.choices[0], "text")
                    until_ = until_init
                    for term in until_:
                        if len(term) > 0:
                            s = s.split(term)[0]
                    response.extend([s])

                # partial caching
                self.cache_hook.add_partial("generate_until", (chunk[0], {"until": until_}), response)

                res.extend([[response, logs[0]]])
            else:
                response = oa_completion(
                    model=self.model,
                    prompt=inps,
                    max_tokens=self.max_gen_toks,
                    stop=until,
                    seed=self.seed,
                    **request_args,
                )
                for resp, (context, args_), lg in zip(response.choices, chunk, logs):
                    s = getattr(resp, "text")

                    until_ = until_init

                    for term in until_:
                        if len(term) > 0:
                            s = s.split(term)[0]

                    # partial caching
                    self.cache_hook.add_partial("greedy_until", (context, {"until": until_}), s)

                    res.extend([[s, lg]])
        return re_ord.get_original(res)
    
    def loglikelihood_rolling(self, requests):
        # Isn't used because we override _loglikelihood_tokens
        raise NotImplementedError()

    def _model_call(self, inps):
        # Isn't used because we override _loglikelihood_tokens
        raise NotImplementedError()

    def _model_generate(self, context, max_length, eos_token_id):
        # Isn't used because we override generate_until
        raise NotImplementedError()
