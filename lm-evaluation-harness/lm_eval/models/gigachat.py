import transformers

try:
    from sber_hf import configuration, modeling
except ImportError:
    from unittest.mock import Mock

    configuration = modeling = Mock()

from .huggingface import AutoCausalLM


class GigaChatLM(AutoCausalLM):
    AUTO_CONFIG_CLASS: transformers.AutoConfig = configuration.RuGPTNeoXConfig
    AUTO_TOKENIZER_CLASS: transformers.AutoTokenizer = transformers.GPT2Tokenizer
    AUTO_MODEL_CLASS: transformers.AutoModel = modeling.RuGPTNeoXForCausalLM
