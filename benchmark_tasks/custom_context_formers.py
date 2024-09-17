from lm_eval.api.samplers import ContextSampler

### Copy of code from lm_eval_utils
from jinja2 import BaseLoader, Environment, StrictUndefined
import re


def regex_replace(string, pattern, repl, count: int = 0):
    """Implements the `re.sub` function as a custom Jinja filter."""
    return re.sub(pattern, repl, string, count=count)


env = Environment(
    loader=BaseLoader, undefined=StrictUndefined, keep_trailing_newline=True
)
env.filters["regex_replace"] = regex_replace
### End of copy
# TODO: make env creation a function to be imported
# TODO: make rendering of envs a function to use it

SPLIT_VALUE = "{context}"


class ruTiEContextFormer(ContextSampler):
    """
    This class is meant to redefine the behaviour of default ContextSampler.
    Sampler is called to pick few-shots from dataset[fewshot_split], join them
    in string or list of dictionaries (if chat_template is enabled).
    This class is not Sampler, but Former. No random is implied.

    How it works:
    1. The task.build_all_requests method is called. One doc comes into this Former.
    2. This doc has some id number - question_id. It shows how many documents
    precede this doc. We need to pick exactly (question_id - 1) docs as fewshots.
    3. The sample method takes these  (question_id - 1) docs from fewshot_split of
    the dataset.
    4.1. Without chat_template get_context method applies doc_to_text_without_instruction
    pattern to each doc, joins them in one string and puts inside "{context}" part of
    each test sample. The method itself returns empty string as the test sample is simply
    added to the fewshots, so we just put fewshots in the test sample's instruction!
    4.2. With chat_template, but without fewshot_as_multiturn one dictionary is
    formed out of the result of get_context method. It returns "", so the test sample
    with fewshots already inserted in the instruction is added to this empty string to form
    the valid dictionary.
    4.3. With chat_template and fewshot_as_multiturn each few-shot goes in a separate pair
    of dictionaries (user - assistant). So, the test sample instruction is split into two
    parts by "{context}" part where the context (where few-shots should have been added).
    Each few-shot doc is turned by doc_to_text_without_target pattern. The first part of
    the instruction is added to the first few-shot from the left. So that the first user
    prompt contains the beginning of the instruction. The other fewshots are added next
    as dictionary pairs. doc_to_target is used to form the assistant role content. The
    second part of the instruction is filled with test sample (doc) data. So that:

    was => test_doc instruction: "prompt_details1 - context - sample_data - prompt_details2"
    become => [{"prompt_details1 - fewshot_data1"}, {"fewshot_target1"},
    {"fewshot_data2"}, {"fewshot_target2"}, {"sample_data - prompt_details2"}]
    """

    def sample(self, n, doc):
        # id that tells how many previous samples to add as fewshots
        sample_id = doc["meta"]["question_id"]
        dialog_id = doc["meta"]["dialog_id"]

        # the dialogs are already sorted by process_docs, need to pick only the required samples
        start_idx = dialog_id * 500
        end_idx = start_idx + sample_id
        # choose only docs from the same dialog and assure they have q_id < doc[q_id]
        samples = self.docs[start_idx:end_idx]
        return samples

    def get_context(self, doc, num_fewshot):
        # draw `n_samples` docs from fewshot_docs
        fewshotex = self.sample(num_fewshot, doc)

        # pick jinja template for no_instruction format of task
        no_instruction_template = self.config.fewshot_config.get(
            "doc_to_text_without_instruction", self.config.doc_to_text
        )
        # make fillable env beforehand
        no_instruction_env = env.from_string(no_instruction_template)

        # split the instruction of the test question into two parts:
        # 1. part that contains some info and context tag only
        # 2. part that contains the question itself and all other tags (question, choice, etc.)
        first_part, second_part = doc["instruction"].split(SPLIT_VALUE)

        if len(fewshotex) == 0:
            labeled_examples = ""
        else:
            # the first fewshot should include the first part of the test question instruction
            # it is the imitation of placing all fewshots inside <context> tag of doc instruction
            labeled_examples = [
                first_part + no_instruction_env.render(**elem)
                if idx == 0
                else no_instruction_env.render(**elem)
                for idx, elem in enumerate(fewshotex)
            ]

        if len(fewshotex) != 0:
            # the first part is already at the beginning of user prompt
            # this second part is at the end with the test question itself
            doc["instruction"] = second_part
        else:
            # 0 samples to add as context = no context at all
            doc["instruction"] = doc["instruction"].replace(SPLIT_VALUE, "")

        # join in a string, add fewshot_delimiter to separate the last fewshot from test question
        labeled_examples = (
            self.fewshot_delimiter.join(labeled_examples) + self.fewshot_delimiter
        )

        return labeled_examples

    # used when apply_chat_template=True
    def get_chat_context(
        self,
        doc,
        num_fewshot,
        fewshot_as_multiturn: bool = False,
    ):
        chat_history = []

        # draw `n_samples` docs from fewshot_docs
        fewshotex = self.sample(num_fewshot, doc)

        # Load extra templates from config parameters
        no_instruction_template = self.config.fewshot_config.get(
            "doc_to_text_without_target", self.config.doc_to_text
        )
        only_target_template = self.config.doc_to_target

        # make fillable envs beforehand
        no_instruction_env = env.from_string(no_instruction_template)
        only_target_env = env.from_string(only_target_template)

        if not fewshot_as_multiturn:
            # get fewshot context as one user turn
            chat_history.extend(
                [
                    {
                        "role": "user",
                        "content": self.get_context(doc, num_fewshot),
                    }
                ]
            )
        else:
            if doc["meta"]["question_id"] > 0:
                first_part, second_part = doc["instruction"].split(SPLIT_VALUE)
                doc["instruction"] = second_part
                for idx, document in enumerate(fewshotex):
                    processed_doc = no_instruction_env.render(**document)
                    processed_target = only_target_env.render(**document)

                    if idx == 0:
                        processed_doc = first_part + processed_doc

                    chat_history.extend(
                        [
                            {
                                "role": "user",
                                "content": processed_doc,
                            },
                            {
                                "role": "assistant",
                                "content": processed_target,
                            },
                        ]
                    )

        return chat_history
