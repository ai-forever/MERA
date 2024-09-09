from lm_eval.api.samplers import ContextSampler
from lm_eval.utils import apply_template


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
    added to the fewshots, so we jsut put fewshots in the test sample's instruction!
    4.2. With chat_template, but without fewshot_as_multiturn one dictionary is
    formed out of the result of get_context method. It returnes "", so the test sample
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
        # choose only docs from the same dialog and assure they are sorted by q_id
        docs_for_sampling = [
            elem for elem in self.docs if elem["meta"]["dialog_id"] == dialog_id
        ]
        docs_for_sampling = sorted(
            docs_for_sampling, key=lambda x: x["meta"]["question_id"]
        )
        # pick only previous docs from the same dialog in ascending order
        samples = [
            elem
            for elem in docs_for_sampling
            if elem["meta"]["question_id"] < sample_id
        ]
        return samples

    def get_context(self, doc, num_fewshot):
        # draw `n_samples` docs from fewshot_docs
        fewshotex = self.sample(num_fewshot, doc)

        # pick jinja template for no_instruction format of task
        no_instruction_template = self.config.fewshot_config.get(
            "doc_to_text_without_instruction", self.config.doc_to_text
        )

        # all shots in short versions joined in a string
        if len(fewshotex) == 0:
            joined_samples = ""
        else:
            samples = [
                apply_template(no_instruction_template, elem) for elem in fewshotex
            ]
            joined_samples = "\n" + self.fewshot_delimiter.join(samples) + "\n"

        # put fewshots inside the current test sample instruction instead of context
        doc["instruction"] = doc["instruction"].replace("{context}", joined_samples)

        # no need to return anything, fewshots are already place in doc's instruction
        return ""

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
                first_part, second_part = doc["instruction"].split("{context}")
                doc["instruction"] = second_part
                for idx, document in enumerate(fewshotex):
                    processed_doc = apply_template(no_instruction_template, document)
                    processed_target = apply_template(only_target_template, document)

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
