from lm_eval.api.samplers import ContextSampler
from lm_eval.utils import apply_template


class FewshotSampler(ContextSampler):
    def sample(self, n, doc):
        # for ruMMLU need only domain docs as shots
        if doc["meta"].get("domain", False):
            docs_for_sampling = [
                item
                for item in self.docs
                if item["meta"]["domain"] == doc["meta"]["domain"]
            ]
        # for USE need the same id_task for all shots
        elif doc["meta"].get("id_task", False):
            docs_for_sampling = [
                item
                for item in self.docs
                if item["meta"]["id_task"] == doc["meta"]["id_task"]
            ]
        # for other tasks all docs are suitable as shots
        else:
            docs_for_sampling = self.docs
        # no random sampling for RuTie to preserve order of instances
        if doc["meta"].get("question_id", False):
            return docs_for_sampling[:n]
        return self.rnd.sample(docs_for_sampling, n)

    def get_context(self, doc, num_fewshot):
        # draw an extra fewshot sample if using same split as evaluating on
        n_samples = (
            num_fewshot + 1
            if self.config.fewshot_split == self.config.test_split
            else num_fewshot
        )

        # draw `n_samples` docs from fewshot_docs
        # need doc parameter to filter sample by metadata
        fewshotex = self.sample(n_samples, doc)

        # get rid of the doc that's the one we're evaluating, if it's in the fewshot
        selected_docs = [x for x in fewshotex if x != doc][:num_fewshot]

        # Load extra templates from fewshot config parameters
        no_instruction_template = self.config.fewshot_config.get(
            "doc_to_text_without_instruction", self.config.doc_to_text
        )
        # restore instructions in doc_to_text
        self.config.doc_to_text = self.config.fewshot_config.get(
            "query", no_instruction_template
        )

        # new format of constructing labeled_examples
        labeled_examples = ""
        for idx, doc in enumerate(selected_docs):
            doc_content = self.doc_to_text(doc)
            doc_target = self.doc_to_target(doc)
            labeled_examples += (
                (
                    doc_content
                    if self.config.doc_to_choice is None or isinstance(doc_content, str)
                    else self.doc_to_choice(doc)[doc_content]
                )
                if idx == 0
                else apply_template(no_instruction_template, doc)
            )
            labeled_examples += self.target_delimiter
            labeled_examples += (
                str(doc_target[0])
                if isinstance(doc_target, list)
                else doc_target
                if self.config.doc_to_choice is None or isinstance(doc_target, str)
                else str(self.doc_to_choice(doc)[doc_target])
            )
            labeled_examples += self.fewshot_delimiter

        # set doc_to_text to be instructionless to be usable with the next sample
        self.task.config.doc_to_text = no_instruction_template

        return labeled_examples

    # used when apply_chat_template=True
    def get_chat_context(
        self,
        doc,
        num_fewshot,
        fewshot_as_multiturn: bool = False,
    ):
        chat_history = []
        # draw an extra fewshot sample if using same split as evaluating on
        n_samples = (
            num_fewshot + 1
            if self.config.fewshot_split == self.config.test_split
            else num_fewshot
        )
        # draw `n_samples` docs from fewshot_docs
        # need doc parameter to filter sample by metadata
        fewshotex = self.sample(n_samples, doc)

        # get rid of the doc that's the one we're evaluating, if it's in the fewshot
        # TODO: should we just stop people from using fewshot from same split as evaluating?
        selected_docs = [x for x in fewshotex if x != doc][:num_fewshot]

        # Load extra templates from fewshot config parameters
        no_instruction_template = self.config.fewshot_config.get(
            "doc_to_text_without_instruction", self.config.doc_to_text
        )
        # restore instructions in doc_to_text
        self.config.doc_to_text = self.config.fewshot_config.get(
            "query", no_instruction_template
        )

        # construct dicts of examples
        if fewshot_as_multiturn:
            for idx, doc in enumerate(selected_docs):
                doc_content = self.doc_to_text(doc)
                doc_target = self.doc_to_target(doc)
                chat_history.append(
                    {
                        "role": "user",
                        "content": (
                            doc_content
                            if self.config.doc_to_choice is None
                            or isinstance(doc_content, str)
                            else self.doc_to_choice(doc)[doc_content]
                        )
                        if idx == 0
                        else apply_template(no_instruction_template, doc),
                    }
                )
                chat_history.append(
                    {
                        "role": "assistant",
                        "content": str(doc_target[0])
                        if isinstance(doc_target, list)
                        else doc_target
                        if self.config.doc_to_choice is None
                        or isinstance(doc_target, str)
                        else str(self.doc_to_choice(doc)[doc_target]),
                    }
                )
        else:
            # get fewshot context as one user turn
            chat_history.append(
                {"role": "user", "content": self.get_context(doc, num_fewshot)}
            )

        # set doc_to_text to be instructionless to be usable with the next sample
        self.task.config.doc_to_text = no_instruction_template

        return chat_history
