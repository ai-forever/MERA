"""
The RuDetox Diagnostic (ruDetox) dataset.

RuDetox Diagnostic is a part of RuDetox - a parallel corpus for text detoxification.
Given a sentence written in a toxic style, the model is asked to rewrite it in a polite
style preserving original meaning and fluency.

Homepage: https://mera.a-ai.ru/
"""

from lm_eval.base import Task, rf
from lm_eval.metrics import mean
import torch
import numpy as np


class ruDetox(Task):
    VERSION = 0
    DATASET_NAME = "rudetox"

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def training_docs(self):
        if self.has_training_docs():
            if self._training_docs is None:
                self._training_docs = list(self.dataset["train"])
            return self._training_docs

    def test_docs(self):
        if self.has_test_docs():
            return list(self.dataset["test"])

    def doc_to_target(self, doc):
        return " " + doc["outputs"]

    def doc_to_text(self, doc):
        prompt = doc["instruction"].format(toxic_comment=doc["inputs"]).strip()
        return prompt

    def should_decontaminate(self):
        return True

    def doc_to_decontamination_query(self, doc):
        return doc["inputs"]

    def construct_requests(self, doc, ctx):
        completion = rf.greedy_until(ctx, {"until": ["\n"]})
        return completion

    def process_results(self, doc, results, package):
        completion = results[0]
        completion = completion.strip()
        (
            meaning_model,
            meaning_tokenizer,
            style_model,
            style_tokenizer,
            cola_model,
            cola_tokenizer,
            style_calibrator,
            content_calibrator,
            fluency_calibrator,
        ) = package
        accuracy = self.evaluate_style(
            style_model,
            style_tokenizer,
            [completion],
            target_label=0,
            batch_size=32,
            verbose=False,
        )
        similarity = self.evaluate_meaning(
            meaning_model,
            meaning_tokenizer,
            [doc["inputs"]],
            [completion],
            batch_size=32,
            verbose=False,
            bidirectional=False,
            target_label="paraphrase",
        )
        fluency = self.evaluate_cola(
            cola_model,
            cola_tokenizer,
            texts=[completion],
            batch_size=32,
            verbose=False,
            target_label=1,
        )

        accuracy = self.style_cal(style_calibrator, accuracy)
        similarity = self.meaning_cal(content_calibrator, similarity)
        fluency = self.fluency_cal(fluency_calibrator, fluency)
        joint = accuracy * similarity * fluency

        return {
            "accuracy": np.mean(accuracy),
            "similarity": np.mean(similarity),
            "fluency": np.mean(fluency),
            "joint": np.mean(joint),
        }

    def aggregation(self):
        return {"accuracy": mean, "similarity": mean, "fluency": mean, "joint": mean}

    def higher_is_better(self):
        return {"accuracy": True, "similarity": True, "fluency": True, "joint": True}

    def style_cal(self, calibrator, x):
        return calibrator.predict(x[:, np.newaxis])

    def meaning_cal(self, calibrator, x):
        return calibrator.predict(x[:, np.newaxis])

    def fluency_cal(self, calibrator, x):
        return calibrator.predict(x[:, np.newaxis])

    def prepare_target_label(self, model, target_label):
        if target_label in model.config.id2label:
            pass
        elif target_label in model.config.label2id:
            target_label = model.config.label2id.get(target_label)
        elif target_label.isnumeric() and int(target_label) in model.config.id2label:
            target_label = int(target_label)
        else:
            raise ValueError(f'target_label "{target_label}" not in model labels or ids: {model.config.id2label}.')
        return target_label

    def classify_texts(
        self,
        model,
        tokenizer,
        texts,
        second_texts=None,
        target_label=None,
        batch_size=32,
        raw_logits=False,
        verbose=False,
    ):
        target_label = self.prepare_target_label(model, target_label)
        res = []

        for i in range(0, len(texts), batch_size):
            inputs = [texts[i : i + batch_size]]

            if second_texts is not None:
                inputs.append(second_texts[i : i + batch_size])

            inputs = tokenizer(*inputs, return_tensors="pt", padding=True, truncation=True, max_length=512).to(
                model.device
            )

            with torch.no_grad():
                try:
                    logits = model(**inputs).logits
                    if raw_logits:
                        preds = logits[:, target_label]
                    elif logits.shape[-1] > 1:
                        preds = torch.softmax(logits, -1)[:, target_label]
                    else:
                        preds = torch.sigmoid(logits)[:, 0]
                    preds = preds.cpu().numpy()
                except Exception:
                    print(i, i + batch_size)
                    preds = [0] * len(inputs)
            res.append(preds)
        return np.concatenate(res)

    def evaluate_style(
        self,
        model,
        tokenizer,
        texts,
        target_label=1,  # 1 is formal, 0 is informal
        batch_size=32,
        verbose=False,
    ):
        target_label = self.prepare_target_label(model, target_label)
        scores = self.classify_texts(
            model,
            tokenizer,
            texts,
            batch_size=batch_size,
            verbose=verbose,
            target_label=target_label,
        )
        return scores

    def evaluate_meaning(
        self,
        model,
        tokenizer,
        original_texts,
        rewritten_texts,
        target_label="entailment",
        bidirectional=True,
        batch_size=32,
        verbose=False,
        aggregation="prod",
    ):
        target_label = self.prepare_target_label(model, target_label)
        scores = self.classify_texts(
            model,
            tokenizer,
            original_texts,
            rewritten_texts,
            batch_size=batch_size,
            verbose=verbose,
            target_label=target_label,
        )
        if bidirectional:
            reverse_scores = self.classify_texts(
                model,
                tokenizer,
                rewritten_texts,
                original_texts,
                batch_size=batch_size,
                verbose=verbose,
                target_label=target_label,
            )
            if aggregation == "prod":
                scores = reverse_scores * scores
            elif aggregation == "mean":
                scores = (reverse_scores + scores) / 2
            elif aggregation == "f1":
                scores = 2 * reverse_scores * scores / (reverse_scores + scores)
            else:
                raise ValueError('aggregation should be one of "mean", "prod", "f1"')
        return scores

    def encode_cls(
        self,
        texts,
        model,
        tokenizer,
        batch_size=32,
        verbose=False,
    ):
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            with torch.no_grad():
                out = model(**tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(model.device))
                embeddings = out.pooler_output
                embeddings = torch.nn.functional.normalize(embeddings).cpu().numpy()
                results.append(embeddings)
        return np.concatenate(results)

    def evaluate_cosine_similarity(
        self,
        model,
        tokenizer,
        original_texts,
        rewritten_texts,
        batch_size=32,
        verbose=False,
    ):
        scores = (
            self.encode_cls(
                original_texts,
                model=model,
                tokenizer=tokenizer,
                batch_size=batch_size,
                verbose=verbose,
            )
            * self.encode_cls(
                rewritten_texts,
                model=model,
                tokenizer=tokenizer,
                batch_size=batch_size,
                verbose=verbose,
            )
        ).sum(1)
        return scores

    def evaluate_cola(
        self,
        model,
        tokenizer,
        texts,
        target_label=1,
        batch_size=32,
        verbose=False,
    ):
        target_label = self.prepare_target_label(model, target_label)
        scores = self.classify_texts(
            model,
            tokenizer,
            texts,
            batch_size=batch_size,
            verbose=verbose,
            target_label=target_label,
        )
        return scores

    def evaluate_cola_relative(
        self,
        model,
        tokenizer,
        original_texts,
        rewritten_texts,
        target_label=1,
        batch_size=32,
        verbose=False,
        maximum=0,
    ):
        target_label = self.prepare_target_label(model, target_label)
        original_scores = self.classify_texts(
            model,
            tokenizer,
            original_texts,
            batch_size=batch_size,
            verbose=verbose,
            target_label=target_label,
        )
        rewritten_scores = self.classify_texts(
            model,
            tokenizer,
            rewritten_texts,
            batch_size=batch_size,
            verbose=verbose,
            target_label=target_label,
        )
        scores = rewritten_scores - original_scores
        if maximum is not None:
            scores = np.minimum(0, scores)
        return scores

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def rotation_calibration(
        self,
        data,
        coef=1.0,
        px=1,
        py=1,
        minimum=0,
        maximum=1,
    ):
        result = (data - px) * coef + py
        if minimum is not None:
            result = np.maximum(minimum, result)
        if maximum is not None:
            result = np.minimum(maximum, result)
        return result
