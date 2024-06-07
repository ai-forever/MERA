from src.registry import register_task
from src.tasks.task import Task
from src.metrics import mean
from src.utils import load_pickle
from typing import Dict, List, Optional, Union
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedTokenizer,
)
import numpy as np
import torch


def prepare_target_label(
    model: torch.nn.Module, target_label: Union[str, int]
) -> Union[str, int]:
    if target_label in model.config.id2label:
        pass
    elif target_label in model.config.label2id:
        target_label = model.config.label2id.get(target_label)
    elif target_label.isnumeric() and int(target_label) in model.config.id2label:
        target_label = int(target_label)
    else:
        raise ValueError(
            f'target_label "{target_label}" not in model labels or ids: {model.config.id2label}.'
        )
    return target_label


def classify_texts(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizer,
    texts: List[str],
    second_texts: Optional[List[str]] = None,
    target_label: Optional[Union[str, int]] = None,
) -> float:
    target_label = prepare_target_label(model, target_label)
    inputs = [texts]
    if second_texts is not None:
        inputs.append(second_texts)
    inputs = tokenizer(
        *inputs,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to(model.device)

    with torch.no_grad():
        logits = model(**inputs).logits
        if logits.shape[-1] > 1:
            preds = torch.softmax(logits, -1)[:, target_label]
        else:
            preds = torch.sigmoid(logits)[:, 0]
        preds = preds.view(-1).cpu().numpy()
    return preds


@register_task
class ruDetox(Task):
    def __init__(self, conf):
        super().__init__(conf)
        no_load_models = False
        if hasattr(self.conf, "no_load_models"):
            no_load_models = self.conf.no_load_models
        if not no_load_models:
            self.__load_models()

    def __load_models(self):
        self.style_model = AutoModelForSequenceClassification.from_pretrained(
            self.task_conf.style_model_path
        )
        self.style_model.to(self.task_conf.device)
        self.style_tokenizer = AutoTokenizer.from_pretrained(
            self.task_conf.style_model_path
        )

        style_calibration = load_pickle(self.task_conf.calibrations_ru_path)
        self.style_calibration = lambda x: style_calibration.predict(x[:, np.newaxis])

        self.meaning_model = AutoModelForSequenceClassification.from_pretrained(
            self.task_conf.meaning_model_path
        )
        self.meaning_model.to(self.task_conf.device)
        self.meaning_tokenizer = AutoTokenizer.from_pretrained(
            self.task_conf.meaning_model_path
        )

        meaning_calibration = load_pickle(self.task_conf.calibrations_ru_path)
        self.meaning_calibration = lambda x: meaning_calibration.predict(
            x[:, np.newaxis]
        )

        self.cola_model = AutoModelForSequenceClassification.from_pretrained(
            self.task_conf.cola_model_path
        )
        self.cola_model.to(self.task_conf.device)
        self.cola_tokenizer = AutoTokenizer.from_pretrained(
            self.task_conf.cola_model_path
        )

        fluency_calibration = load_pickle(self.task_conf.calibrations_ru_path)
        self.fluency_calibration = lambda x: fluency_calibration.predict(
            x[:, np.newaxis]
        )

    def evaluate_style(self, texts):
        target_label = prepare_target_label(self.style_model, 1)
        scores = classify_texts(
            self.style_model,
            self.style_tokenizer,
            [texts],
            target_label=target_label,
        )
        return float(self.style_calibration(scores))

    def evaluate_meaning(self, original_texts, rewritten_texts):
        target_label = prepare_target_label(self.meaning_model, "paraphrase")
        scores = classify_texts(
            self.meaning_model,
            self.meaning_tokenizer,
            [original_texts],
            [rewritten_texts],
            target_label=target_label,
        )
        return float(self.meaning_calibration(scores))

    def evaluate_cola(self, texts):
        target_label = prepare_target_label(self.cola_model, 1)
        scores = classify_texts(
            self.cola_model,
            self.cola_tokenizer,
            [texts],
            target_label=target_label,
        )
        return float(self.fluency_calibration(scores))

    def aggregation(self) -> Dict:
        return {"sta": mean, "sim": mean, "fl": mean, "j": mean}

    def process_results(self, doc_true, doc_pred) -> Dict:
        y_true = self.doc_to_y_true(doc_true)
        y_pred = self.doc_to_y_pred(doc_pred)
        sta = self.evaluate_style(y_pred)
        sim = self.evaluate_meaning(y_true, y_pred)
        fl = self.evaluate_cola(y_pred)
        j = sta * sim * fl
        return {"sta": sta, "sim": sim, "fl": fl, "j": j}

    def sample_submission(self):
        res = []
        for doc_id in self.gold.doc_ids():
            doc = {"outputs": self.gold[doc_id]["inputs"], "meta": {"id": doc_id}}
            res.append(doc)
        return {"data": {self.split: res}}
