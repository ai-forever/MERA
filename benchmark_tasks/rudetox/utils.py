from typing import Dict, List, TypedDict

import numpy as np
import torch
from scipy.interpolate import interp1d
from sklearn.isotonic import IsotonicRegression
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from lm_eval.api.filter import Filter
from lm_eval.api.registry import register_filter


def process_results(doc: Dict, results: List[str]) -> Dict:
    # - simple identity function that returns results
    if len(doc["outputs"]) > 0:
        sta, sim, fl, j, _ = results[0]
        return {"j": j, "sta": sta, "sim": sim, "fl": fl}
    return {"j": 0, "sta": 0, "sim": 0, "fl": 0}


class InterpolationParams(TypedDict):
    axis: int
    bounds_error: bool
    copy: bool
    fill_value: np.ndarray
    x: np.ndarray
    y: np.ndarray


class CalibratorParams(TypedDict):
    X_min_: float
    X_max_: float
    X_thresholds_: np.ndarray
    y_thresholds_: np.ndarray
    y_max: float
    y_min: float
    f_: interp1d
    increasing_: bool


class CalibratorSignature(TypedDict):
    out_of_bounds: str
    increasing: bool
    y_max: float
    y_min: float


@register_filter("rudetoxscoring")
class ruDetoxScoring(Filter):
    COLA_MODEL = "s-nlp/ruRoberta-large-RuCoLa-v1"
    MEANING_MODEL = "s-nlp/rubert-base-cased-conversational-paraphrase-v1"
    STYLE_MODEL = "IlyaGusev/rubertconv_toxic_clf"

    def __init__(self) -> None:
        """
        Can define custom behavior here, if an individual instantiation of a Filter class should have state.
        """

    def apply(self, resps, docs):
        """
        Assuming each entry of `resps` is a list of model responses, we discard all but the first response.
        """

        # TODO: specify device when passing one to filters becomes available
        available_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # loading models in apply so that they are not attributes of class
        # to avoid deleting them
        meaning_model = AutoModelForSequenceClassification.from_pretrained(
            self.MEANING_MODEL
        ).to(device=available_device)
        meaning_tokenizer = AutoTokenizer.from_pretrained(self.MEANING_MODEL)
        style_model = AutoModelForSequenceClassification.from_pretrained(
            self.STYLE_MODEL
        ).to(device=available_device)
        style_tokenizer = AutoTokenizer.from_pretrained(self.STYLE_MODEL)
        cola_model = AutoModelForSequenceClassification.from_pretrained(
            self.COLA_MODEL
        ).to(device=available_device)
        cola_tokenizer = AutoTokenizer.from_pretrained(self.COLA_MODEL)

        calibrator = self.get_calibrator()

        metrics = []
        for idx, obj in enumerate(resps):
            completion = obj[0].strip()
            accuracy = self.compute_accuracy(style_model, style_tokenizer, [completion])
            similarity = self.compute_similarity(
                meaning_model, meaning_tokenizer, [docs[idx]["inputs"]], [completion]
            )
            fluency = self.compute_fluency(cola_model, cola_tokenizer, [completion])
            accuracy = self.style_cal(calibrator, accuracy)
            similarity = self.meaning_cal(calibrator, similarity)
            fluency = self.fluency_cal(calibrator, fluency)
            joint = accuracy * similarity * fluency
            metrics.extend(
                [
                    [
                        accuracy.item(),
                        similarity.item(),
                        fluency.item(),
                        joint.item(),
                        completion,
                    ]
                ]
            )
        return metrics

    def get_calibrator(self):
        func_params: InterpolationParams = {
            "axis": 0,
            "bounds_error": False,
            "copy": True,
            "fill_value": np.array(np.nan),
            "x": np.array(
                [
                    0.00027650000000001285,
                    0.00034440000000002247,
                    0.00034559999999994595,
                    0.001278699999999966,
                    0.0012788000000000244,
                    0.0019349499999999908,
                    0.001937570000000055,
                    0.002486699999999953,
                    0.0024885000000000046,
                    0.002688770000000007,
                    0.0026899599999999912,
                    0.02320409999999995,
                    0.023239140000000047,
                    0.029833699999999963,
                    0.02988875000000002,
                    0.06509770000000004,
                    0.06567305000000001,
                    0.1304537,
                    0.13059336,
                    0.32918555000000005,
                    0.32961607000000004,
                    0.3611194999999999,
                    0.36121990000000004,
                    0.44504560000000004,
                    0.4457099,
                    0.6144483700000001,
                    0.6155048599999999,
                    0.65356693,
                    0.6536861,
                    0.65981287,
                    0.6614356299999999,
                    0.68675756,
                    0.68679786,
                    0.82462016,
                    0.82473633,
                    0.8333929,
                    0.8343167300000001,
                    0.8455976000000001,
                    0.84581335,
                    0.88914423,
                    0.88916642,
                    0.92163785,
                    0.92167094,
                    0.941321425,
                    0.941556983,
                    0.9431188070000001,
                    0.94315397,
                    0.9672734000000001,
                    0.9672936599999999,
                    0.973687772,
                    0.973703632,
                    0.973939763,
                    0.973964272,
                    0.97632647,
                    0.97637341,
                    0.98159888,
                    0.981640944,
                    0.983554833,
                    0.983579926,
                    0.990303015,
                    0.99031182,
                    0.9926355486,
                    0.9926386364,
                    0.9935122714,
                    0.9935136237,
                    0.993559784,
                    0.993560375,
                    0.995515268,
                    0.995516817,
                    0.9958820036,
                    0.9958824734,
                    0.9973993234,
                    0.9974000666,
                    0.9982809896,
                    0.9982815925,
                    0.9988658517,
                    0.9988669739,
                    0.9988697876,
                    0.9988699318,
                    0.998884157,
                    0.9988841575,
                    0.999144743,
                    0.99914501555,
                    0.9991559096,
                    0.9991560005,
                    0.99934383045,
                    0.999345196,
                    0.99939462944,
                    0.9993951180000001,
                    0.9994204223,
                    0.99942057085,
                    0.99951546398,
                    0.99951708468,
                    0.999558611,
                    0.99955952057,
                    0.99971446983,
                ],
                dtype=np.float64,
            ),
            "y": np.array(
                [
                    0.0,
                    0.0,
                    0.03896103896103896,
                    0.03896103896103896,
                    0.04522613065326633,
                    0.04522613065326633,
                    0.04678362573099415,
                    0.04678362573099415,
                    0.04918032786885246,
                    0.04918032786885246,
                    0.06148055207026349,
                    0.06148055207026349,
                    0.08139534883720931,
                    0.08139534883720931,
                    0.08627450980392157,
                    0.08627450980392157,
                    0.1827956989247312,
                    0.1827956989247312,
                    0.21987951807228914,
                    0.21987951807228914,
                    0.26666666666666666,
                    0.26666666666666666,
                    0.296,
                    0.296,
                    0.3129496402877698,
                    0.3129496402877698,
                    0.34146341463414637,
                    0.34146341463414637,
                    0.34615384615384615,
                    0.34615384615384615,
                    0.4166666666666667,
                    0.4166666666666667,
                    0.4439461883408072,
                    0.4439461883408072,
                    0.45,
                    0.45,
                    0.49056603773584906,
                    0.49056603773584906,
                    0.5181518151815182,
                    0.5181518151815182,
                    0.5494505494505495,
                    0.5494505494505495,
                    0.5570175438596491,
                    0.5570175438596491,
                    0.6,
                    0.6,
                    0.6490630323679727,
                    0.6490630323679727,
                    0.6725352112676056,
                    0.6725352112676056,
                    0.7142857142857143,
                    0.7142857142857143,
                    0.7171717171717171,
                    0.7171717171717171,
                    0.7177700348432056,
                    0.7177700348432056,
                    0.7183098591549296,
                    0.7183098591549296,
                    0.7589285714285714,
                    0.7589285714285714,
                    0.7815315315315315,
                    0.7815315315315315,
                    0.7972350230414746,
                    0.7972350230414746,
                    0.8181818181818182,
                    0.8181818181818182,
                    0.8354037267080745,
                    0.8354037267080745,
                    0.8714285714285714,
                    0.8714285714285714,
                    0.8748317631224765,
                    0.8748317631224765,
                    0.8763157894736842,
                    0.8763157894736842,
                    0.9046728971962616,
                    0.9046728971962616,
                    0.9090909090909091,
                    0.9090909090909091,
                    0.9230769230769231,
                    0.9230769230769231,
                    0.9404466501240695,
                    0.9404466501240695,
                    0.9523809523809523,
                    0.9523809523809523,
                    0.9620060790273556,
                    0.9620060790273556,
                    0.9647887323943662,
                    0.9647887323943662,
                    0.975609756097561,
                    0.975609756097561,
                    0.9760956175298805,
                    0.9760956175298805,
                    0.979381443298969,
                    0.979381443298969,
                    1.0,
                    1.0,
                ],
                dtype=np.float64,
            ),
        }
        params: CalibratorParams = {
            "X_min_": 0.00027650000000001285,
            "X_max_": 0.99971446983,
            "X_thresholds_": func_params["x"],
            "y_thresholds_": func_params["y"],
            "y_max": 1,
            "y_min": 0,
            "f_": interp1d(**func_params),
            "increasing_": True,
        }
        signature: CalibratorSignature = {
            "out_of_bounds": "clip",
            "increasing": True,
            "y_max": 1,
            "y_min": 0,
        }
        model = IsotonicRegression()
        model.set_params(**signature)
        for param_name, param_value in params.items():
            setattr(model, param_name, param_value)
        return model

    def compute_accuracy(
        self, style_model, style_tokenizer, completion, target_label=0, batch_size=32
    ):
        return self.evaluate_style_and_cola(
            model=style_model,
            tokenizer=style_tokenizer,
            texts=completion,
            target_label=target_label,
            batch_size=batch_size,
        )

    def compute_similarity(
        self,
        meaning_model,
        meaning_tokenizer,
        doc,
        completion,
        target_label="paraphrase",
        batch_size=32,
        bidirectional=False,
    ):
        return self.evaluate_meaning(
            model=meaning_model,
            tokenizer=meaning_tokenizer,
            original_texts=doc,
            rewritten_texts=completion,
            target_label=target_label,
            batch_size=batch_size,
            bidirectional=bidirectional,
            aggregation="prod",
        )

    def compute_fluency(
        self, cola_model, cola_tokenizer, completion, target_label=1, batch_size=32
    ):
        return self.evaluate_style_and_cola(
            model=cola_model,
            tokenizer=cola_tokenizer,
            texts=completion,
            target_label=target_label,
            batch_size=batch_size,
        )

    def style_cal(self, calibrator, x):
        return calibrator.predict(x[:, np.newaxis])

    def meaning_cal(self, calibrator, x):
        return calibrator.predict(x[:, np.newaxis])

    def fluency_cal(self, calibrator, x):
        return calibrator.predict(x[:, np.newaxis])

    def prepare_target_label(self, model, target_label):
        if target_label in model.config.id2label:
            pass  # needed so that labels from config do not fall in ValueError
        elif target_label in model.config.label2id:
            target_label = model.config.label2id.get(target_label)
        elif (
            str(target_label).isnumeric() and int(target_label) in model.config.id2label
        ):
            target_label = int(target_label)
        else:
            raise ValueError(
                f'target_label "{target_label}" not in model labels or ids: {model.config.id2label}.'
            )
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
    ):
        target_label = self.prepare_target_label(model, target_label)
        res = []

        filled_second_texts = second_texts is not None

        for i in range(0, len(texts), batch_size):
            inputs = [texts[i : i + batch_size]]

            if filled_second_texts:
                inputs.append(second_texts[i : i + batch_size])

            inputs = tokenizer(
                *inputs,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(model.device)

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

    def evaluate_style_and_cola(
        self, model, tokenizer, texts, target_label=1, batch_size=32
    ):
        target_label = self.prepare_target_label(model, target_label)
        scores = self.classify_texts(
            model, tokenizer, texts, target_label=target_label, batch_size=batch_size
        )
        return scores

    def evaluate_meaning(
        self,
        model,
        tokenizer,
        original_texts,
        rewritten_texts,
        target_label="entailment",
        batch_size=32,
        bidirectional=True,
        aggregation="prod",
    ):
        target_label = self.prepare_target_label(model, target_label)
        scores = self.classify_texts(
            model,
            tokenizer,
            original_texts,
            rewritten_texts,
            target_label=target_label,
            batch_size=batch_size,
        )
        if bidirectional:
            reverse_scores = self.classify_texts(
                model,
                tokenizer,
                rewritten_texts,
                original_texts,
                target_label=target_label,
                batch_size=batch_size,
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
