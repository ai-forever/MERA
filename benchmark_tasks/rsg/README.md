# RussianSuperGLUE

RussianSuperGLUE: A Russian Language Understanding Evaluation Benchmark

https://arxiv.org/abs/2010.15925

Modern universal language models and transformers such as BERT, ELMo, XLNet, RoBERTa
and others need to be properly compared and evaluated.
In the last year, new models and methods for pretraining and transfer learning have
driven striking performance improvements across a range of language understanding tasks.

We offer testing methodology based on tasks, typically proposed for “strong AI” — logic,
commonsense, reasoning. Adhering to the GLUE and SuperGLUE methodology,
we present a set of test tasks for general language understanding and leaderboard models.

For the first time a complete test for Russian language was developed,
which is similar to its English analog. Many datasets were composed for the first time,
and a leaderboard of models for the Russian language with comparable results is also presented.

## Description

Recent advances in the field of universal language models and transformers require the development of a methodology for
their broad diagnostics and testing for general intellectual skills - detection of natural language inference,
commonsense reasoning, ability to perform simple logical operations regardless of text subject or lexicon. For the first
time, a benchmark of nine tasks, collected and organized analogically to the SuperGLUE methodology, was developed from
scratch for the Russian language. We provide baselines, human level evaluation, an open-source framework for evaluating
models and an overall leaderboard of transformer models for the Russian language.

## RWSD

A Russian Winograd Schema Dataset (RWSD) is a task in which each example contains
a sentence with two selected phrases. The task is to define whether they are used
in the same sense or not. The schema takes its name from a well-known example by
Terry Winograd. See the https://russiansuperglue.com/tasks/task_info/RWSD
for the details.

## PARus

A Russian Winograd Schema Dataset (RWSD) is a task in which each example contains
a sentence with two selected phrases. The task is to define whether they are used
in the same sense or not. The schema takes its name from a well-known example by
Terry Winograd. See the https://russiansuperglue.com/tasks/task_info/RWSD
for the details.

## RCB

The Russian Commitment Bank (RCB) is a corpus of naturally occurring discourse whose final
sentence contains a clause-embedding predicate under an entailment canceling operator
(question, modal, negation, antecedent of conditional).

## Homepage

https://mera.a-ai.ru/

https://russiansuperglue.com

https://arxiv.org/abs/2010.15925

## Citation

```
@article{shavrina2020russiansuperglue,
    title={RussianSuperGLUE: A Russian Language Understanding Evaluation Benchmark},
    author={Shavrina, Tatiana and Fenogenova, Alena and Emelyanov, Anton and Shevelev, Denis and Artemova,
    Ekaterina and Malykh, Valentin and Mikhailov, Vladislav and Tikhonova, Maria and Chertok, Andrey and
    Evlampiev, Andrey},
    journal={arXiv preprint arXiv:2010.15925},
    year={2020}
}
```

## License

MIT License
