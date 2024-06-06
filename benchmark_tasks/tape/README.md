# TAPE (Text Attack and Perturbation Evaluation)

TAPE: Assessing Few-shot Russian Language Understanding

https://arxiv.org/pdf/2210.12813.pdf

## Description

TAPE (Text Attack and Perturbation Evaluation) is a novel benchmark for few-shot
Russian language understanding evaluation that includes six complex NLU tasks, covering
multi-hop reasoning, ethical concepts, logic and commonsense knowledge.

## CheGeKa

The CheGeKa game setup is similar to Jeopardy. The player should come up with
the answer to the question basing on wit, commonsense and deep knowledge.
The task format is QA with a free response form and is based on the reviewed
unpublished data subsets by (Mikhalkova, 2021).

## MultiQ

Multi-hop reasoning has been the least addressed QA direction for Russian. We
have developed a semi-automatic pipeline for multi-hop dataset generation based
on Wikidata.
First, we extract the triplets from Wikidata and search for their intersections.
Two triplets (subject, verb, object) are needed to compose an answerable multi-hop
question. For instance, the question 'What continent is the country of which
Johannes Block was a citizen?' is formed by a sequence of five graph units: 'Block,
Johannes', 'citizenship', 'Germany', 'part of the world', 'Europe'. Second, several
hundreds of the question templates are curated by a few authors manually, which are
further used to fine-tune ruT5-largeto generate multi-hop questions given a
five-fold sequence. Third, the resulting questions undergo a paraphrasing and manual
validation procedure to control the quality and diversity. Finally, each question is
linked to two Wikipedia paragraphs, where all graph units appear in the natural
language. The task is to select the answer span using information from both
paragraphs.

## ruOpenBookQA

OpenBookQA for Russian is mainly based on the work of (Mihaylov et al., 2018):
it is a QA dataset with multiple-choice elementary-level science questions,
which probe the understanding of 1k+ core science facts. The dataset is mainly
composed of automatic translation and human validation and correction.

# ruWorldTree

The WorldTree task is very similar to the pipeline on the OpenBookQA, the main
difference being the additional lists of facts and the logical order that is
attached to the output of each answer to a question (Jansen et al., 2018).

## Homepage

https://mera.a-ai.ru

https://tape-benchmark.com

https://arxiv.org/pdf/2210.12813.pdf

## Citation

```
@article{taktasheva2022tape,
  title={TAPE: Assessing Few-shot Russian Language Understanding},
  author={Taktasheva, Ekaterina and Shavrina, Tatiana and Fenogenova, Alena and Shevelev, Denis and Katricheva, Nadezhda and Tikhonova, Maria and Akhmetgareeva, Albina and Zinkevich, Oleg and Bashmakova, Anastasiia and Iordanskaia, Svetlana and others},
  journal={arXiv preprint arXiv:2210.12813},
  year={2022}
}
```

## License

MIT License
