import os

from functools import partial
from collections import defaultdict
from datasets import DatasetDict, load_dataset


# pass token to access private datasets
HF_TOKEN = os.environ.get("HF_TOKEN", False)
load_dataset = partial(load_dataset, token=HF_TOKEN, cache_dir=".cache")


def load_data(self, **kwargs):
    
    if self.data_loaded:
        return

    self.dataset = load_dataset(
        self.description["hf_hub_name"], revision=self.description.get("revision", None)
    )

    self.data_loaded = True

def load_retrieval_data(hf_hub_name, eval_splits):
    eval_split = eval_splits[0]
    corpus_dataset = load_dataset(hf_hub_name, 'corpus')
    queries_dataset = load_dataset(hf_hub_name, 'queries')
    qrels = load_dataset(hf_hub_name + '-qrels')[eval_split]

    corpus = {e['_id']: {'text': e['text']} for e in corpus_dataset['corpus']}
    queries = {e['_id']: e['text'] for e in queries_dataset['queries']}
    relevant_docs = defaultdict(dict)
    for e in qrels:
        relevant_docs[e['query-id']][e['corpus-id']] = e['score']

    corpus = DatasetDict({eval_split:corpus})
    queries = DatasetDict({eval_split:queries})
    relevant_docs = DatasetDict({eval_split:relevant_docs})
    return corpus, queries, relevant_docs