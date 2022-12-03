import os
import random
from copy import deepcopy
from tqdm import tqdm
from typing import Optional, Any
from reader import AbstractCorpusReader
from selector import AbstractPhraseSelector, DefaultPhraseSelector

from sklearn.ensemble import RandomForestClassifier
from extractor import FeatureExtractorWrapper, AbstractFeatureExtractor


def load_quality_file(file: Optional[str] = None):
    if not os.path.exists(file):
        raise FileNotFoundError

    phrases = set()
    with open(file, "r") as f:
        for line in f:
            line = line.rstrip("\n")
            phrases.add(line.strip())
    return phrases


def _organize_phrases_pool(quality_phrases: Optional[set] = None,
                           frequent_phrases: Optional[set] = None) -> tuple[list, list]:
    pos_pool, neg_pool = [], []
    for p in frequent_phrases:
        if p in quality_phrases:
            pos_pool.append(p)
            continue
        _p = ''.join(p.split(' '))
        if _p in quality_phrases:
            pos_pool.append(p)
            continue
        neg_pool.append(p)
    return pos_pool, neg_pool


class AutoPhrase:
    def __init__(self,
                 reader: Optional[AbstractCorpusReader] = None,
                 selector: Optional[AbstractPhraseSelector] = None,
                 classifier: Optional[Any] = None,
                 threshold: Optional[float] = .4,
                 extractors: Optional[list[AbstractFeatureExtractor]] = None,
                 **kwargs):
        self.reader = reader
        self.selector = selector or DefaultPhraseSelector()
        self.threshold = threshold
        self.classifier = classifier or RandomForestClassifier(**kwargs)
        self.extractors = extractors
        self.extractor_wrapper = FeatureExtractorWrapper(self.extractors)

    def mine(self,
             corpus_file: Optional[str] = None,
             quality_file: Optional[str] = None,
             N_grams: Optional[int] = 4,
             epochs: Optional[int] = 10,
             topk: Optional[int] = 300,
             verbose: Optional[bool] = True):
        self.reader.read(corpus_file, N_grams, verbose)
        quality_phases = load_quality_file(quality_file)
        frequent_phrases = self.selector.select(
            extractors=self.extractors,
            topk=topk
        )
        init_pos_pool, init_neg_pool = _organize_phrases_pool(quality_phases, frequent_phrases)
        pos_pool, neg_pool = init_pos_pool, init_neg_pool

        for _ in tqdm(range(epochs)):
            x, y = self._prepare_training_data(pos_pool, neg_pool)
            self.classifier.fit(x, y)
            pos_pool, neg_pool = self._reorganize_phrase_pools(pos_pool, neg_pool)

        predictions = sorted(self._predict_proba(init_neg_pool), key=lambda X: X[1], reverse=True)
        return predictions

    def _prepare_training_data(self,
                               pos_pool: Optional[list] = None,
                               neg_pool: Optional[list] = None) -> tuple[list, list]:
        x, y = [], []
        examples = []
        for p in pos_pool:
            examples.append((self._compose_feature(p), 1))
        for p in neg_pool:
            examples.append((self._compose_feature(p), 0))
        random.shuffle(examples)
        for _x, _y in examples:
            x.append(_x)
            y.append(_y)
        return x, y

    def _compose_feature(self, phrase: Optional[str] = None) -> list:
        features = self.extractor_wrapper.extract(phrase)
        features = sorted(features.items(), key=lambda x: x[0])
        features = [x[1] for x in features]
        return features

    def _reorganize_phrase_pools(self,
                                 pos_pool: Optional[list] = None,
                                 neg_pool: Optional[list] = None) -> tuple[list, list]:
        new_pos_pool, new_neg_pool = [], []
        new_pos_pool.extend(deepcopy(pos_pool))

        pairs = self._predict_proba(neg_pool)
        pairs = sorted(pairs, key=lambda x: x[1], reverse=True)

        for idx, (p, prob) in enumerate(pairs):
            if prob > self.threshold:
                new_pos_pool.append(p)
                continue
            new_neg_pool.append(p)

        return new_pos_pool, new_neg_pool

    def _predict_proba(self,
                       phrases: Optional[list[str]] = None) -> list[tuple[str, Any]]:
        features = [self._compose_feature(phrase) for phrase in phrases]
        pos_probs = [prob[1] for prob in self.classifier.predict_proba(features)]
        pairs = [(phrase, prob) for phrase, prob in zip(phrases, pos_probs)]
        return pairs
