import abc
import os
from typing import Optional

from tokenizer import AbstractTokenizer
from utils import ngrams
from extractor import FeatureExtractorWrapper


class AbstractCorpusReader(abc.ABC):

    @abc.abstractmethod
    def read(self,
             corpus_file: Optional[str] = None,
             *args,
             **kwargs):
        pass


class DefaultCorpusReader(AbstractCorpusReader):
    def __init__(self,
                 tokenizer: Optional[AbstractTokenizer] = None):
        super().__init__()
        self.tokenizer = tokenizer

    def read(self,
             corpus_file: Optional[str] = None,
             N_grams: Optional[int] = 4,
             verbose: Optional[bool] = True,
             extractor_wrapper: Optional[FeatureExtractorWrapper] = None,
             *args,
             **kwargs):
        if not os.path.exists(corpus_file):
            raise FileNotFoundError
        with open(corpus_file, "r") as f:
            for line in f:
                line = line.rstrip('\n')
                tokens = self.tokenizer.tokenize(line)
                for n in range(1, N_grams + 1):
                    for (start, end), window in ngrams(tokens, n):
                        extractor_wrapper.update_ngrams(start, end, window, n, **kwargs)
