import abc
import logging
from typing import Optional

from extractor import NgramsExtractor
from utils import STOPWORDS


class AbstractPhraseFilter(abc.ABC):

    @abc.abstractmethod
    def apply(self,
              pair: Optional[tuple] = None,
              **kwargs):
        """Filter phrase
        Args:
            pair: Python tuple of (phrase, freq)
        Returns:
            True if you need to drop this phrase, else False
        """
        return False

    @abc.abstractmethod
    def batch_apply(self,
                    batch_pairs: Optional[list[tuple]] = None,
                    **kwargs):
        """Filter a batch of phrases.
        Args:
            batch_pairs: List of tuple (phrase, freq)
        Returns:
            candidates: Filtered List of phrase tuple (phrase, freq)
        """
        return batch_pairs


class PhraseFilterWrapper(AbstractPhraseFilter):

    def __init__(self,
                 phrase_filters: Optional[list[AbstractPhraseFilter]] = None):
        super().__init__()
        self.filters = phrase_filters or []

    def apply(self,
              pair: Optional[tuple] = None,
              **kwargs):
        if any(f.apply(pair) for f in self.filters):
            return True
        return False

    def batch_apply(self,
                    batch_pairs: Optional[list[tuple]] = None,
                    **kwargs):
        candidates = batch_pairs
        for f in self.filters:
            candidates = f.batch_apply(candidates)
        return candidates


class DefaultPhraseFilter(AbstractPhraseFilter):
    def __init__(self,
                 min_len: Optional[int] = 2,
                 min_freq: Optional[int] = 3,
                 drop_stop_words: Optional[bool] = True) -> None:
        super().__init__()
        self.min_len = min_len
        self.min_freq = min_freq
        self.drop_stop_words = drop_stop_words

    def apply(self,
              pair: Optional[tuple] = None,
              **kwargs) -> bool:
        phrase, freq = pair
        if freq < self.min_freq:
            return True
        if len(phrase) < self.min_len:
            return True
        if self.drop_stop_words and ''.join(phrase.split(' ')) in STOPWORDS:
            return True
        return False

    def batch_apply(self,
                    batch_pairs: Optional[list[tuple]] = None,
                    **kwargs):
        raise NotImplementedError


class AbstractPhraseSelector(abc.ABC):

    @abc.abstractmethod
    def select(self, **kwargs):
        raise NotImplementedError()


class DefaultPhraseSelector(AbstractPhraseSelector):

    def __init__(self,
                 phrase_filters: Optional[list[AbstractPhraseFilter]] = None,
                 use_default_phrase_filters: Optional[bool] = True,
                 drop_stop_words: Optional[bool] = True,
                 min_freq: Optional[int] = 3,
                 min_len: Optional[int] = 2):
        """
        Args:
            drop_stop_words: Python boolean, filter stopwords or not.
            min_freq: Python int, min frequency of phrase occur in corpus.
            min_len: Python int, filter shot phrase whose length is less than this.
            phrase_filters: List of AbstractPhraseFilter, used to filter phrases
        """
        super().__init__()
        self.min_freq = min_freq
        self.min_len = min_len
        self.drop_stop_words = drop_stop_words
        if not phrase_filters and use_default_phrase_filters:
            self.phrase_filter = PhraseFilterWrapper(phrase_filters=[
                DefaultPhraseFilter(
                    min_len=self.min_len,
                    min_freq=min_freq,
                    drop_stop_words=drop_stop_words
                )
            ])
            logging.info('Using default phrase filters.')
        else:
            logging.info('Using custom phrase filters.')
            self.phrase_filter = PhraseFilterWrapper(phrase_filters)

    def select(self,
               extractors,
               topk: Optional[int] = 300,
               **kwargs):
        """Select topk frequent phrases.
        Args:
            extractors: List of AbstractFeatureExtractor, used to select frequent phrases
            topk: Python int, max number of phrases to select.
        Returns:
            phrases: Python list, selected frequent phrases from NgramsExtractor
        """
        ngrams_extractor = None
        for e in extractors:
            if isinstance(e, NgramsExtractor):
                ngrams_extractor = e
                break
        if ngrams_extractor is None:
            raise ValueError('Must provide an instance of NgramsExtractor!')

        candidates = []
        for n in range(1, ngrams_extractor.N + 1):
            counter = ngrams_extractor.ngrams_freq[n]
            for phrase, count in counter.items():
                if self.phrase_filter.apply((phrase, count)):
                    continue
                candidates.append((phrase, count))

        candidates = self.phrase_filter.batch_apply(candidates)
        candidates = sorted(candidates, key=lambda x: x[1], reverse=True)
        phrases = [x[0] for x in candidates[:topk]]
        return phrases
