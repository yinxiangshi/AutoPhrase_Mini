import abc
from typing import Optional
from nltk.tokenize import sent_tokenize


class AbstractTokenizer(abc.ABC):
    @abc.abstractmethod
    def tokenize(self,
                 text: Optional[str] = None,
                 **kwargs):
        pass


class NLTKTokenizer(AbstractTokenizer):
    def tokenize(self,
                 text: Optional[str] = None,
                 **kwargs):
        text = text.lower()
        return sent_tokenize(text)
