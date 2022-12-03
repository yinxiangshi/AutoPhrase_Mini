from typing import Optional
import os

from nltk.corpus import stopwords

STOPWORDS = stopwords
STOPWORDS = STOPWORDS.words()


def add_stopwords(file: Optional[str] = None) -> list:
    if not os.path.exists(file):
        raise FileNotFoundError
    with open(file, "r") as f:
        for line in f:
            line = line.rstrip()
            STOPWORDS.append(line)
    return STOPWORDS


def ngrams(sequence: Optional[list[str]] = None,
           n: Optional[int] = 2):
    start, end = 0, 0
    while end < len(sequence):
        end = start + n
        if end > len(sequence):
            return
        yield (start, end), tuple(sequence[start: end])
        start += 1
