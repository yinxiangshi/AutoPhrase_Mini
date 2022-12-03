# AutoPhrase_Mini

This is a simplified version of AutoPhraseX(https://github.com/luozhouyang/AutoPhraseX)

# Warning: this repo may contains bug, I don't have time to fix them today.

## Usage

Before you start running this, you need to install `nltk` and `sklearn`

`conda create -n autophrase python=3.9`

`conda activate autophrase`

`conda install nltk`

`conda install scikit-learn`

Then, get in to the folder of this repo, create a script, do:

```python
import nltk

nltk.download()
```

After this, you can start running this:

Example:

```python
from autophrase import AutoPhrase
from tokenizer import NLTKTokenizer
from reader import DefaultCorpusReader
from selector import DefaultPhraseSelector
from extractor import NgramsExtractor, IDFExtractor, EntropyExtractor
from utils import add_stopwords

add_stopwords("Your_Stop_words_File_Path")

autophrase = AutoPhrase(
    reader=DefaultCorpusReader(tokenizer=NLTKTokenizer()),
    selector=DefaultPhraseSelector(),
    extractors=[
        NgramsExtractor(N=4),
        IDFExtractor(),
        EntropyExtractor()
    ]
)

predictions = autophrase.mine(
    corpus_file="Your_Dataset_Path",
    quality_file="Your_Quality_File_Path"
)

for pred in predictions:
    print(pred)
```