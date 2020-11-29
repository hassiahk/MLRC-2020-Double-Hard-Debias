import numpy as np
import string
from typing import Dict, Tuple, List, Optional
from tqdm import tqdm

__all__ = [
    "load_glove_txt",
    "limit_vocab",
]


def load_glove_txt(path: str) -> Tuple[np.ndarray, Dict[str, int], List[str]]:
    """
    Loads GloVe embeddings from txt file.
    """

    with open(path) as f:
        lines = f.readlines()

    word_vec = []
    vocab = []

    for line in lines:
        tokens = line.strip().split(" ")

        # `tokens` is a list of word and 300 dimension word vector
        assert len(tokens) == 301

        vocab.append(tokens[0])
        word_vec.append([float(elem) for elem in tokens[1:]])

    word2idx = {word: index for index, word in enumerate(vocab)}
    word_vec = np.array(word_vec).astype(float)

    return word_vec, word2idx, vocab


def has_punctuation(word: str) -> bool:
    """
    Checks whether the word has any punctuations.
    """

    return any(char in string.punctuation for char in word)


def has_digit(word: str) -> bool:
    """
    Checks whether the word has any digits.
    """

    return any(char in "0123456789" for char in word)


def limit_vocab(
    word_vec: np.ndarray,
    word2idx: Dict[str, int],
    vocab: List[str],
    exclude: Optional[str] = None,
) -> Tuple[np.ndarray, Dict[str, int], List[str]]:
    """
    Limits the vocabulary by removing the words given in `exclude`.
    """

    vocab_limited = []
    for w in tqdm(vocab[:50000]):
        if w.lower() != w:
            continue
        if len(w) >= 20:
            continue
        if has_digit(w):
            continue
        if "_" in w:
            p = [has_punctuation(subw) for subw in w.split("_")]
            if not any(p):
                vocab_limited.append(w)
            continue
        if has_punctuation(w):
            continue
        vocab_limited.append(w)

    if exclude:
        vocab_limited = list(set(vocab_limited) - set(exclude))

    print("Size of vocabulary:", len(vocab_limited))

    wv_limited = np.zeros((len(vocab_limited), len(word_vec[0, :])))
    for i, w in enumerate(vocab_limited):
        wv_limited[i, :] = word_vec[word2idx[w], :]

    w2i_limited = {w: i for i, w in enumerate(vocab_limited)}

    return wv_limited, w2i_limited, vocab_limited
