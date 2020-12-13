import string
from typing import Tuple, Dict, List, Optional
import numpy as np
import scipy.spatial
from tqdm import tqdm
from sklearn.decomposition import PCA

__all__ = [
    "load_glove_txt",
    "limit_vocab",
    "cosine_similarity",
    "compute_bias_by_projection",
]


def load_glove_txt(path: str) -> Tuple[np.ndarray, Dict[str, int], List[str]]:
    """
    Loads GloVe embeddings from txt file.
    """

    with open(path, "r", encoding="utf-8") as glove_file:
        lines = glove_file.readlines()

    word_vec = []
    vocab = []

    for line in lines:
        tokens = line.strip().split(" ")

        # `tokens` is a list which consists of the word and respective 300 dimension word vector
        assert len(tokens) == 301

        vocab.append(tokens[0])
        word_vec.append(tokens[1:])

    word2idx = {word: index for index, word in enumerate(vocab)}
    word_vec = np.array(word_vec).astype(float)
    print(f"word_vec shape: {word_vec.shape}, word2idx length: {len(word2idx)}, vocab length: {len(vocab)}")

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
    exclude: Optional[List[str]] = None,
) -> Tuple[np.ndarray, Dict[str, int], List[str]]:
    """
    Limits the vocabulary by removing the words given in `exclude`.
    """

    vocab_limited = []

    for word in tqdm(vocab[:50000]):
        if word.lower() != word:
            continue
        if len(word) >= 20:
            continue
        if has_digit(word):
            continue
        if "_" in word:
            punctuations = [has_punctuation(sub_word) for sub_word in word.split("_")]
            if not any(punctuations):
                vocab_limited.append(word)
            continue
        if has_punctuation(word):
            continue
        vocab_limited.append(word)

    if exclude:
        vocab_limited = list(set(vocab_limited) - set(exclude))

    print(f"Vocabulary size: {len(vocab_limited)}")

    word_vec_limited = np.zeros((len(vocab_limited), len(word_vec[0, :])))
    for i, word in enumerate(vocab_limited):
        word_vec_limited[i, :] = word_vec[word2idx[word], :]

    word2idx_limited = {word: i for i, word in enumerate(vocab_limited)}

    return word_vec_limited, word2idx_limited, vocab_limited


def cosine_similarity(word_vec1: np.ndarray, word_vec2: np.ndarray) -> float:
    """
    Computes Cosine Similarity between two word embeddings or vectors.

    Args:
        word_vec1: Word vector of a word.
        word_vec2: Word vector of a word.

    Returns:
        Cosine similarity coefficient.
    """

    return 1 - scipy.spatial.distance.cosine(word_vec1, word_vec2)


def compute_bias_by_projection(
    word_vec: np.ndarray,
    word2idx: Dict[str, int],
    vocab: List[str],
    he_embed: np.ndarray,
    she_embed: np.ndarray,
) -> Dict[str, float]:
    """
    Computes bias of each word by taking the difference of the word's similarity to `he` word embedding
    and the word's similarity to `she` word embedding.

    Args:
        word_vec: Word vector of different words.
        word2idx: Word to index mapping.
        vocab: Vocabulary.
    """

    gender_bias: dict = {}

    for word in vocab:
        vector = word_vec[word2idx[word]]
        gender_bias[word] = cosine_similarity(vector, he_embed) - cosine_similarity(vector, she_embed)

    return gender_bias


def perform_pca(pairs, word_vec, word2idx) -> PCA:
    """
    Performs PCA.
    """

    matrix = []
    cnt = 0

    if isinstance(pairs[0], list):
        for word1, word2 in pairs:
            if not (word1 in word2idx and word2 in word2idx):
                continue
            center = (word_vec[word2idx[word1], :] + word_vec[word2idx[word2], :]) / 2
            matrix.append(word_vec[word2idx[word1], :] - center)
            matrix.append(word_vec[word2idx[word2], :] - center)
            cnt += 1
    else:
        for word in pairs:
            if not word in word2idx:
                continue
            matrix.append(word_vec[word2idx[word], :])
            cnt += 1

        embeds = np.array(matrix)
        wv_mean = np.mean(np.array(embeds), axis=0)
        wv_hat = np.zeros(embeds.shape).astype(float)

        for i in range(len(embeds)):
            wv_hat[i, :] = embeds[i, :] - wv_mean
        matrix = wv_hat

    matrix = np.array(matrix)
    pca = PCA()
    pca.fit(matrix)
    print(f"Pairs used in PCA: {cnt}")
    return pca


def remove_vector_component(vector1, vector2):
    """
    Removes the projection of vector1 on vector2.
    """
    return vector1 - vector2 * vector1.dot(vector2) / vector2.dot(vector2)


def get_embeddings(words, word_vec, word2idx):
    """
    Get embeddings for the given words.
    """

    embeddings = [word_vec[word2idx[word]] for word in words]

    return embeddings
