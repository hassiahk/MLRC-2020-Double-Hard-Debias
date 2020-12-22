from typing import Dict
import numpy as np


class DoubleHardDebias:
    """
    Implements Double-Hard Debias.
    """
    def debias(self, word_vec: np.ndarray, word2idx: Dict[str, int]):
        """
        Debiasing the word embeddings.
        """
        return word_vec, word2idx

    def cluster_and_evaluate(self):
        """
        Evaluate the word embeddings after clustering.
        """
        pass

    def visualize(self):
        """
        Visualize the results.
        """
        pass
