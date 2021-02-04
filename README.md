## Double-Hard Debias
In this repo, we tried to reproduce the results claimed in [Double-Hard Debias: Tailoring Word Embeddings for Gender Bias Mitigation (ACL 2020)](https://arxiv.org/abs/2005.00965) as part of Reproducibility Challenge 2020 hosted by [PaperswithCode](https://paperswithcode.com/)

### Abstract of the Paper
> Word embeddings derived from human-generated corpora inherit strong gender bias which can be further amplified by downstream models. Some commonly adopted debiasing approaches, including the seminal Hard Debias algorithm, apply post-processing procedures that project pre-trained word embeddings into a subspace orthogonal to an inferred gender subspace. We discover that semantic-agnostic corpus regularities such as word frequency captured by the word embeddings negatively impact the performance of these algorithms. We propose a simple but effective technique, Double Hard Debias, which purifies the word embeddings against such corpus regularities prior to inferring and removing the gender subspace. Experiments on three bias mitigation benchmarks show that our approach preserves the distributional semantics of the pre-trained word embeddings while reducing gender bias to a significantly larger degree than prior approaches.

### Motivation
Despite widespread use in natural language processing (NLP) tasks, word embeddings have been criticized for inheriting unintended gender bias
from training corpora. [Bolukbasi et al. (2016)](https://arxiv.org/abs/1607.06520) highlights that in `word2vec` embeddings trained on the Google News dataset ([Mikolov et al., 2013a](https://arxiv.org/abs/1301.3781)), `programmer` is more closely associated with `man` and `homemaker` is more closely associated with `woman`. Such gender bias also propagates to downstream tasks. Studies have shown that coreference resolution systems exhibit gender bias in predictions due to the use of biased word embeddings ([Zhao et al., 2018a](https://arxiv.org/abs/1804.06876); [Rudinger et al., 2018](https://arxiv.org/abs/1804.09301)).

------------------------------------------------------------------------------------
## Usage

### Requirements
- `Python >= 3.6`.
- `Word Embeddings Benchmarks`. Install them following the instructions [here](https://github.com/kudkudak/word-embeddings-benchmarks).

### Installation
Clone the repo:
```bash
git clone https://github.com/uvavision/Double-Hard-Debias.git
```
Install the dependencies:
```bash
pip install -r requirements.txt
```
To run in develop mode, this is needed if you are just running our notebooks without changing anything:
```bash
python setup.py develop
```

### Data
- `Word Embeddings`: You can find the authors debiased embeddings and ours [here](https://drive.google.com/drive/folders/1ZCF075LCwW6Lq2Y-G9LXhCYqudaXfPRC). Download and keep them in the [data](https://github.com/hassiahk/Double-Hard-Debias/tree/main/data) folder.
- `Special Word Lists`: You can find them in the [data](https://github.com/hassiahk/Double-Hard-Debias/tree/main/data) folder.
- `Google Word Analogy` - Word Analogy dataset by Google. You can find it [here](https://drive.google.com/drive/folders/1V81RdUmueRaG9M_ZkBCpSEQwp0AldNE5).
- `MSR Word Analogy` - MSR Word Analogy dataset. You can find it [here](https://drive.google.com/drive/folders/1bc1bdIRwc12q-rVLXBm78cTICxJNfT9i).
- You can find all the external data used in the repo [here](https://drive.google.com/drive/folders/1ZCF075LCwW6Lq2Y-G9LXhCYqudaXfPRC).

### Double-Hard Debias
You can find the detailed step by step procedure to implement `Double-Hard Debias` in [`GloVe_Double_Hard_Debias.ipynb`](https://github.com/hassiahk/Double-Hard-Debias/blob/main/notebooks/GloVe_Double_Hard_Debias.ipynb). (`PyPi` package coming soon)

### Reproducibility Results
- In [`Normalized_Unnormalized_GloVe_Evaluate.ipynb`](https://github.com/hassiahk/Double-Hard-Debias/blob/main/notebooks/Normalized_Unnormalized_GloVe_Evaluate.ipynb), we experimented with both normalized and unnormalized embeddings to see which one gives better results.
- You can find the results for Double-Hard Debias and other debiasing approaches on `GloVe` in [`GloVe_Evaluate.ipynb`](https://github.com/hassiahk/Double-Hard-Debias/blob/main/notebooks/GloVe_Evaluate.ipynb).
- We also did some qualitative analysis by computing bias of some highly biased words before and after debiasing. You can find the analysis in [`Qualitative_Analysis.ipynb`](https://github.com/hassiahk/Double-Hard-Debias/blob/main/notebooks/Qualitative_Analysis.ipynb).
