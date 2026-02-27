"""
Minimal local BM25 implementation used as an offline fallback.

This provides the `BM25Okapi` interface used by retrieval modules:
- constructor(tokenized_corpus)
- get_scores(tokenized_query)
"""

from collections import Counter, defaultdict
import math
from typing import Iterable, List, Sequence

import numpy as np


class BM25Okapi:
    def __init__(
        self,
        corpus: Sequence[Sequence[str]],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if corpus is None:
            raise ValueError("corpus cannot be None")

        self.k1 = float(k1)
        self.b = float(b)

        self.corpus_size = len(corpus)
        self.term_freqs: List[Counter] = []
        self.doc_len: List[int] = []
        self.df = defaultdict(int)
        self.idf = {}

        for doc in corpus:
            tokens = list(doc)
            counts = Counter(tokens)
            self.term_freqs.append(counts)
            self.doc_len.append(len(tokens))
            for term in counts:
                self.df[term] += 1

        self.avgdl = (sum(self.doc_len) / self.corpus_size) if self.corpus_size else 0.0
        self._build_idf()

    def _build_idf(self) -> None:
        # BM25 idf variant used by common implementations.
        for term, freq in self.df.items():
            self.idf[term] = math.log(
                1.0 + (self.corpus_size - freq + 0.5) / (freq + 0.5)
            )

    def get_scores(self, query_tokens: Sequence[str]) -> np.ndarray:
        if self.corpus_size == 0:
            return np.array([], dtype=float)

        scores = np.zeros(self.corpus_size, dtype=float)
        if self.avgdl <= 0:
            return scores

        for term in query_tokens:
            idf = self.idf.get(term)
            if idf is None:
                continue

            for i, tf in enumerate(self.term_freqs):
                freq = tf.get(term, 0)
                if freq <= 0:
                    continue

                dl = self.doc_len[i]
                denom = freq + self.k1 * (1.0 - self.b + self.b * dl / self.avgdl)
                if denom == 0:
                    continue
                scores[i] += idf * (freq * (self.k1 + 1.0) / denom)

        return scores

