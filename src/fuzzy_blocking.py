import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import awesome_cossim_topn


def build_char_tfidf(
    corpus,
    ngram_range=(3, 4),
    min_df=5,
    max_features=None,
):
    """
    Fit a character n-gram TF-IDF vectorizer.

    Character n-grams are useful for:
      - typos
      - transliteration variants
      - punctuation/casing differences
      - partial spelling overlap

    analyzer="char_wb" creates n-grams within word boundaries,
    which usually works well for entity names.
    """

    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=ngram_range,
        min_df=min_df,
        max_features=max_features,
        dtype=np.float32,
        norm="l2",
        lowercase=False,  # already normalized beforehand
    )

    matrix = vectorizer.fit_transform(corpus)

    return vectorizer, matrix


def retrieve_topk(
    query_matrix,
    corpus_matrix,
    top_k=50,
    lower_bound=0.0,
):
    """
    Return sparse cosine-similarity matrix containing only
    the top_k similarities per query row.

    Both matrices are L2-normalized by TF-IDF, so dot product
    is cosine similarity.
    """

    result = awesome_cossim_topn(
        query_matrix,
        corpus_matrix.T,
        ntop=top_k,
        lower_bound=lower_bound,
    )

    return result


def extract_ranked_candidates(
    similarity_matrix,
    corpus_entity_ids,
):
    """
    Convert sparse top-k similarity output into:

        [
            [(entity_id, score), ...],
            ...
        ]

    sorted descending by score for each query.
    """

    output = []

    for row_idx in range(similarity_matrix.shape[0]):

        start = similarity_matrix.indptr[row_idx]
        end = similarity_matrix.indptr[row_idx + 1]

        cols = similarity_matrix.indices[start:end]
        scores = similarity_matrix.data[start:end]

        order = np.argsort(-scores)

        ranked = [
            (
                corpus_entity_ids[cols[i]],
                float(scores[i]),
            )
            for i in order
        ]

        output.append(ranked)

    return output