from itertools import chain
import torch
import numpy as np
from splade.tasks.transformer_evaluator import SparseRetrieval
import time
from tqdm.auto import tqdm
import math
import re


def get_splade_vector_for_query(query_text, model, tokenizer):
    """
    Generates a SPLADE sparse vector for a given text query.
    The output is a dictionary compatible with Pyserini's impact search.
    """
    # print(f"Generating SPLADE vector for query: '{query_text}'")
    
    tokens = tokenizer(query_text, return_tensors='pt')
    
    with torch.no_grad():
        logits = model(**tokens).logits

    # Apply the SPLADE transformation: log(1 + ReLU(logits)) * attention_mask
    sparse_vec = torch.log(1 + torch.relu(logits[0])) * tokens['attention_mask'][0].unsqueeze(-1)
    sparse_vec = torch.max(sparse_vec, dim=0).values  # Aggregate across tokens
    
    # Get the non-zero dimensions and their weights
    indices = sparse_vec.nonzero().squeeze()
    weights = sparse_vec[indices].squeeze()

    # Create the query vector dictionary for Pyserini
    # IMPORTANT: The keys must be the string representation of the token IDs,
    # matching the format of your index.jsonl file.
    query_vector = {
        idx.item(): round(weight.item(), 4)
        for idx, weight in zip(indices, weights)
    }
    
    query_vector_tokens = {
        tokenizer.convert_ids_to_tokens(int(idx)): round(weight.item(), 4)
        for idx, weight in zip(indices, weights)
    }
    
    # print(f"Generated a sparse vector with {len(query_vector)} non-zero dimensions.")
    return query_vector, query_vector_tokens


def get_splade_vectors_for_queries(queries, model, tokenizer, max_length=256, agg='max'):
    """
    Generates SPLADE sparse vectors for a batch of text queries.
    The output is a list of dictionaries compatible with Pyserini's impact search.
    """
    tokens = tokenizer(queries,
                       add_special_tokens=True,
                       padding="longest",
                       truncation="longest_first",
                       max_length=max_length,
                       return_attention_mask=True,
                       return_tensors='pt')
    tokens = {key: value.to(model.device) for key, value in tokens.items()}
    
    with torch.no_grad():
        logits = model(**tokens).logits
    
    if agg == 'max':
        sparse_vecs = torch.log(1 + torch.relu(logits)) * tokens['attention_mask'].unsqueeze(-1)
        sparse_vecs = torch.max(sparse_vecs, dim=1).values  # Aggregate across tokens for each query
    elif agg == 'sum':
        sparse_vecs = torch.log(1 + torch.relu(logits)) * tokens['attention_mask'].unsqueeze(-1)
        sparse_vecs = torch.sum(sparse_vecs, dim=1)  # Aggregate across tokens for each query
    elif agg == 'cls':
        sparse_vecs = torch.log(1 + torch.relu(logits[:,0,:]))  # Use CLS token representation
    else:
        raise ValueError(f"Unknown aggregation method: {agg}")
    
    query_vectors = []
    for sparse_vec in sparse_vecs:
        indices = sparse_vec.nonzero().squeeze()
        weights = sparse_vec[indices].squeeze()
        query_vector = {
            idx.item(): weight.item()
            for idx, weight in zip(indices, weights)
        }
        query_vectors.append(query_vector)
    
    return query_vectors, tokens


def get_splade_vectors_for_queries_pre_tokenized(queries, model):
    """
    Generates SPLADE sparse vectors for a batch of pre-tokenized queries.
    The output is a list of dictionaries compatible with Pyserini's impact search.
    """

    with torch.no_grad():
        logits = model(**queries).logits

    sparse_vecs = torch.log(1 + torch.relu(logits)) * queries['attention_mask'].unsqueeze(-1)
    sparse_vecs = torch.max(sparse_vecs, dim=1).values  # Aggregate across tokens for each query

    query_vectors = []
    for sparse_vec in sparse_vecs:
        indices = sparse_vec.nonzero().squeeze()
        weights = sparse_vec[indices].squeeze()
        query_vector = {
            idx.item(): weight.item()
            for idx, weight in zip(indices, weights)
        }
        query_vectors.append(query_vector)

    return query_vectors


class SpladeRetriever(SparseRetrieval):
    
    def __init__(self, index_dir, tokenizer, collection=None):
        self.tokenizer = tokenizer
        self.collection = collection  # Store the collection for text lookup
        vocab_size = tokenizer.vocab_size
        
        class DummyModel:
            def __init__(self, vocab_size):
                self.output_dim = vocab_size
                
            def eval(self):
                pass
                
            def to(self, device):
                return self
        
        dummy_model = DummyModel(vocab_size)
        
        # Create a minimal config for the parent class
        config = {
            "index_dir": index_dir,
            "out_dir": "/tmp"  # We won't use this for file output
        }
        
        # Initialize parent class
        super().__init__(
            model=dummy_model, 
            config=config, 
            dim_voc=vocab_size,
            compute_stats=False,
            restore=False  # No model to restore
        )
        
        print(f"Index loaded successfully!")
        print(f"Index contains {len(self.sparse_index)} posting lists")
        print(f"Index contains {len(self.doc_ids)} documents")
        if self.collection:
            print(f"Collection with {len(self.collection)} documents loaded for text retrieval")
    
    def _get_document_text(self, doc_id):
        """
        Get document text by doc_id from the collection
        
        Args:
            doc_id: Document ID to look up
            
        Returns:
            str: Document text or None if not found
        """
        return self.collection[doc_id]['text'] if self.collection and doc_id < len(self.collection) else None
    
    def _convert_token_dict_to_vectors(self, query_dict):
        token_ids = []
        scores = []
        
        for token, score in query_dict.items():
            if isinstance(token, str):
                # Convert string token to ID using tokenizer
                if self.tokenizer is not None:
                    token_id = self.tokenizer.vocab.get(token, None)
                    if token_id is not None:
                        token_ids.append(token_id)
                        scores.append(float(score))
                    else:
                        print(f"Warning: Token '{token}' not found in vocabulary")
                else:
                    print(f"Warning: Cannot convert string token '{token}' without tokenizer")
            else:
                # Assume it's already an integer token ID
                token_ids.append(int(token))
                scores.append(float(score))
        
        return np.array(token_ids, dtype=np.int32), np.array(scores, dtype=np.float32)
    
    def retrieve_fast(self, query_representation, top_k=10, threshold=0.0):
        col_np, values_np = self._convert_token_dict_to_vectors(query_representation)

        filtered_indexes, scores = self.numba_score_float(
            self.numba_index_doc_ids,
            self.numba_index_doc_values,
            col_np,
            values_np,
            threshold=threshold,
            size_collection=self.sparse_index.nb_docs()
        )

        filtered_indexes, scores = self.select_topk(filtered_indexes, scores, k=top_k)
        sorted_indexes = np.argsort(scores)[::-1]
        filtered_indexes, scores = filtered_indexes[sorted_indexes], scores[sorted_indexes]

        return filtered_indexes, scores

    def retrieve(self, query_representation, top_k=10, threshold=0.0):
        
        # start_time = time.time()
        # Convert the query representation to numpy arrays
        col_np, values_np = self._convert_token_dict_to_vectors(query_representation)
        # conversion_time = time.time() - start_time
        
        # start_time = time.time()
        # Perform retrieval using the parent class's numba function
        filtered_indexes, scores = self.numba_score_float(
            self.numba_index_doc_ids,
            self.numba_index_doc_values,
            col_np,
            values_np,
            threshold=threshold,
            size_collection=self.sparse_index.nb_docs()
        )
        # scoring_time = time.time() - start_time
        
        # start_time = time.time()
        # Select top-k using the parent class's method
        filtered_indexes, scores = self.select_topk(filtered_indexes, scores, k=top_k)
        sorted_indexes = np.argsort(scores)[::-1]
        filtered_indexes, scores = filtered_indexes[sorted_indexes], scores[sorted_indexes]
        # print(filtered_indexes, scores)
        # topk_time = time.time() - start_time
        
        # start_time = time.time()
        # Convert document indices to document IDs and get text
        results = []
        for doc_idx, score in zip(filtered_indexes, scores):
            doc_id = self.doc_ids[doc_idx]
            doc_text = self._get_document_text(doc_id) if self.collection else None
            
            if doc_text is not None:
                results.append((doc_id, float(score), doc_text))
            else:
                results.append((doc_id, float(score), None))
        
        # formatting_time = time.time() - start_time
        
        # print(f"Time taken - Conversion: {conversion_time:.4f}s, Scoring: {scoring_time:.4f}s, Top-k: {topk_time:.4f}s, Formatting: {formatting_time:.4f}s")
        
        return results
    
    
# def rank_expansion_tokens_by_tf_idf_fast(query_vector_tokens, top_docs, token_idfs, tokenizer):
#     # Compute term frequencies for query tokens in the top documents
#     tf_scores = {token: 0 for token in query_vector_tokens}

#     top_doc_texts = [doc_text.lower() for _, _, doc_text in top_docs]
#     tokenized_docs = tokenizer(
#         top_doc_texts,
#         truncation=True,
#         add_special_tokens=False,
#         return_attention_mask=False,
#         return_token_type_ids=False
#     )['input_ids']
    
#     tokenized_docs = list(chain.from_iterable(tokenized_docs))
#     doc_length_sum = len(tokenized_docs)
#     for token in query_vector_tokens:
#         tf_scores[token] = tokenized_docs.count(token)

#     # Rank tokens by their TF-IDF scores
#     tf_idf_scores = {token: tf / doc_length_sum * token_idfs.get(token, 0) for token, tf in tf_scores.items()}
#     ranked_tokens = sorted(tf_idf_scores.items(), key=lambda item: item[1], reverse=True)
#     return ranked_tokens


def get_idf(term_df, collection_size):
    return np.log((collection_size + 1) / (term_df + 1)) + 1


def get_tf(tokenized_doc, token):
    tf = tokenized_doc.count(token) / (np.log(1 + len(tokenized_doc) + 1e-10))
    if np.isnan(tf):
        print(tokenized_doc, token, tokenized_doc.count(token), np.log(1 + len(tokenized_doc)))
    return tf


def get_decay(position):
    return 1 / np.log(2 + position)


def tokenize_rm3_query(query):
    tokens = re.findall(r'([^\s\^]+)(?:\^([\d\.]+))?', query)[1:]
    tokens = [item[0] for item in tokens]
    return tokens


def tokenize_and_stem(text, string_reader, tokenizer, stemmer, stop):
    # text = text.encode("latin1").decode("utf-8", "ignore").lower()
    text = text.lower() 
    java_reader = string_reader(text)
    token_stream = tokenizer.tokenise(java_reader)
    
    tokens = []
    while True:
        t = token_stream.next()
        if t is None: 
            break
        if t in stop:
            continue
        stemmed_t = stemmer.stem(t)
        tokens.append(stemmed_t)
    
    if len(tokens) == 0:
        print(f"Warning: No tokens found for text: '{text}'")
    return tokens


def rank_expansion_tokens_by_tf_idf_fast(query_vector_tokens, top_docs, token_dfs, collection_size, tokenizer, normalize_scores=False):
    # Compute term frequencies for query tokens in the top documents
    tf_scores = {token: 0 for token in query_vector_tokens}

    top_doc_texts = [doc_text.lower() for _, _, doc_text in top_docs]
    tokenized_docs = tokenizer(
        top_doc_texts,
        truncation=True,
        add_special_tokens=False,
        return_attention_mask=False,
        return_token_type_ids=False
    )['input_ids']

    for token in query_vector_tokens:
        for idx, doc_tokens in enumerate(tokenized_docs):
            tf_scores[token] += get_tf(doc_tokens, token) * get_decay(idx)
    
    # Rank tokens by their TF-IDF scores
    tf_idf_scores = {token: tf * get_idf(token_dfs.get(token, 0), collection_size) for token, tf in tf_scores.items()}
    for token, score in tf_idf_scores.items():
        if np.isnan(score):
            print(f"Warning: TF-IDF score for token '{token}' is NaN (TF: {tf_scores[token]}, DF: {token_dfs.get(token, 0)}, Collection Size: {collection_size})")
    
    if normalize_scores and tf_idf_scores:
        max_score = max(tf_idf_scores.values())
        if max_score > 0:
            tf_idf_scores = {token: score / max_score for token, score in tf_idf_scores.items()}
    return tf_idf_scores


# def rank_expansion_tokens_by_tf_idf_fast(query_vector_tokens, top_docs, token_dfs, collection_size, tokenizer, normalize_scores=False):
#     # Compute term frequencies for query tokens in the top documents
#     tf_scores = {token: 0 for token in query_vector_tokens}

#     top_doc_texts = [doc_text.lower() for _, _, doc_text in top_docs]
#     tokenized_docs = tokenizer(
#         top_doc_texts,
#         truncation=True,
#         add_special_tokens=False,
#         return_attention_mask=False,
#         return_token_type_ids=False
#     )['input_ids']

#     sum_doc_length = sum(len(doc_tokens) for doc_tokens in tokenized_docs)
#     for token in query_vector_tokens:
#         for idx, doc_tokens in enumerate(tokenized_docs):
#             tf_scores[token] += get_tf(doc_tokens, token)
    
#     # Rank tokens by their TF-IDF scores
#     tf_idf_scores = {token: tf / sum_doc_length * get_idf(token_dfs.get(token, 0), collection_size) for token, tf in tf_scores.items()}
#     for token, score in tf_idf_scores.items():
#         if np.isnan(score):
#             print(f"Warning: TF-IDF score for token '{token}' is NaN (TF: {tf_scores[token]}, DF: {token_dfs.get(token, 0)}, Collection Size: {collection_size})")
    
#     if normalize_scores and tf_idf_scores:
#         max_score = max(tf_idf_scores.values())
#         if max_score > 0:
#             tf_idf_scores = {token: score / max_score for token, score in tf_idf_scores.items()}
#     return tf_idf_scores


def rank_expansion_tokens_by_tf_idf_bm25(query_vector_tokens, top_docs, token_dfs, collection_size, tokenize, normalize_scores=False):
    tf_scores = {token: 0 for token in query_vector_tokens}
    top_doc_texts = [doc_text.lower() for doc_text in top_docs]
    tokenized_docs = [tokenize(doc_text) for doc_text in top_doc_texts]
    
    for token in query_vector_tokens:
        for idx, doc_tokens in enumerate(tokenized_docs):
            tf_scores[token] += get_tf(doc_tokens, token) * get_decay(idx)
    
    tf_idf_scores = {token: tf * get_idf(token_dfs.get(token, 0), collection_size) for token, tf in tf_scores.items()}
    
    if normalize_scores and tf_idf_scores:
        max_score = max(tf_idf_scores.values())
        if max_score > 0:
            tf_idf_scores = {token: score / max_score for token, score in tf_idf_scores.items()}
    return tf_idf_scores


# def rank_expansion_tokens_by_tf_idf_bm25(query_vector_tokens, top_docs, token_dfs, collection_size, tokenize, normalize_scores=False):
#     tf_scores = {token: 0 for token in query_vector_tokens}
#     top_doc_texts = [doc_text.lower() for doc_text in top_docs]
#     tokenized_docs = [tokenize(doc_text) for doc_text in top_doc_texts]
    
#     sum_doc_length = sum(len(doc_tokens) for doc_tokens in tokenized_docs)
#     for token in query_vector_tokens:
#         for idx, doc_tokens in enumerate(tokenized_docs):
#             tf_scores[token] += get_tf(doc_tokens, token)
    
#     tf_idf_scores = {token: tf / sum_doc_length * get_idf(token_dfs.get(token, 0), collection_size) for token, tf in tf_scores.items()}
    
#     if normalize_scores and tf_idf_scores:
#         max_score = max(tf_idf_scores.values())
#         if max_score > 0:
#             tf_idf_scores = {token: score / max_score for token, score in tf_idf_scores.items()}
#     return tf_idf_scores


# def dcg(scores, k=10):
#     """Compute DCG@k for a list of relevance scores."""
#     return sum((2**rel - 1) / math.log2(idx + 2) for idx, rel in enumerate(scores[:k]))


# def ndcg(system_results, relevance_judgments, k=10):
#     """
#     Compute NDCG@k for all queries.
    
#     system_results: {qid: {docid: score}}
#     relevance_judgments: {qid: {docid: relevance}}
#     """
#     ndcg_scores = {}

#     for qid, doc_scores in system_results.items():
#         if qid not in relevance_judgments:
#             continue  # skip queries without judgments

#         # Sort system results by score
#         ranked_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
#         # Get relevance scores for ranked docs
#         rels = [relevance_judgments[qid].get(docid, 0) for docid, _ in ranked_docs]

#         # Compute DCG@k
#         dcg_k = dcg(rels, k)

#         # Compute ideal DCG@k
#         ideal_rels = sorted(relevance_judgments[qid].values(), reverse=True)
#         idcg_k = dcg(ideal_rels, k)

#         # Normalize
#         ndcg_scores[qid] = dcg_k / idcg_k if idcg_k > 0 else 0.0

#     return ndcg_scores


def recall(system_results, relevance_judgments, k=10):
    """
    Compute Recall@k for all queries.
    
    system_results: {qid: {docid: score}}
    relevance_judgments: {qid: {docid: relevance}}
    """
    recall_scores = {}

    for qid, doc_scores in system_results.items():
        if qid not in relevance_judgments:
            continue  # skip queries without judgments

        # Sort system results by score
        ranked_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        top_k_docs = [docid for docid, _ in ranked_docs[:k]]

        # Relevant documents (with relevance > 0)
        relevant_docs = {docid for docid, rel in relevance_judgments[qid].items() if rel > 0}
        if not relevant_docs:
            recall_scores[qid] = 0.0
            continue

        # Retrieved relevant docs in top-k
        retrieved_relevant = sum(1 for d in top_k_docs if d in relevant_docs)

        recall_scores[qid] = retrieved_relevant / len(relevant_docs)

    return recall_scores