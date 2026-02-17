import argparse
import pickle
import json
from pydoc_data.topics import topics
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from utils.utils import SpladeRetriever, get_splade_vectors_for_queries, rank_expansion_tokens_by_tf_idf_fast, rank_expansion_tokens_by_tf_idf_bm25
from collections import defaultdict
import os
import pyterrier as pt
from utils.utils import tokenize_rm3_query, tokenize_and_stem
from functools import partial
from nltk.corpus import stopwords


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calculate wackiness scores for SPLADE expansion tokens."
    )
    # Core inputs
    parser.add_argument("--model-name", type=str,
                        default="./models/splade_max_modernbert/checkpoint/model",
                        help="HF model name or local path for AutoModelForMaskedLM and tokenizer.")
    parser.add_argument("--index-dir", type=str,
                        default=None,
                        help="Path to SPLADE index directory.")
    parser.add_argument("--original-collection-path", type=str,
                        default="./data/msmarco/corpus.jsonl",
                        help="Path to original MS MARCO collection JSONL.")
    parser.add_argument("--queries-path", type=str,
                        default="./data/msmarco/queries.jsonl",
                        help="Path to queries JSONL.")
    parser.add_argument("--qrels-train-path", type=str,
                        default="./data/msmarco/qrels/train.tsv",
                        help="Path to train qrels TSV.")
    parser.add_argument("--doc-ids-path", type=str, default=None,
                        help="Path to doc_ids.pkl. Defaults to <index_dir>/doc_ids.pkl.")
    parser.add_argument("--token-freqs-path", type=str,
                        default="./data/idfs/msmarco_splade_v2_idfs.json",
                        help="Path to JSON with token frequencies.")
    parser.add_argument("--token-scores-save-path", type=str,
                        default="./experiments/wackiness_scores/splade_v2_original_wackiness_scores.json",
                        help="Where to save the token scores JSON.")

    # Runtime parameters
    parser.add_argument("--sample-size", type=int, default=10_000, help="Number of train-qrel queries to sample.")
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size for query processing.")
    parser.add_argument("--save-step", type=int, default=100, help="Save intermediate results every N batches.")
    parser.add_argument("--top-k", type=int, default=100, help="Top-k documents to retrieve per query.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--threads", type=int, default=32, help="Number of threads for bm25 retrieval.")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cuda",
                        help="Computation device: auto, cpu, or cuda.")
    parser.add_argument("--query-source", type=str, choices=["queries", "documents"], default="queries",
                        help="Use 'queries' to sample train-qrel queries or 'documents' to sample documents as pseudo-queries.")
    parser.add_argument("--terrier-mem", type=int, default=32000, help="Memory (MB) for Terrier.")
    parser.add_argument("--fb-terms", type=int, default=10, help="Number of feedback terms for RM3.")
    parser.add_argument("--fb-docs", type=int, default=10, help="Number of feedback docs for RM3.")
    parser.add_argument("--normalize-scores", action="store_true", help="Whether to normalize wackiness scores to [0, 1].")
    

    return parser.parse_args()

def main():
    args = parse_args()

    np.random.seed(args.seed)
    
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    # Load collection
    with open(args.original_collection_path, "r") as f:
        original_collection = [json.loads(line) for line in tqdm(f, desc="Loading collection")]
    doc_text_map = {int(item["_id"]): item["text"] for item in original_collection}

    if args.query_source == "queries":
        # Load train qrels
        qrels_train = pd.read_csv(args.qrels_train_path, sep="\t", header=None, names=["query_id", "doc_id", "relevance"])

        # Load queries and filter to those present in train qrels, then sample
        with open(args.queries_path, "r") as f:
            queries = [json.loads(line) for line in tqdm(f, desc="Loading queries")]
        train_query_ids = set(qrels_train["query_id"].astype(str))
        filtered_queries = [q for q in queries if q["_id"] in train_query_ids]
        print(f"Total queries loaded: {len(queries)}; queries with train qrels: {len(filtered_queries)}")

        sampled_indices = np.random.choice(len(filtered_queries), size=args.sample_size, replace=False)
        print(f"Sampled {len(sampled_indices)} queries from {len(filtered_queries)} filtered queries.")
        sampled_items = [filtered_queries[i] for i in sampled_indices]
        query_texts = [item["text"] for item in sampled_items]
        query_ids = [item["_id"] for item in sampled_items]
    else:
        sampled_indices = np.random.choice(len(original_collection), size=args.sample_size, replace=False)
        sampled_items = [original_collection[i] for i in sampled_indices]
        query_texts = [item.get("text", "") for item in sampled_items]
        query_ids = [item.get("docid", str(i)) for i, item in zip(sampled_indices, sampled_items)]
        print(f"Sampled {len(sampled_items)} documents as pseudo-queries from collection of size {len(original_collection)}")
        
    # Token IDFs from frequencies
    token_frequencies = json.load(open(args.token_freqs_path, "r"))
    corpus_size = len(original_collection)
    
    # Pipeline for RM3
    if args.model_name == 'rm3':
        sampled_items = pd.DataFrame(sampled_items)
        sampled_items = sampled_items[['_id', 'text']]
        sampled_items.columns = ['qid', 'query']
        sampled_items['query'] = sampled_items['query'].astype(str).str.replace(r"[^\w\s]", " ", regex=True).str.strip()
        
        dataset = pt.get_dataset("msmarco_passage")
        if args.index_dir:
            index = pt.IndexFactory.of(args.index_dir)
        else:
            index = pt.IndexFactory.of(dataset.get_index("terrier_stemmed"))

        bm25 = pt.terrier.Retriever(index, wmodel="BM25", num_results=args.fb_docs, verbose=True)
        rm3 = pt.rewrite.RM3(index, fb_terms=args.fb_terms, fb_docs=args.fb_docs)
        bm25_final = pt.terrier.Retriever(
            index,
            wmodel="BM25",
            num_results=args.top_k,
            verbose=True,
            threads=args.threads
        )
        retriever = bm25 >> rm3 >> bm25_final
        
        results_df = retriever.transform(sampled_items)
        
        tokenizer = pt.java.autoclass("org.terrier.indexing.tokenisation.EnglishTokeniser")()
        stemmer = pt.java.autoclass("org.terrier.terms.PorterStemmer")()
        string_reader = pt.java.autoclass("java.io.StringReader")
        stop = set(stopwords.words("english"))

        tokenize = partial(tokenize_and_stem, tokenizer=tokenizer, stemmer=stemmer, string_reader=string_reader, stop=stop)
        wackiness_scores = defaultdict(list)
        for i, (qid, group) in enumerate(tqdm(results_df.groupby("qid"), total=results_df['qid'].nunique())):
            all_tokens = set(tokenize_rm3_query(group.iloc[0]['query']))
            query_tokens = set(tokenize(group.iloc[0]['query_0']))
            expansion_tokens = all_tokens - query_tokens
            doc_texts = [doc_text_map[int(doc_id)] for doc_id in group['docno']]
            
            token_scores = rank_expansion_tokens_by_tf_idf_bm25(expansion_tokens, doc_texts, token_frequencies, corpus_size, tokenize, normalize_scores=args.normalize_scores)
            for token, score in token_scores.items():
                wackiness_scores[token].append(score)
            
            if i > 0 and i % (args.save_step) == 0:
                json.dump(dict(wackiness_scores), open(args.token_scores_save_path, "w"), indent=4)
                print("Saved intermediate token scores.")
                
        json.dump(dict(wackiness_scores), open(args.token_scores_save_path, "w"), indent=4)
        print(f"Saved final token scores to {args.token_scores_save_path}")
            
    elif args.model_name == 'query2doc':
        sampled_items = pd.DataFrame(sampled_items)
        sampled_items = sampled_items[['_id', 'text']]
        sampled_items.columns = ['qid', 'query']
        
        # Load expanded queries and merge with sampled items
        expanded_queries = pd.read_json('./experiments/q2d_analysis/query2doc_msmarco/train.jsonl', lines=True)
        expanded_queries.columns = ['qid', 'query', 'pseudo_doc']
        expanded_queries['qid'] = expanded_queries['qid'].astype(np.int64)
        sampled_items['qid'] = sampled_items['qid'].astype(np.int64)
        sampled_items = sampled_items.merge(expanded_queries[['qid', 'pseudo_doc']], on='qid', how='left')
        sampled_items['pseudo_doc'] = sampled_items['pseudo_doc'].apply(lambda x: str(x).lower())
        sampled_items['query_0'] = sampled_items['query']
        sampled_items['query'] = (sampled_items['query'] + " ") * 5 + sampled_items['pseudo_doc']
        
        # Apply cleaning to the combined query + pseudo-doc text
        sampled_items['query_0'] = sampled_items['query_0'].astype(str).str.replace(r"[^\w\s]", " ", regex=True).str.strip()
        sampled_items['query'] = sampled_items['query'].astype(str).str.replace(r"[^\w\s]", " ", regex=True).str.strip()
        
        dataset = pt.get_dataset("msmarco_passage")
        if args.index_dir:
            index = pt.IndexFactory.of(args.index_dir)
        else:
            index = pt.IndexFactory.of(dataset.get_index("terrier_stemmed"))

        bm25 = pt.terrier.Retriever(index, wmodel="BM25", num_results=args.fb_docs, verbose=True, threads=args.threads)
        retriever = bm25
        
        results_df = retriever.transform(sampled_items)
        
        tokenizer = pt.java.autoclass("org.terrier.indexing.tokenisation.EnglishTokeniser")()
        stemmer = pt.java.autoclass("org.terrier.terms.PorterStemmer")()
        string_reader = pt.java.autoclass("java.io.StringReader")
        stop = set(stopwords.words("english"))
        
        tokenize = partial(tokenize_and_stem, tokenizer=tokenizer, stemmer=stemmer, string_reader=string_reader, stop=stop)
        wackiness_scores = defaultdict(list)
        for i, (qid, group) in enumerate(tqdm(results_df.groupby("qid"), total=results_df['qid'].nunique())):
            all_tokens = set(tokenize(group.iloc[0]['query']))
            query_tokens = set(tokenize(group.iloc[0]['query_0']))
            expansion_tokens = all_tokens - query_tokens
            doc_texts = [doc_text_map[int(doc_id)] for doc_id in group['docno']]
            
            token_scores = rank_expansion_tokens_by_tf_idf_bm25(expansion_tokens, doc_texts, token_frequencies, corpus_size, tokenize, normalize_scores=args.normalize_scores)
            for token, score in token_scores.items():
                wackiness_scores[token].append(score)
            
            if i > 0 and i % (args.save_step) == 0:
                json.dump(dict(wackiness_scores), open(args.token_scores_save_path, "w"), indent=4)
                print("Saved intermediate token scores.")
                
        json.dump(dict(wackiness_scores), open(args.token_scores_save_path, "w"), indent=4)
        print(f"Saved final token scores to {args.token_scores_save_path}")
        
    elif args.model_name == 'doct5query':
        if args.query_source != "documents":
            raise ValueError("DocT5Query wackiness score calculation only supports --query-source 'documents' since it relies on using documents as pseudo-queries to analyze expansion tokens. Please set --query-source to 'documents' when using --model-name 'doct5query'.")
        
        # Load expanded documents and merge with sampled items
        augmented_collection = pd.read_pickle("./experiments/doct5query/doct5query_passages.pkl")
        sampled_items = pd.DataFrame(sampled_items)
        sampled_items = sampled_items[['_id', 'text']]
        sampled_items['_id'] = sampled_items['_id'].astype(str)
        augmented_collection['_id'] = augmented_collection['_id'].astype(str)
        augmented_collection = augmented_collection[augmented_collection['_id'].isin(sampled_items['_id'])]
        augmented_collection.columns = ['qid', 'query', 'query_0']
        sampled_items = augmented_collection
        
        # Apply cleaning to the combined query + pseudo-doc text
        sampled_items['query_0'] = sampled_items['query_0'].astype(str).str.lower().replace(r"[^\w\s]", " ", regex=True).str.strip()
        sampled_items['query'] = sampled_items['query'].astype(str).str.lower().replace(r"[^\w\s]", " ", regex=True).str.strip()
        
        index = pt.IndexFactory.of(args.index_dir)
        
        bm25 = pt.terrier.Retriever(index, wmodel="BM25", num_results=args.fb_docs, verbose=True, threads=args.threads)
        retriever = bm25
        
        results_df = retriever.transform(sampled_items)
        
        tokenizer = pt.java.autoclass("org.terrier.indexing.tokenisation.EnglishTokeniser")()
        stemmer = pt.java.autoclass("org.terrier.terms.PorterStemmer")()
        string_reader = pt.java.autoclass("java.io.StringReader")
        stop = set(stopwords.words("english"))
        
        tokenize = partial(tokenize_and_stem, tokenizer=tokenizer, stemmer=stemmer, string_reader=string_reader, stop=stop)
        wackiness_scores = defaultdict(list)
        for i, (qid, group) in enumerate(tqdm(results_df.groupby("qid"), total=results_df['qid'].nunique())):
            all_tokens = set(tokenize(group.iloc[0]['query']))
            query_tokens = set(tokenize(group.iloc[0]['query_0']))
            expansion_tokens = all_tokens - query_tokens
            doc_texts = [doc_text_map[int(doc_id)] for doc_id in group['docno']]
            
            token_scores = rank_expansion_tokens_by_tf_idf_bm25(expansion_tokens, doc_texts, token_frequencies, corpus_size, tokenize, normalize_scores=args.normalize_scores)
            for token, score in token_scores.items():
                wackiness_scores[token].append(score)
            
            if i > 0 and i % (args.save_step) == 0:
                json.dump(dict(wackiness_scores), open(args.token_scores_save_path, "w"), indent=4)
                print("Saved intermediate token scores.")
                
        json.dump(dict(wackiness_scores), open(args.token_scores_save_path, "w"), indent=4)
        print(f"Saved final token scores to {args.token_scores_save_path}")
        
    else:
        doc_ids_path = args.doc_ids_path or os.path.join(args.index_dir, "doc_ids.pkl")

        # Load doc ID mapping
        doc_ids = pickle.load(open(doc_ids_path, "rb"))
        
        # Load model and tokenizer
        model = AutoModelForMaskedLM.from_pretrained(args.model_name).to(device)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)

        # Retriever
        retriever = SpladeRetriever(index_dir=args.index_dir, tokenizer=tokenizer, collection=None)

        def convert_retrieved_doc_ids_our_format(retrieved_doc_ids, retrieved_doc_scores):
            return [
                (int(doc_id), score, original_collection[doc_ids[int(doc_id)]]["text"])
                for doc_id, score in zip(retrieved_doc_ids, retrieved_doc_scores)
            ]

        wackiness_scores = defaultdict(list)
        for i in tqdm(range(0, len(query_texts), args.batch_size), desc="Processing batches"):
            batch_queries = query_texts[i:i + args.batch_size]
            tokenized_queries = tokenizer(
                batch_queries,
                truncation=True,
                add_special_tokens=False,
                return_attention_mask=False,
                return_token_type_ids=False
            )["input_ids"]

            query_vectors, _ = get_splade_vectors_for_queries(batch_queries, model, tokenizer)

            for qid, query_vector, tokenized_query in zip(query_ids[i:i + args.batch_size], query_vectors, tokenized_queries):
                retrieved_doc_ids, retrieved_doc_scores = retriever.retrieve_fast(query_vector, top_k=args.top_k)
                top_docs_our_format = convert_retrieved_doc_ids_our_format(retrieved_doc_ids, retrieved_doc_scores)
                ranked_tokens = rank_expansion_tokens_by_tf_idf_fast(query_vector, top_docs_our_format, token_frequencies, corpus_size, tokenizer, normalize_scores=args.normalize_scores)
                for (token, score) in ranked_tokens.items():
                    if token not in tokenized_query:
                        wackiness_scores[token].append(score)

            if i > 0 and i % (args.batch_size) == 0:
                json.dump(dict(wackiness_scores), open(args.token_scores_save_path, "w"), indent=4)
                print("Saved intermediate token scores.")

        json.dump(dict(wackiness_scores), open(args.token_scores_save_path, "w"), indent=4)
        print(f"Saved final token scores to {args.token_scores_save_path}")

if __name__ == "__main__":
    main()