import argparse
import json
from collections import Counter
from tqdm import tqdm
from transformers import AutoTokenizer
from concurrent.futures import ThreadPoolExecutor
    

def tokenize_batch(texts, tokenizer):
    tokenized = tokenizer(
        texts,
        truncation=True,
        add_special_tokens=False,
        return_attention_mask=False,
        return_token_type_ids=False
    )
    batch_counts = Counter()
    for token_ids in tokenized['input_ids']:
        unique_tokens = set(token_ids)
        batch_counts.update(unique_tokens)
    return batch_counts


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="naver/splade_v2_max")
    parser.add_argument('--original_collection', type=str, default="data/msmarco/corpus.jsonl")
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--batch_size', type=int, default=10000)
    parser.add_argument('--num_threads', type=int, default=16)
    return parser.parse_args()


def main():
    global tokenizer
    args = parse_args()

    with open(args.original_collection, 'r') as f:
        original_collection = [json.loads(line) for line in tqdm(f, desc="Loading collection")]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    batch_size = args.batch_size
    num_threads = args.num_threads
    num_docs = len(original_collection)
    batches = [
        [doc['text'].lower() for doc in original_collection[i:i+batch_size]]
        for i in range(0, num_docs, batch_size)
    ]

    doc_freq = Counter()
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(tokenize_batch, batch) for batch in batches]
        for future in tqdm(futures, desc="Merging Results"):
            doc_freq.update(future.result())

    doc_freq_str = {
        tokenizer.convert_ids_to_tokens([token_id])[0]: count
        for token_id, count in doc_freq.items()
    }

    with open(args.output_path, 'w') as f:
        json.dump(doc_freq_str, f, indent=4)


if __name__ == '__main__':
    main()