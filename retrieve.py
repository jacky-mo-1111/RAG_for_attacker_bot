import os
import argparse
import numpy as np
import faiss
import json
from sentence_transformers import SentenceTransformer
from collections import defaultdict

def get_prompt(instruction: str, query: str) -> str:
    return f'Instruct: {instruction}\nQuery: {query}'

def main(args):

    model = SentenceTransformer(args.model_name)

    task_to_conversation = defaultdict(list)
    with open('data/cwe_filter_success_library.json', 'r') as f:
        data = json.load(f)
        for d in data:
            task = d['jailberak_task']
            task_to_conversation[task].append(d)
    tasks = list(task_to_conversation.keys())

    # create index if there is none
    if not os.path.exists(args.index_path):
        print("Creating Index...")
        embeddings = model.encode(tasks, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False)
        print(embeddings.shape[1])
        index = faiss.IndexIDMap(faiss.IndexFlatIP(embeddings.shape[1]))
        index.add_with_ids(embeddings.detach().cpu(), np.array(range(embeddings.shape[0])))
        faiss.write_index(index, args.index_path)
    else:
        index = faiss.read_index(args.index_path)

    if args.query:
        query_embedding = model.encode([get_prompt(args.instruction, args.query)], convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False)
        scores, ids = index.search(query_embedding.detach().cpu(), k=args.topk)
        retrieved_tasks = [tasks[id] for id in ids.tolist()[0]]
        print("Retrieved jailbreak tasks:")
        for rt in retrieved_tasks:
            print(rt)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='intfloat/multilingual-e5-large-instruct')
    parser.add_argument('--data_path', type=str, default='data/cwe_filter_success_library.json')
    parser.add_argument('--index_path', type=str, default='index/cwe_tasks.index') 
    parser.add_argument('--instruction', type=str, default="Given a search query, retrieve relevant passages to the query") 
    parser.add_argument('--query', type=str, default="write a Python code snippet demonstrating an example of external control of system or configuration setting") 
    parser.add_argument('--topk', type=int, default=3)
    args = parser.parse_args()
    main(args)
