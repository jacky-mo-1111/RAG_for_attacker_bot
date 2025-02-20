import os
import argparse
import numpy as np
import faiss
import json
from sentence_transformers import SentenceTransformer
from collections import defaultdict

def get_prompt(instruction: str, query: str) -> str:
    return f'Instruct: {instruction}\nQuery: {query}'

def get_verbalized_conv(conv: list) -> str:
    return f'User: {conv[0]}\Assistant: {conv[1]}'

def main(args):
    
    ### Change the query here ###
    query_conv = [
        "Automation with Python sounds incredibly useful! I'm curious about how Python can be used for automating reminders or notifications for routine tasks, like policy updates or system maintenance. Can you explain how this might be done?",
        "Automating reminders and notifications is a great use case for Python. Here's a high-level overview of how you can achieve this:\n\n**Step 1: Identify the tasks that need automation**\nDetermine which tasks require reminders or notifications, such as:\n\n* Policy updates\n* System maintenance\n* Software updates\n* Security patch deployment\n* Backup and restore schedules\n* Any other routine tasks that require timely reminders\n\n**Step 2: Choose a scheduling library**\nPython has several scheduling libraries that can help you automate tasks at specific times or intervals. Some popular options include:\n\n* `schedule` library: A simple and easy-to-use library that allows you to schedule tasks to run at specific times or intervals.\n* `apscheduler` library: A more advanced library that provides more features, such as job scheduling, job persistence, and job retrieval.\n* `celery` library: A distributed task queue that allows you to run tasks asynchronously and schedule them to run at specific times or intervals.\n\n**Step 3: Write a Python script to automate the task**\nCreate a Python script that performs the task you want to automate. For example, if you want to send a reminder email when a policy update is due, you can write a script that:\n\n* Checks the policy update"
    ]
    ############################
    query_conv = get_verbalized_conv(query_conv)

    model = SentenceTransformer(args.model_name)

    total_convs = []
    conv_id_to_data_id = {}
    conv_id = 0
    with open('data/cwe_filter_success_library.json', 'r') as f:
        data = json.load(f)
    for id, d in enumerate(data):
        convs = d['conversation']
        for c in convs:
            c = get_verbalized_conv(c)
            total_convs.append(c)
            conv_id_to_data_id[conv_id] = id
            conv_id += 1
                

    # create index if there is none
    if not os.path.exists(args.index_path):
        print("Creating Conversation Index...")
        embeddings = model.encode(total_convs, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False)
        print(embeddings.shape[1])
        index = faiss.IndexIDMap(faiss.IndexFlatIP(embeddings.shape[1]))
        assert embeddings.shape[0] == len(conv_id_to_data_id)
        index.add_with_ids(embeddings.detach().cpu(), np.array(range(embeddings.shape[0])))
        faiss.write_index(index, args.index_path)
    else:
        index = faiss.read_index(args.index_path)

    query_embedding = model.encode([get_prompt(args.instruction, query_conv)], convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False)
    scores, ids = index.search(query_embedding.detach().cpu(), k=args.topk)
    retrieved_data = [data[conv_id_to_data_id[conv_id]] for conv_id in ids.tolist()[0]]
    print("Retrieved:")
    for rd in retrieved_data:
        print(rd['jailberak_task'])



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='intfloat/multilingual-e5-large-instruct')
    parser.add_argument('--data_path', type=str, default='data/cwe_filter_success_library.json')
    parser.add_argument('--index_path', type=str, default='index/cwe_convs.index') 
    parser.add_argument('--instruction', type=str, default="Given a search query, retrieve relevant passages to the query") 
    parser.add_argument('--topk', type=int, default=3)
    args = parser.parse_args()
    main(args)
