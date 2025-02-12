# RAG_for_attacker_bot
The RAG component of attacker bot jailbreaking LLM4Code
## Environment Setup
### Create and activate virtual environment
```
cd RAG_for_attacker_bot
python -m venv venv
source venv/bin/activate
```
### Install libraries
```
pip install faiss-cpu
pip install sentence-transformers
```
## Run
```
python retrieve.py
```
### Parameters
- `model_name`
- `data_path`
- `index_path`
- `instruction`: need instruction when encoding query
- `query`: search query
- `topk`