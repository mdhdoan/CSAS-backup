import json
import sys

from openpyxl import load_workbook

from sentence_transformers import SentenceTransformer

import torch


def load_data_file(file_name, sheet_name):
    wb = load_workbook(file_name)
    ws = wb[sheet_name]
    headers, inputs = [], []
    for i, row in enumerate(ws): 
        if i == 0:
            headers = [cell.value for cell in row]
            continue
        inputs.append({headers[j]: cell.value for j, cell in enumerate(row)})
    return inputs


def load_jsonl(file_name, slice=None, single=False):
    with open(file_name, 'rt') as in_file:
        documents = dict()
        lines = in_file.readlines()[0:int(slice)] if slice else in_file.readlines()
        for d in [json.loads(line.strip()) for line in lines]:
            documents[d['metadata']['link']] = d
        print(f"[{file_name}] Read {len(documents)} documents.")
        return documents


if __name__ == '__main__':
    input_data_file = sys.argv[1]
    input_sheet_name = sys.argv[2]
    jsonl_data_file = sys.argv[3]
    llm_model_name = sys.argv[4]
    
    model = SentenceTransformer("all-mpnet-base-v2")

    inputs = load_data_file(input_data_file, input_sheet_name)
    documents = load_jsonl(jsonl_data_file)

    tor_topic = [input['TOR topic'] for input in inputs][0]
    tor_question = [input['TOR questions'] for input in inputs][0]
    
    t_embeddings = model.encode([tor_topic], convert_to_tensor=True)
    q_embeddings = model.encode([tor_question], convert_to_tensor=True)
    
    for input in inputs:
        document = documents[input['Link']]
        print(f"\n{input['Link']}")

        problem = input['Problem']
        p_embeddings = model.encode([problem], convert_to_tensor=True)
        similarity_scores = model.similarity(p_embeddings, t_embeddings)[0]
        scores, indices = torch.topk(similarity_scores, k=1)
        print(f"\t[TOPIC PROBLEM SIM] -- {tor_topic}")
        for score, idx in zip(scores, indices):
            print(f"\t--- {score:.4f} --- {problem}")

        similarity_scores = model.similarity(p_embeddings, q_embeddings)[0]
        scores, indices = torch.topk(similarity_scores, k=1)
        print(f"\t[QUESTION PROBLEM SIM] -- {tor_question}")
        for score, idx in zip(scores, indices):
            print(f"\t--- {score:.4f} --- {problem}")

        risk = input['Risk']
        r_embeddings = model.encode([risk], convert_to_tensor=True)
        similarity_scores = model.similarity(r_embeddings, t_embeddings)[0]
        scores, indices = torch.topk(similarity_scores, k=1)
        print(f"\t[TOPIC RISK SIM] -- {tor_topic}")
        for score, idx in zip(scores, indices):
            print(f"\t--- {score:.4f} --- {problem}")

        similarity_scores = model.similarity(r_embeddings, q_embeddings)[0]
        scores, indices = torch.topk(similarity_scores, k=1)
        print(f"\t[QUESTION RISK SIM] -- {tor_question}")
        for score, idx in zip(scores, indices):
            print(f"\t--- {score:.4f} --- {problem}")
