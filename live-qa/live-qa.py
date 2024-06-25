import configparser
from datetime import datetime
import json

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter


def load_config(file_name):
    config = configparser.ConfigParser()
    config.read_file(open(file_name))
    return config


class EmbeddingInput(BaseModel):
    text: str


class ModelManager:

    def __init__(self):
        config = load_config("live-qa.ini")
        config_section = config['general']

        device = "cpu" if config_section['device'] == "mps" else config_section['device']
        hf_token = config_section['HF_TOKEN'] if 'HF_TOKEN' in config_section else None 

        max_chunk_size = eval(config_section['chunk_size'])
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=max_chunk_size, chunk_overlap=0)

        embedding_model_name = config_section['embedding_model']
        print(f"[LOAD] {embedding_model_name} started.\n", flush=True)
        if hf_token:
            self.embedding_model = SentenceTransformer(embedding_model_name, device=device, token=hf_token)
        else:
            self.embedding_model = SentenceTransformer(embedding_model_name, device=device)

        self.embedding_prompt = None
        if 'embedding_prompt' in config_section and config_section['embedding_prompt']:
            self.embedding_prompt = f"{config_section['embedding_prompt']}: "
        print(f"[LOAD] {embedding_model_name} done.\n", flush=True)


    def create_embeddings(self, texts):
        if self.embedding_prompt:
            embeddings = self.embedding_model.encode(texts, prompt=self.embedding_prompt, show_progress_bar=True)
        else:
            embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        return embeddings


start_time = datetime.now()
model_manager = ModelManager()
app = FastAPI()
end_time = datetime.now()
seconds = (end_time - start_time).total_seconds()
print(f"Started in {seconds} seconds.\n", flush=True)


@app.post("/create_embeddings")
def create_embeddings(input: EmbeddingInput):
    start_time = datetime.now()
    result = {"embeddings": []}
    
    print(f"[TEXT] {input.text }\n", flush=True)
    if input.text and len(input.text) > 0:
        chunks = [d.page_content for d in model_manager.text_splitter.create_documents([input.text])]
        embeddings = model_manager.create_embeddings(chunks).tolist()
        result = {"embeddings": [[chunk, embedding] for chunk, embedding in zip(chunks, embeddings)]}
    
    end_time = datetime.now()
    seconds = (end_time - start_time).total_seconds()
    print(f"{seconds} seconds --- {len(result)}\n", flush=True)
    return result


@app.get("/")
def root():
    return "Hello!"



