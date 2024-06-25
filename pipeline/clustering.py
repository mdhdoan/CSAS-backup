import configparser
from datetime import datetime
import json
import os
import sys

import googlemaps

from sentence_transformers import SentenceTransformer, util

from transformers.pipelines import pipeline

from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.representation import KeyBERTInspired
from bertopic import BERTopic

from langchain_community.embeddings import OllamaEmbeddings


def load_config(file_name):
    config = configparser.ConfigParser()
    config.read_file(open(file_name))
    return config


def load_jsonl(file_name, slice=None, single=False):
    if not os.path.isfile(file_name):
        return None
    
    with open(file_name, 'rt') as in_file:
        lines = in_file.readlines()[0:int(slice)] if slice else in_file.readlines()
        documents = [json.loads(line.strip()) for line in lines]
        print(f"[{file_name}] Read {len(documents)} documents.")
        if single and len(documents) == 1:
            return documents[0]
        return documents


def save_jsonl(documents, file_name, single=False):
    with open(file_name, 'wt') as out_file:
        if single:
            out_file.write(f"{json.dumps(documents)}\n")
            print(f"[{file_name}] - Wrote 1 document.")
        else:
            for document in documents:
                out_file.write(f"{json.dumps(document)}\n")
            print(f"[{file_name}] - Wrote {len(documents)} documents.")


class GeoLocator():
    
    def __init__(self, config_section):
        self.config_section = config_section
        resolved_locations = load_jsonl(self.config_section['resolved_locations'], single=True)
        self.location_dict = resolved_locations if resolved_locations else dict()
        self.geo_coder = googlemaps.Client(key=self.config_section['GOOGLE_API_KEY'])
        self.call_counter = 0
        
    def geocode(self, text):
        if text in self.location_dict:
            print(f"[GEOCODE] [CACHE] --- [{self.call_counter}] --- {text}", flush=True)
            return self.location_dict[text]
        
        self.call_counter += 1
        r = self.geo_coder.geocode(text, language='en')
        if r and len(r) > 0:
            self.location_dict[text] = {
                'name': r[0]['formatted_address'],
                'lat': r[0]['geometry']['location']['lat'],
                'lng': r[0]['geometry']['location']['lng'],
                'place_id': r[0]['place_id'],
                'types': r[0]['types'],
            }
            print(f"[GEOCODE] [FOUND] --- [{self.call_counter}] --- {text}", flush=True)
            save_jsonl(self.location_dict, self.config_section['resolved_locations'], single=True)
            return self.location_dict[text]
        
        print(f"[GEOCODE] [ERROR] --- [{self.call_counter}] --- {text}", flush=True)
        return None


def load_news_articles(config_section):
    documents = []
    languages = eval(config_section['languages'])
    file_path = config_section['file_path']
    
    for file_name in eval(config_section['article_file_list']):
        full_file_name = f"{file_path}/preprocessed/{file_name}"
        docs = load_jsonl(full_file_name)
        docs = [d for d in docs if d['metadata']['language'] in languages]
        documents.extend(docs)
    print(f"Loaded total {len(documents)} documents.")

    texts = [d['metadata']['summary'] for d in documents]
    return documents, texts


def create_clustering_tools(embedding_model, min_cluster_size):
    umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42)
    hdbscan_model = HDBSCAN(min_cluster_size=min_cluster_size, metric='euclidean', cluster_selection_method='eom', prediction_data=True)
    vectorizer_model = CountVectorizer(stop_words="english", ngram_range=(1, 3), min_df=3)
    ctfidf_model = ClassTfidfTransformer(bm25_weighting=True, reduce_frequent_words=True)
    representation_model = KeyBERTInspired()

    return BERTopic(
        embedding_model=embedding_model,            # Step 1 - Extract embeddings
        umap_model=umap_model,                      # Step 2 - Reduce dimensionality
        hdbscan_model=hdbscan_model,                # Step 3 - Cluster reduced embeddings
        vectorizer_model=vectorizer_model,          # Step 4 - Tokenize topics
        ctfidf_model=ctfidf_model,                  # Step 5 - Extract topic words
        representation_model=representation_model,  # Step 6 - (Optional) Fine-tune topic representations
        top_n_words=30,
        calculate_probabilities=True,
        verbose=True
    )
    

def cluster_news_articles(documents, texts, config_section, device, hf_token, partial_id=None, min_cluster_size=15):
    partial_text = f"{partial_id}-" if partial_id is not None else ""
   
    embedding_model_name = config_section['embedding_model']
    model_part_name = embedding_model_name.split('/')[-1]

    cluster_file_name = f"{config_section['file_path']}/temp/{model_part_name}-{partial_text}{config_section['topic_pkl_file']}"
    if os.path.isfile(cluster_file_name):
        print(f"Loading {cluster_file_name} ...")
        topic_model = BERTopic.load(cluster_file_name)

    else:
        print(f"Creating {cluster_file_name} ...")
        if hf_token:
            embedding_model = SentenceTransformer(embedding_model_name, device=device, token=hf_token)
        else:
            embedding_model = SentenceTransformer(embedding_model_name, device=device)
            
        if 'embedding_prompt' in config_section and config_section['embedding_prompt']:
            embedding_prompt = f"{config_section['embedding_prompt']}: "
            embeddings = embedding_model.encode(texts, prompt=embedding_prompt, show_progress_bar=True)
        else:
            embeddings = embedding_model.encode(texts, show_progress_bar=True)
            
        topic_model = create_clustering_tools(embedding_model, min_cluster_size=min_cluster_size)
        topic_model.fit_transform(texts, embeddings)
        
        if not partial_text:
            topic_model.save(cluster_file_name, serialization="pickle")
            print(f"[{cluster_file_name}] saved.")
    
        doc_viz_file = f"{config_section['file_path']}/viz/{model_part_name}-{partial_text}doc.html"
        hie_viz_file = f"{config_section['file_path']}/viz/{model_part_name}-{partial_text}hie.html"

        titles = [d['metadata']['title'] for d in documents]
        if not os.path.isfile(doc_viz_file):
            topic_model.visualize_documents(titles, embeddings=embeddings)

            # Reduce dimensionality of embeddings, this step is optional but much faster to perform iteratively:
            reduced_embeddings = UMAP(n_neighbors=10, n_components=2, min_dist=0.0, metric='cosine').fit_transform(embeddings)
            viz_doc = topic_model.visualize_documents(titles, reduced_embeddings=reduced_embeddings, width=2560, height=1440, custom_labels=True)
            viz_doc.write_html(doc_viz_file)

        if not os.path.isfile(hie_viz_file):
            viz_hie = topic_model.visualize_hierarchy(custom_labels=True)
            viz_hie.write_html(hie_viz_file)
    
    print(topic_model.get_topic_info())
    
    tpc_file_name = f"{config_section['file_path']}/clusters/{model_part_name}-{partial_text}cls.json"
    if os.path.isfile(tpc_file_name):
        print(f"Loading {tpc_file_name} ...")
        topic_dict = load_jsonl(tpc_file_name, single=True)

    else:
        print(f"Creating {tpc_file_name} ...")
        document_info = topic_model.get_document_info(texts)
        headers, rows = document_info.columns.tolist(), document_info.values.tolist()
        
        topic_dict, document_index = dict(), 0
        for row in rows:
            row_dict = {header: value for header, value in zip(headers, row)}
            topic_id = row_dict['Topic']
            if topic_id not in topic_dict:
                topic_dict[topic_id] = {
                    'name': row_dict['Name'],
                    'representative_docs': [],
                    'keywords': row_dict['Representation'],
                    'articles': [],
                    # 'link_list': []
                }
            doc_link = documents[document_index]['metadata']['link']
            doc_title = documents[document_index]['metadata']['title']
            topic_dict[topic_id]['articles'].append([document_index, doc_title, doc_link, row_dict['Probability']])
            # topic_dict[topic_id]['link_list'].append(doc_link)
            if int(topic_id) != -1 and row_dict['Representative_document']:
                topic_dict[topic_id]['representative_docs'].append([document_index, doc_link, row_dict['Probability'], row_dict['Document']])
            document_index += 1
        
        save_jsonl(topic_dict, tpc_file_name, single=True)
    
    return topic_dict
    

if __name__ == '__main__':
    start_time = datetime.now()

    config = load_config(sys.argv[1])
    option = sys.argv[2]
    
    device = config['general']['device']
    hf_token = config['general']['HF_TOKEN'] if 'HF_TOKEN' in config['general'] else None 
    
    geo_locator = None
    if 'locations' in config and 'GOOGLE_API_KEY' in config['locations']:
        geo_locator = GeoLocator(config['locations'])
        # print(geo_locator.geocode('800 Benvenuto Ave, Brentwood Bay, BC V8M 1J8'))
        # print(geo_locator.geocode('Av. Gustave Eiffel, 75007 Paris, France'))
        # print(geo_locator.geocode('Piazza del Duomo, 56126 Pisa PI, Italy'))
    
    articles, summaries = load_news_articles(config[option])
    topic_dict = cluster_news_articles(articles, summaries, config[option], device, hf_token)
    
    min_size = eval(config['general']['super_cluster_min_size'])
    for topic_id, topic in topic_dict.items():
        if len(topic['articles']) < min_size:
            continue
        docs, txts = [], []
        for article in topic['articles']:
            document_index, _, _, _ = article
            docs.append(articles[document_index])
            txts.append(summaries[document_index])
        try:
            cluster_news_articles(docs, txts, config[option], device, hf_token, partial_id=topic_id, min_cluster_size=5)
        except ValueError as ex:
            pass

    end_time = datetime.now()
    seconds = (end_time - start_time).total_seconds()
    print(f"Executed in {seconds} secs.", flush=True)

