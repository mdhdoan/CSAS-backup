import configparser
from datetime import datetime
import json
import os
from queue import Queue
import sys

import googlemaps

from sentence_transformers import SentenceTransformer

from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance
from bertopic import BERTopic

from langchain.chains.summarize import load_summarize_chain
from langchain.output_parsers.list import NumberedListOutputParser
from langchain.prompts.prompt import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.llms import Ollama


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
        for d in documents:
            if 'metadata' in d:
                d['metadata']['summary'] = d['metadata']['summary'].replace('Article body copy', '')
            if 'page_content' in d:
                d['page_content'] = d['page_content'].replace('Article body copy', '').strip()
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


class LLM_Tool():
    
    def __init__(self, config):
        output_parser = NumberedListOutputParser()
        format_instructions = output_parser.get_format_instructions()
        
        chunk_size = int(config['general']['chunk_size'])
        self.text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n"], chunk_size=chunk_size)

        self.tool_dict = dict()
        
        summarizer_model_name, chain_type = config['general']['llm_summarizer'].split(',')
        summarizer_llm = Ollama(model=summarizer_model_name, temperature=0.0)
        self.tool_dict['summarizer'] = load_summarize_chain(summarizer_llm, chain_type=chain_type)
        
        labeler_prompt = self.create_labeler_prompt(format_instructions)
        labeler_model_name = config['general']['llm_labeler']
        labeler_llm = Ollama(model=labeler_model_name, temperature=0.0)
        self.tool_dict['labeler'] = labeler_prompt | labeler_llm | output_parser

    def create_labeler_prompt(self, format_instructions):
        LABELING_PROMPT_TEMPLATE = """
        You are a helpful, respectful and honest assistant for labeling topics.

        I have a summary of a set of articles: 
        {summary}

        The articles share the following keywords delimited by triple backquotes (```):
        ```{keywords}```

        Create a concise label for this set of articles.
        {format_instructions}
        """
        labeling_prompt = PromptTemplate(
            input_variables=["summary", "keywords"],
            partial_variables={"format_instructions": format_instructions}, 
            template=LABELING_PROMPT_TEMPLATE)

        return labeling_prompt


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


class TopicModeler():

    def __init__(self, config,  model_option, llm_tool, geo_locator):
        self.model_section = config[model_option]

        self.languages = eval(self.model_section['languages'])
        self.file_path = self.model_section['file_path']
        self.min_cluster_sizes = eval(config['general']['min_cluster_sizes'])

        device = config['general']['device']
        hf_token = config['general']['HF_TOKEN'] if 'HF_TOKEN' in config['general'] else None 

        self.embedding_model_name = self.model_section['embedding_model']
        self.embedding_model_file_name = self.embedding_model_name.split('/')[-1]
        if hf_token:
            self.embedding_model = SentenceTransformer(self.embedding_model_name, device=device, token=hf_token)
        else:
            self.embedding_model = SentenceTransformer(self.embedding_model_name, device=device)

        self.embedding_prompt = None
        if 'embedding_prompt' in self.model_section and self.model_section['embedding_prompt']:
            self.embedding_prompt = f"{self.model_section['embedding_prompt']}: "

        self.geo_locator = geo_locator
        self.llm_tool = llm_tool

    def load_news_articles(self):
        documents = []
        for file_name in eval(self.model_section['article_file_list']):
            full_file_name = f"{self.file_path}/preprocessed/{file_name}"
            docs = load_jsonl(full_file_name)
            docs = [d for d in docs if d['metadata']['language'] in self.languages]
            documents.extend(docs)

        print(f"Loaded total {len(documents)} documents.")
        return documents, [d['metadata']['summary'] for d in documents]

    def create_clustering_tools(self, n_neighbors=15, n_components=5, min_cluster_size=15, ngrams=3, min_df=3):
        umap_model = UMAP(n_neighbors=n_neighbors, n_components=n_components, min_dist=0.0, metric='cosine', random_state=42)
        hdbscan_model = HDBSCAN(min_cluster_size=min_cluster_size, metric='euclidean', cluster_selection_method='eom', prediction_data=True)
        vectorizer_model = CountVectorizer(stop_words="english", ngram_range=(1, ngrams), min_df=min_df)
        ctfidf_model = ClassTfidfTransformer(bm25_weighting=True, reduce_frequent_words=True)
        representation_model = {
            "KBI": KeyBERTInspired(top_n_words=30),
            "MMR": MaximalMarginalRelevance(top_n_words=30, diversity=.5),
        }
        return BERTopic(
            embedding_model=self.embedding_model,            # Step 1 - Extract embeddings
            umap_model=umap_model,                      # Step 2 - Reduce dimensionality
            hdbscan_model=hdbscan_model,                # Step 3 - Cluster reduced embeddings
            vectorizer_model=vectorizer_model,          # Step 4 - Tokenize topics
            ctfidf_model=ctfidf_model,                  # Step 5 - Extract topic words
            representation_model=representation_model,  # Step 6 - (Optional) Fine-tune topic representations
            calculate_probabilities=True,
            top_n_words=30,
            verbose=True
        )

    def create_embeddings(self, texts):
        if self.embedding_prompt:
            embeddings = self.embedding_model.encode(texts, prompt=self.embedding_prompt, show_progress_bar=True)
        else:
            embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        return embeddings

    def create_visualizations(self, topic_model, titles, embeddings):
        doc_viz_file = f"{self.file_path}/viz/{self.embedding_model_file_name}-doc.html"
        hie_viz_file = f"{self.file_path}/viz/{self.embedding_model_file_name}-hie.html"

        if not os.path.isfile(doc_viz_file):
            topic_model.visualize_documents(titles, embeddings=embeddings)
            reduced_embeddings = UMAP(n_neighbors=10, n_components=2, min_dist=0.0, metric='cosine').fit_transform(embeddings)
            viz_doc = topic_model.visualize_documents(titles, reduced_embeddings=reduced_embeddings, width=2560, height=1440, custom_labels=True)
            viz_doc.write_html(doc_viz_file)

        if not os.path.isfile(hie_viz_file):
            viz_hie = topic_model.visualize_hierarchy(custom_labels=True)
            viz_hie.write_html(hie_viz_file)

    def create_topic_dict(self, topic_ids, topic_model, documents, texts):
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
                    'KBI': row_dict['KBI'],
                    'MMR': row_dict['MMR'],
                    'articles': [],
                    'id': '-'.join([f"{e}" for e in topic_ids + [topic_id]])
                }
                
            doc_link = documents[document_index]['metadata']['link']
            doc_title = documents[document_index]['metadata']['title']
            topic_dict[topic_id]['articles'].append([document_index, doc_title, doc_link, row_dict['Probability']])
            if int(topic_id) != -1 and row_dict['Representative_document']:
                topic_dict[topic_id]['representative_docs'].append([document_index, doc_link, row_dict['Probability'], row_dict['Document']])
            document_index += 1
        
        for topic_id in sorted(topic_dict):
            info = topic_dict[topic_id]
            if int(topic_id) == -1:
                info['summary'] = 'Outliers'
            else:
                reps = [d[3] for d in info['representative_docs']]
                docs = self.llm_tool.text_splitter.create_documents(reps)
                output = self.llm_tool.tool_dict['summarizer'].invoke({"input_documents": docs})
                info['summary'] = ''.join([e for e in output['output_text'].split('\n') if 'refined' not in e and 'summary' not in e])
                if info['summary']:
                    info['embedding'] = self.create_embeddings([info['summary']]).tolist()[0]
                len_sum_emb = len(info['embedding']) if 'embedding' in info else None
                print(f"[SUM] --- {topic_id} --- [{len(info['representative_docs'])}] --- {len_sum_emb} -- {info['summary']}")

                output = self.llm_tool.tool_dict['labeler'].invoke({'summary': info['summary'], 'keywords': list(set(info['keywords'] + info['KBI'] + info['MMR']))})
                if output:
                    info['label'] = output[0]
                    info['all_labels'] = output
                else:
                    info['label'] = 'No label'
                    info['all_labels'] = ['No label']
                print(f"[LBL] --- {topic_id} -- {info['label']} -- {info['all_labels']}")
        
        return topic_dict

    def cluster_documents(self, documents, texts, topic_ids=[], use_cache=True):
        topic_model = None
        cluster_file_name = f"{self.file_path}/temp/{self.embedding_model_file_name}-{self.model_section['topic_pkl_file']}"

        if not topic_ids and use_cache:
            if os.path.isfile(cluster_file_name):
                print(f"Loading {cluster_file_name} ...")
                topic_model = BERTopic.load(cluster_file_name)

        if not topic_model:
            embeddings = self.create_embeddings(texts)
            level = len(topic_ids)
            min_size = self.min_cluster_sizes[level] if 0 <= level < len(self.min_cluster_sizes) else self.min_cluster_sizes[-1]
            print(f"Clustering {topic_ids} ({min_size}) ...")
            topic_model = self.create_clustering_tools(min_cluster_size=min_size)
            topic_model.fit_transform(texts, embeddings)

            if not topic_ids and not use_cache:
                topic_model.save(cluster_file_name, serialization="pickle")
                print(f"[{cluster_file_name}] topic model saved.")

                titles = [d['metadata']['title'] for d in documents]
                self.create_visualizations(topic_model, titles, embeddings)
                print(f"[{cluster_file_name}] visualizations saved.")

                embedded_documents = []
                for document, embedding in zip(documents, embeddings.tolist()):
                    embedded_documents.append({
                        'link': document['metadata']['link'],
                        'embedding': embedding
                    })
                    
                emb_file_name = f"{self.file_path}/clusters/{self.embedding_model_file_name}-emb.jsonl"
                print(f"Creating {emb_file_name} ...")
                save_jsonl(embedded_documents, emb_file_name)

        print(topic_model.get_topic_info())

        # prefix_str = ''.join([f"-{e}" for e in topic_ids]) if topic_ids else ''
        # tpc_file_name = f"{self.file_path}/clusters/{self.embedding_model_file_name}{prefix_str}-cls.json"
        # print(f"Creating {tpc_file_name} ...")
        topic_dict = self.create_topic_dict(topic_ids, topic_model, documents, texts)
        # save_jsonl(topic_dict, tpc_file_name, single=True)
        
        return topic_dict

    def cluster_news_articles(self, use_cache):
        initial_topic_ids = []
        lower_limit_cluster = eval(config['general']['lower_limit_cluster'])
        queue = Queue()

        articles, summaries = self.load_news_articles()
        article_dict = dict()
        for article, summary in zip(articles, summaries):
            article_dict[article['metadata']['link']] = [article, summary]
        initial_topic_dict = self.cluster_documents(articles, summaries, topic_ids=initial_topic_ids, use_cache=use_cache)
        queue.put([initial_topic_dict, initial_topic_ids])
        
        while not queue.empty():
            item = queue.get()
            if not item:
                break
            
            topic_dict, topic_ids = item
            for topic_id in sorted(topic_dict.keys()):
                topic = topic_dict[topic_id]
                subtopic_ids = topic_ids + [topic_id]
                if len(topic['articles']) < lower_limit_cluster:
                    continue
                
                docs, sums = [], []
                for article in topic['articles']:
                    _, _, link, _ = article
                    doc, sum = article_dict[link]
                    docs.append(doc)
                    sums.append(sum)
                
                try:
                    subtopic_dict = self.cluster_documents(docs, sums, topic_ids=subtopic_ids, use_cache=use_cache)
                    topic['sub_topics'] = subtopic_dict
                    queue.put([subtopic_dict, subtopic_ids])
                except ValueError as ex:
                    pass
    
        tpc_file_name = f"{self.file_path}/clusters/{self.embedding_model_file_name}-all-cls.json"
        print(f"Creating {tpc_file_name} ...")
        save_jsonl(initial_topic_dict, tpc_file_name, single=True)

    def flatten_topic_tree(self):
        tpc_file_name = f"{self.file_path}/clusters/{self.embedding_model_file_name}-all-cls.json"
        topic_dict = load_jsonl(tpc_file_name, single=True)
        
        def add(t_dict, q):
            for t_id in sorted(t_dict.keys()):
                st_dict = t_dict[t_id]
                d = {
                    k: v if k != 'sub_topics' else [u['id'] for _, u in st_dict['sub_topics'].items()]
                    for k, v in st_dict.items() 
                }
                q.put(d)
                if 'sub_topics' in st_dict:
                    q = add(st_dict['sub_topics'], q)
            return q
        
        queue = Queue()
        queue = add(topic_dict, queue)
        topic_list = []
        while not queue.empty():
            item = queue.get()
            if not item:
                break
            topic_list.append(item)
    
        tpc_file_name = f"{self.file_path}/clusters/{self.embedding_model_file_name}-cls.jsonl"
        print(f"Creating {tpc_file_name} ...")
        save_jsonl(topic_list, tpc_file_name)


if __name__ == '__main__':
    start_time = datetime.now()

    config = load_config(sys.argv[1])
    option = sys.argv[2]

    geo_locator = None
    if 'locations' in config and 'GOOGLE_API_KEY' in config['locations']:
        geo_locator = GeoLocator(config['locations'])

    llm_tool = LLM_Tool(config)
    modeler = TopicModeler(config, option, llm_tool, geo_locator)
    modeler.cluster_news_articles(False)
    modeler.flatten_topic_tree()

    end_time = datetime.now()
    seconds = (end_time - start_time).total_seconds()
    print(f"Executed in {seconds} secs.", flush=True)

