from collections import defaultdict
from datetime import datetime
import json
import sys

from newspaper import Article, Config
from openpyxl import load_workbook


def load_data_file(file_name, sheet_names):
    if ';' in sheet_names:
        sheet_names = sheet_names.split(';')
    else:
        sheet_names = [sheet_names]

    wb = load_workbook(file_name)
    input_dict = defaultdict(list)
    for sheet_name in sheet_names:
        ws = wb[sheet_name]
        headers, inputs = [], input_dict[sheet_name]
        for i, row in enumerate(ws): 
            if i == 0:
                headers = [cell.value for cell in row]
                continue
            inputs.append({headers[j]: cell.value for j, cell in enumerate(row)})
        print(f"[{sheet_name}] --- {len(inputs)}")
    return input_dict


def load_article(input_dict, url_file_path, timeout = 30, user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'):
    config = Config()
    config.request_timeout = timeout
    config.browser_user_agent = user_agent

    for sheet_name, rows in input_dict.items():
        url_file_name = f"{url_file_path}/{sheet_name}.jsonl"
        with open(url_file_name, 'a+t') as url_out_file:
            for url in [row['URL'] for row in rows]:
                print(f"{url} ... ")
                article = Article(url, config=config)
                try:
                    article.download()
                    article.parse()
                    article.nlp()
                    publish_date = getattr(article, "publish_date", "")
                    if isinstance(publish_date, datetime):
                        publish_date = publish_date.strftime('%Y-%m-%dT%H-%M-%S')
                    else:
                        publish_date = str(publish_date)
                        
                    metadata = {
                        "title": getattr(article, "title", ""),
                        "link": getattr(article, "url", getattr(article, "canonical_link", "")),
                        "authors": getattr(article, "authors", []),
                        "language": getattr(article, "meta_lang", ""),
                        "description": getattr(article, "meta_description", ""),
                        "publish_date": publish_date,
                        "keywords": getattr(article, "keywords", []),
                        "summary": getattr(article, "summary", ""),
                    }
                    document = {
                        "metadata": metadata,
                        "page_content": article.text
                    }
                    url_out_file.write(f"{json.dumps(document)}\n")
                    url_out_file.flush()
                    print(f"{url} --- {metadata['title']} --- [{len(document['page_content'])}]")
                except Exception as e:
                    print(f"ERR [{e}]", flush=True)
    

if __name__ == '__main__':
    input_data_file = sys.argv[1]
    input_sheet_names = sys.argv[2]
    out_file_path = sys.argv[3]

    input_dict = load_data_file(input_data_file, input_sheet_names)
    load_article(input_dict, out_file_path)
