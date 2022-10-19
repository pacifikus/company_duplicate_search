import json
import logging
import os

import elasticsearch.helpers
from dotenv import find_dotenv, load_dotenv
from elasticsearch import Elasticsearch

load_dotenv(find_dotenv())


class ElasticResponse:
    def __init__(self, resp):
        self.status_code = 400
        if 'acknowledged' in resp and resp['acknowledged']:
            print("request acknowledged!")
            self.status_code = 200
        else:
            self.status_code = resp['status']
            self.text = json.dumps(resp, indent=2)


class BulkResponse:
    def __init__(self, resp):
        self.status_code = 400
        if resp[0] > 0:
            self.status_code = 201


class SearchResponse:
    def __init__(self, resp):
        self.status_code = 400
        if 'hits' in resp:
            self.status_code = 200
        else:
            self.status_code = resp['status']
            self.text = json.dumps(resp, indent=2)


class ElasticClient:
    def __init__(self, host=None, https=False, config_path=''):
        self.host = host
        self.protocol = 'https' if https else 'http'
        self.config_path = config_path

        self.elastic = Elasticsearch(
            f'{self.protocol}://{self.host}:9200',
            verify_certs=False,
            http_auth=(
                os.environ.get("ELASTICSEARCH_LOGIN"),
                os.environ.get("ELASTICSEARCH_PASSWORD")
            )
        )
        self.logger = logging.getLogger(__name__)

    def resp_msg(self, msg, resp, throw=True):
        self.logger.info(f'{msg} [Status: {resp.status_code}]')
        if resp.status_code >= 400:
            if throw:
                raise RuntimeError(resp.text)

    def delete_index(self, index):
        resp = self.elastic.indices.delete(index=index, ignore=[400, 404], ignore_unavailable=True)
        self.resp_msg(f"Deleted index {index}", ElasticResponse(resp))

    def create_index(self, index_name):
        with open(self.config_path) as src:
            settings = json.load(src)
            resp = self.elastic.indices.create(index=index_name, body=settings)
            self.logger.info(f"Create_index: resp={resp}")
            self.resp_msg(f"Created index {index_name}", ElasticResponse(resp))

    def index_documents(self, index, docs_dict):

        def bulk_docs(docs):
            for doc in docs:
                if 'id' not in doc:
                    raise ValueError("Expecting docs to have field 'id' that uniquely identifies document")
                add_command = {
                    "_index": index,
                    "_id": doc['id'],
                    "_text": doc['text'],
                    "_source": doc
                }
                yield add_command

        resp = elasticsearch.helpers.bulk(self.elastic, bulk_docs(docs_dict), chunk_size=100)
        self.elastic.indices.refresh(index=index)
        self.resp_msg(msg=f"Streaming Bulk index DONE {index}", resp=BulkResponse(resp))

    def query(self, index, query):
        resp = self.elastic.search(index=index, body=query)
        self.resp_msg(msg="Searching {index} - {str(query)[:15]}", resp=SearchResponse(resp))

        matches = []
        for hit in resp['hits']['hits']:
            hit['_source']['_score'] = hit['_score']
            matches.append(hit['_source'])

        return matches, resp['took'], resp['hits']['total']['value']
