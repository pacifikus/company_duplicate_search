import time
import pickle
from elastic_client import ElasticClient
import yaml
import logging
import click
from pathlib import Path


def load_vectors(embeddings_filename):
    with open(embeddings_filename, mode='rb') as f:
        vectors = pickle.load(f)

    vectors = [{'text': k, 'vector': v} for k, v in vectors.items()]
    for i, item in enumerate(vectors):
        item['id'] = i
    return vectors


@click.command()
@click.option(
    "-cf",
    "--config_path",
    type=click.Path(exists=True),
    help="Path to config file",
    required=True,
)
def start_indexing(config_path):
    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)
    logger = logging.getLogger(__name__)

    elastic_client = ElasticClient(
        host=config['indexing']['elastic']['host'],
        https=config['indexing']['elastic']['https'],
        config_path=config['indexing']['vector_config_path']
    )

    start_time = time.time()
    index_name = config['indexing']['elastic']['index_name']
    logger.info("Index re-creating...")
    elastic_client.delete_index(index_name)
    elastic_client.create_index(index_name)

    embeddings_filepath = Path(config['data']['processed_data_path']) / Path(config['data']['embeddings_filename'])
    vectors = load_vectors(embeddings_filepath)

    elastic_client.index_documents(index_name, vectors)
    end_time = time.time()
    logger.info(f"All done. Took: {end_time - start_time} seconds")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    start_indexing()
