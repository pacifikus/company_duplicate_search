import time
import pickle
import yaml
import logging
import click
from pathlib import Path

from text_dedup.postprocess import lsh_clustering
from text_dedup.postprocess import get_group_indices

def load_vectors(embeddings_filename):
    with open(embeddings_filename, mode='rb') as f:
        embedding_dict = pickle.load(f)

    vectors = list(embedding_dict.values())
    return vectors


@click.command()
@click.option(
    "-cf",
    "--config_path",
    type=click.Path(exists=True),
    help="Path to config file",
    required=True,
)
@click.option(
    "-np",
    "--num_perm",
    type=click.INT,
    help="Parameter num_perm for lsh_clustering, default 128",
    required=False,
)
def start_indexing(config_path, num_perm):
    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)
    logger = logging.getLogger(__name__)

    start_time = time.time()

    logger.info("Start clustering re-creating...")

    embeddings_filepath = Path(config['data']['processed_data_path']) / Path(config['data']['embeddings_filename'])
    vectors = load_vectors(embeddings_filepath)
    clusters = lsh_clustering(embeddings, seed=1, num_perm=num_perm)
    groups = get_group_indices(clusters)

    end_time = time.time()

    logger.info(f"All done. Took: {end_time - start_time} seconds")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    start_indexing()
