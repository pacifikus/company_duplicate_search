import sys
import logging
import pickle
from pathlib import Path

import click
import pandas as pd
import yaml

from text_dedup.near_dedup import MinHashEmbedder

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.embedder import model

from src.data.preprocess import detect_language, translate_to_eng

def get_minhashembedding_raw(text, num_perm=128):
    embedder = MinHashEmbedder(num_perm)
    embeddings = embedder.embed(text)
    return embeddings

def get_minhashembedding_translated(text, logger, num_perm=128):
    embedder = MinHashEmbedder(num_perm)
    lang = detect_language(text)
    if lang != 'en':
        translated = translate_to_eng(text, logger)
        embeddings = embedder.embed(translated)
    else:
        embeddings = embedder.embed(text)
    return embeddings

def encode_texts(texts):
    return model.encode(texts)


def pack_embeddings(data, embeddings_name_1, embeddings_name_2):
    embeddings_all = {}
    for text, embedding in zip(data['name_1'].values.tolist(), embeddings_name_1):
        embeddings_all[text] = embedding
    for text, embedding in zip(data['name_2'].values.tolist(), embeddings_name_2):
        embeddings_all[text] = embedding
    return embeddings_all


def save_embeddings(embeddings, filepath):
    with open(filepath, mode='wb') as f:
        pickle.dump(embeddings, f)


@click.command()
@click.option(
    "-cf",
    "--config_path",
    type=click.Path(exists=True),
    help="Path to config file",
    required=True,
)
@click.option(
    "-df",
    "--dataset_path",
    type=click.Path(exists=True),
    help="Path to dataset",
    required=True,
)
def get_embeddings(config_path, dataset_path):
    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)
    logger = logging.getLogger(__name__)

    logger.info("Read processed data...")
    data = pd.read_csv(dataset_path)

    logger.info("Compute embeddings...")
    embeddings_name_1 = encode_texts(data['name_1'].values)
    embeddings_name_2 = encode_texts(data['name_2'].values)
    result = pack_embeddings(data, embeddings_name_1, embeddings_name_2)
    embedding_file_path = Path(config['data']['processed_data_path']) / Path(config['data']['embeddings_filename'])

    logger.info(f"Save embedings to {embedding_file_path}...")
    save_embeddings(result, embedding_file_path)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    get_embeddings()
