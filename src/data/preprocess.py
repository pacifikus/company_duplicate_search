import logging
import re
import string
from pathlib import Path

import cleanco
import geonamescache
import pandas as pd
import pycountry
import yaml
from unidecode import unidecode
import click
from tqdm import tqdm

tqdm.pandas()
base_stopwords = [*string.ascii_letters]
geo_names = geonamescache.GeonamesCache()


def get_legal_entities(legal_entities_path):
    with open(legal_entities_path, encoding="utf8") as f:
        legal_entities = f.read().split(',')
        legal_entities = [re.sub(f"[{string.punctuation}]+", ' ', entity) for entity in legal_entities]
        legal_entities = [unidecode(entity) for entity in legal_entities]
        legal_entities = [re.sub(' +', ' ', entity) for entity in legal_entities]
        return legal_entities


def get_usa_states():
    return [item['name'].lower() for item in geo_names.get_us_states().values()]


def get_cities():
    return [item['name'].lower() for item in geo_names.get_cities().values()]


def get_toponyms(chinese_data_path):
    countries = [country.name.lower() for country in list(pycountry.countries)]
    additional_world_parts = [
        'usa', 'us', 'americas', 'america', 'north', 'west', 'east', 'south', 'brasil', 'mexico',
        'eu', 'europe', 'area', 'city', 'asia'
    ]
    chinese_data = pd.read_csv(chinese_data_path)
    chinese_provincial_names = chinese_data['admin_name'].str.lower().unique().tolist()
    chinese_cities = chinese_data['city'].str.lower().unique()
    all_cities = list(set(get_cities()).union(set(chinese_cities.tolist())))
    usa_state = get_usa_states()
    toponyms = countries + additional_world_parts + chinese_provincial_names + all_cities + usa_state
    return toponyms


def get_stopwords(sw_path):
    with open(sw_path, encoding="utf8") as f:
        stopwords = f.read().split(',')
    return stopwords


def rm_company_suffix(text):
    return cleanco.basename(text)


def rm_toponyms(text, toponyms):
    text = ' '.join([item for item in text.split() if item not in toponyms])
    return text


def rm_stopwords(text, stopwords):
    return ' '.join([item for item in text.split() if item not in stopwords])


def resolve_data_path(config):
    legal_entities_path = Path(config['data']['external_data_path']) / Path(config['data']['legal_entities'])
    legal_entities = get_legal_entities(legal_entities_path)
    chinese_data_path = Path(config['data']['external_data_path']) / Path(config['data']['chinese_data'])
    toponyms = get_toponyms(chinese_data_path)
    sw_path = Path(config['data']['raw_data_path']) / Path(config['data']['stopwords'])
    stopwords = get_stopwords(sw_path)
    return legal_entities, toponyms, stopwords


def preprocess(text, config=None, legal_entities=None, stopwords=None, toponyms=None):
    if config:
        legal_entities, toponyms, stopwords = resolve_data_path(config)

    text = unidecode(text)
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]+", ' ', text)
    text = re.sub(r'(?<=\b\w)\s*(?=\w\b)', '', text)
    text = re.sub(f"[{string.digits}]+", ' ', text)
    text = re.sub(r"\(.*\)", '', text)
    text = re.sub(' +', ' ', text)
    text = rm_company_suffix(text)
    text = rm_toponyms(text, toponyms)
    text = ' '.join([item for item in text.split() if item not in base_stopwords])
    text = ' '.join([item for item in text.split() if item not in legal_entities])
    text = rm_stopwords(text, stopwords)
    return text


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
def start_preprocessing(config_path, dataset_path):
    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)

    logger = logging.getLogger(__name__)

    logger.info(f'Read {dataset_path}...')
    data = pd.read_csv(dataset_path)

    logger.info(f'Load stopwords...')
    legal_entities, toponyms, stopwords = resolve_data_path(config)

    data['init_name_1'] = data['name_1'].copy()
    data['init_name_2'] = data['name_2'].copy()
    logger.info(f'Start name_1 preprocessing...')
    data['name_1'] = data['name_1'].progress_apply(lambda x: preprocess(
        x,
        legal_entities=legal_entities,
        toponyms=toponyms,
        stopwords=stopwords
    ))
    logger.info(f'Start name_2 preprocessing...')
    data['name_2'] = data['name_1'].progress_apply(lambda x: preprocess(
        x,
        legal_entities=legal_entities,
        toponyms=toponyms,
        stopwords=stopwords
    ))

    logger.info(f'Save preprocessed file...')
    preprocessed_filepath = Path(config['data']['interim_data_path']) / Path(config['data']['preprocessed_df_name'])
    data.to_csv(preprocessed_filepath, index=False)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    start_preprocessing()
