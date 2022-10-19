from sentence_transformers import SentenceTransformer
import yaml
from pathlib import Path


config_path = 'params.yaml'

with open(config_path) as conf_file:
    config = yaml.safe_load(conf_file)

model_path = Path(config['model']['model_path']) / Path(config['model']['model_name'])
model = SentenceTransformer(model_path)
