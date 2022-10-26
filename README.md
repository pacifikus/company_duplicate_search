company_duplicate_search
==============================

## Project description

Company names similarity search service.
The service is based on [paraphrase-MiniLM-L6-v2](https://www.sbert.net/docs/pretrained_models.html) model
and [Elasticsearch](https://www.elastic.co/).

Project organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── css                <- Style files for Streamlit app
    ├── models             <- Serialized models
    ├── notebooks          <- Jupyter notebooks.
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.│
    ├── scripts            <- .sh scripts for the fast .py scripts running
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment`
    ├── src                <- Source code for use in this project.
    │   │
    │   ├── data           <- Scripts to preprocess data
    │   │   └── preprocess.py
    │   ├── embedder       <- Scripts to get embeddings from preprocessed data
    │   │   └── get_embeding.py
    │   ├── index          <- Scripts to create elasticsearch index from embeddings
    │   │   ├── elastic_client.py    
    │   │   ├── indexer_elastic.py
    │   │   └── vector_settings.json.py
    │   ├── search         <- Scripts to search by index with new company name
    │   │   ├── search_companies.py    
    │   │   └── streamlit_utils.py
    │   └── data          <- Data folder with raw and external data
    └── params.yaml       <- Config file

## Few requirements
- The most similar company names should be given to the user's request
- The service has fast response time
- Some parts of computation are moved to offline

### Offline part

- Computing of embeddings for all company names
- Creation of embeddings indexes

### Online part

- When a request comes from a user, we get its embedding via model and look for the nearest vectors in the embedding space.
- After that we rank the found company names and return result.

## Metrics

From a business point of view, we want to see in the results of the service the output that is as
relevant as possible to the query.
In terms of the initial sample, this means that we want to have the maximum **precision** of the search
results, but at the same time we want the model not to discard a large number of suitable options.

In addition, an unbalanced input dataset requires more flexible work with the choice of metric,
so our main quality metric has become **recall with fixed precision** that measures a **recall score** with
**a precision fixed at 0.8 value**.

## Calculate the load on the system. 

We will proceed from the following assumptions:
- **36000** queries per month (DAU - **300**, average **4** queries per user per day)

Server load: `36000 / (30 * 86400) = 0,013 RPS`

If each server response fits in **1MB**, then we generate traffic at **0,104 Mbps**

Initially, there are **30.000 vectors** (rough estimation, we don't store full-duplicated embeddings)
in the index of dimension 512 float64 => `30.000 * 384 * 8B = 92,16 MB` are needed for storage

To store **metadata** (company names): we need `30.000 * 1KB = 29,3MB`

Expected **embeddings growth**: 1.000 vectors per month = `1.000 * 384 * 8B = 3,072MB`

Expected **metadata growth**: `1.000 * 1KB = 0,97MB`

On the horizon of 1 year, we will need **165,92MB** of space

## Experiments setup

- Hardware
    - CPU count: 1
    - GPU count: 1
    - GPU type: Tesla T4
- Software:
    - Python version: 3.7.14
    - OS: Linux-5.10.133+-x86_64-with-Ubuntu-18.04-bionic

| Model                                 | Fixed Precision | Recall at Precision | 
|---------------------------------------|-----------------|---------------------|
| Tensorflow USE                        | 0.8             | 0.4469              | 
| paraphrase-MiniLM-L6-v2               | 0.8             | 0.5182              | 
| paraphrase-multilingual-mpnet-base-v2 | 0.8             | 0.4908              |


## How to run

First of all, you need to install all project requirements: 
```
pip install -r requirements.txt
```

For next stages you need Elasticsearch index to search by, so to install ElasticSearch locally in single node mode use:

```
docker run --name es01 -p 9200:9200 -p 9300:9300  -e "discovery.type=single-node" -t elasticsearch:8.4.3
```

After that set Elasticsearch login in password in .env file.

The next step is run one of the .sh scripts:
- to preprocess the data use `scripts/preprocess.sh`
- to create embeddings for preprocessed data use `scripts/create_embeddings.sh`
- to create elastic index use `scripts/create_index.sh`
- to streamlit app over index use `scripts/run_app.sh`

You can edit these scripts if you need.

Main project parameters are settled in [params.yaml](https://github.com/pacifikus/company_duplicate_search/blob/main/params.yaml) so you can edit this file too
