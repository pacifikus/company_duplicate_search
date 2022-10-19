import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import yaml
from sklearn.decomposition import PCA

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.index.elastic_client import ElasticClient
from src.data.preprocess import preprocess
from src.embedder.get_embedding import encode_texts
from src.search.streamlit_utils import set_local_css, set_remote_css, set_icon

config_path = 'params.yaml'

with open(config_path) as conf_file:
    config = yaml.safe_load(conf_file)

elastic_client = ElasticClient(
    host=config['indexing']['elastic']['host'],
    https=config['indexing']['elastic']['https'],
    config_path=config['indexing']['vector_config_path'],
)


def get_query_embedding(input_query):
    input_query = preprocess(input_query, config)
    query_vector = encode_texts([input_query])
    query_vector = np.round(query_vector.astype(np.float64), 10)
    query_vector = query_vector.flatten().tolist()
    return query_vector


def get_query_config(query_vector, docs_count, distance_metric):
    if distance_metric == 'Cosine similarity':
        distance_metric = "1.0 + cosineSimilarity(params['query_vector'], 'vector')"
    elif distance_metric == 'l1norm':
        distance_metric = "1 / (1 + l1norm(params.query_vector, 'vector'))"
    elif distance_metric == 'l2norm':
        distance_metric = "1 / (1 + l2norm(params.query_vector, 'vector'))"
    else:
        distance_metric = \
            "double value = dotProduct(params.query_vector, 'vector'); return sigmoid(1, Math.E, -value);"
    es_query = {
        "size": docs_count,
        "_source": ["id", "text", "vector"],
        "query": {
            "script_score": {
                "query": {"match_all": {}},
                "script": {
                    "source": distance_metric,
                    "params": {"query_vector": query_vector}
                }
            }
        }
    }

    return es_query


@st.cache
def _filter_results(results, number_of_rows, number_of_columns) -> pd.DataFrame:
    return results.iloc[0:number_of_rows, 0:number_of_columns]


def create_pca(vectors):
    pca = PCA(2)
    vectorized_2dim = pca.fit_transform(vectors)
    pca_result = pd.DataFrame(vectorized_2dim, columns=['x', 'y'])
    return pca_result


def create_scatter_plot(data, labels, size, scores):
    fig = px.scatter(
        x=data["x"],
        y=data["y"],
        title='PCA visualization',
        color=scores,
        hover_data=[labels],
        size=size,
    )
    st.write(fig)


# index = config['indexing']['elastic']['index_name']
st.title(config['streamlit']['title'])
set_local_css(config['streamlit']['local_css_path'])
set_remote_css(config['streamlit']['remote_css_path'])
set_icon(config['streamlit']['icon'])

query = st.text_input("Type your query here", "Bridgestone company")
button_clicked = st.button("Go")
n_docs = st.sidebar.slider(label="Number of documents to view", **config['streamlit']['slider'])
index = st.sidebar.selectbox(
    'Target index',
    ('paraphrase-minilm-l6-v2', 'paraphrase-minilm-l6-v2_normalized')
)
distance = st.sidebar.radio(
    'Distance',
    ["Cosine similarity", "Dot product", "l1norm", 'l2norm'],
    index=0
)

if button_clicked and query != "":
    st.write(f"Index: {index}")
    input_query_text = query
    query_embedding = get_query_embedding(query)
    query = get_query_config(query_embedding, docs_count=n_docs, distance_metric=distance)
    with st.spinner(text="Searching..."):
        docs, query_time, num_found = elastic_client.query(index, query)

    st.success("Done!")
    st.write("Query time: {} ms".format(query_time))
    st.write("Found documents: {}".format(num_found))
    if num_found > 0:
        df = pd.DataFrame(docs, columns=["id", "text", "_score", "vector"])
        df['vector'] = df['vector'].apply(lambda x: np.array(x).reshape(1, -1))
        st.table(df.drop(['vector'], axis=1))
        chart_data = pd.DataFrame(
            df["_score"],
            columns=['score'])
        st.line_chart(chart_data)
        vectors = df['vector'].values.tolist()
        vectors.append(np.array(query_embedding).reshape(1, -1))
        data_to_pca = np.vstack(vectors)
        pca_vectors = create_pca(data_to_pca)
        labels = df['text'].values.tolist() + [f'{input_query_text} (QUERY)']
        create_scatter_plot(
            data=pca_vectors,
            labels=labels,
            size=[1 for i in range(len(labels))],
            scores=df['_score'].values.tolist() + [2]
        )
