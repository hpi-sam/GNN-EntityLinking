import os
import random
import requests
import numpy as np

from gensim.models import KeyedVectors
import logging

from gensim import utils
import nltk

import networkx as nx
import matplotlib.pyplot as plt


import shutil
import smart_open
from sys import platform
import random
import tensorflow as tf
import json


_is_relevant = [.0, 1.]
_is_not_relevant = [1., 0.]

_logger = logging.getLogger(__name__)


### Basic functions, for processing the data and building a graph

# get_words
# capitalize(word), low_case(word)
# infer_vector_from_word
# infer_vector_from_doc
# get_vectors_from_nodes_in_graph
# get_types_from_nodes_in_graph
# get_edge_name_with_signature
# get_node_name_with_signature
# add_triplets_to_graph_bw
# plot_graph
# get_chunks
# bin_data_into_buckets

from time import sleep
from SPARQLWrapper import SPARQLWrapper, JSON
import json

def get_words(text):
    '''Use: tokenised = get_words(text)
      Pre:
    text is a string
  Post:
    words is a list of the words in the text'''
    tokenizer = nltk.tokenize.TweetTokenizer()
    words = tokenizer.tokenize(utils.to_unicode(text))
    return words


def capitalize(word):
    return word[0].upper() + word[1:]


def low_case(word):
    return word[0].lower() + word[1:]


def infer_vector_from_word(model, word):
    '''Use: vector = infer_vector_from_word(model,word)
  Pre:
      model is models.keyedvectors from gensim (a mapping between keys and vectors)
      word is a key for model, such as a word from the vocabulary
  Post:
      vector is a gensim model vector representation, such as glove embedding, for word'''
    vector = np.zeros(300)
    try:
        vector = model[word]
    except:
        try:
            vector = model[capitalize(word)]
        except:
            try:
                vector = model[low_case(word)]
            except:
                pass
    return vector


def infer_vector_from_doc(model, text):
    '''Use: vector = infer_vector_from_word(model,text)
    Pre:
        model is models.keyedvectors from gensim (a mapping between keys and vectors)
        text is a key for model, in this case a document (string of words)
    Post:
        vector is a gensim model vector representation, such as glove embedding, for text'''
    words = get_words(text)
    vector = np.zeros(300)
    for word in words:
        vector += infer_vector_from_word(model, word)
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector /= norm
    return vector


def get_vectors_from_nodes_in_graph(g, model):
    '''Use: vectors = get_vectors_from_nodes_in_graph(graph, model)
    Pre:
        graph is a networkx graph
        model is models.keyedvectors from gensim (a mapping between keys and vectors)
    Post:
        vector is a numpy array of gensim model vector representations, such as glove embedding, for the text corresponding to each node in graph'''
    nodes = nx.nodes(g)
    vectors = []
    for node in nodes:
        text = node.replace('_', ' ')
        text = text.split('|')[0]
        vectors.append(infer_vector_from_doc(model, text))
    return np.array(vectors)


def get_types_from_nodes_in_graph(g):
    '''Use: vectors = get_types_from_nodes_in_graph(graph)
    Pre:
        graph is a networkx graph
    Post:
        vectors is a binary numpy array, with 1 if a node is a vertex and 0 if a node is an edge'''
    nodes = nx.nodes(g)
    vectors = []
    for node in nodes:
        texts = node.split('|')
        vector = np.zeros(3)
        if 'NODE' in texts:
            vector[0] = 1.
        if 'EDGE' in texts:
            vector[1] = 1.
        vectors.append(vector)
    return np.array(vectors)



### Functions for getting the graph ready

## "With the topology of the Wikidata graph, the information of each node is propagated onto the central item.
##   Ideally, after the graph convolutions, the vector at the position of the central item summarizes the information in the graph."

# get_bw_graph
# get_adjacency_matrices_and_vectors_given_triplets
# convert_text_into_vector_sequence
# get_item_mask_for_words
# infer_vector_from_vector_nodes

def get_bw_graph(triplets):
    '''Use: graph = get_bw_graph(triplets)
    Pre:
        triplets is a list of [node1,relation,node2]
    Post:
        graph is a networkx directed graph representation of the triplets'''
    g_bw = nx.DiGraph()
    add_triplets_to_graph_bw(g_bw, triplets)
    return g_bw


def convert_text_into_vector_sequence(model, text):
    '''Use: seq = convert_text_into_vector_sequence(model, text)
    Pre:
        model is a word2vec mapping
        text is a string
    Post:
        seq is a list of the embeddings of the words in text'''
    words = get_words(text)
    vectors = []
    for word in words:
        vectors.append(infer_vector_from_word(model, word))
    return vectors


def get_item_mask_for_words(text, item):
    '''Use: mask = get_item_mask_for_words(text, item)
    Pre:
        text is a string
        item is a string, name, possibly in text
    Post:
        mask is a binary list,  marks where in the text the item is (1 for the name, 0 everywhere else)
        "This mask acts as a "manually induced" attention of the item to disambiguate for"'''
    words = get_words(text)
    types = []
    words_in_item = get_words(item.lower())
    for word in words:
        types.append([1. if word.lower() in words_in_item else 0.] * 200)
    return types


def infer_vector_from_vector_nodes(vector_list):
    vector = np.zeros(300)
    for v in vector_list:
        vector += v
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector /= norm
    return vector



### Functions for processing the json data file and generating a graph with the wikidata information

# translate_from_url
# create_text_item_graph_dict
# get_json_data
# get_graph_from_wikidata_id
# get_adjacency_matrices_and_vectors_given_triplets


def translate_from_url(url):
    '''Use: item = translate_from_url(url)
    Pre:
        u is a string,  part of some wikidata url like the id
    Post:
        item is the data extracted from url (what comes after '/' and before '-')'''
    if '/' in url and '-' not in url:

        item = url.split('/')[-1]
#             print('from wikidata items_1: {}'.format(item))
    elif '/' in url and '-' in url:
        item = url.split('/')[-1].split('-')[0]
#             print('from wikidata items_2: {}'.format(item))
    else:
        item = url
#             print('from wikidata items_3: {}'.format(item))
    return item


def get_json_data(model, query, json_data, count):
    '''Use: data, * = get_json_data(ned_data)
    Pre:
        json_data is a json file that has been loaded, containing wikidata-disambig data
    Post:
        data is a list of dicts with the information from ned_data.
        Each piece of data contains the constructed graph for the correct and wrong ids
    '''
    data = []
    lost=[]
    i = 0
    for  json_item in json_data[:count]:

        sleep(2.5)
        print(json_item['string'])
        print(i/count)
        print('\n')
        i+=1
        try:
            text = json_item['text']
            item = json_item['string']

            wikidata_id = json_item['correct_id']

    #             print('before problem {}'.format(i))
            text_item_graph_dict = create_text_item_graph_dict(model, query, text, item, wikidata_id)
    #             print('before problem if not with graph {}'.format(i))
            text_item_graph_dict['answer'] = _is_relevant
    #             print('before problem if not with answer {}'.format(i))
            data.append(text_item_graph_dict)

            wikidata_id = json_item['wrong_id']
            text_item_graph_dict = create_text_item_graph_dict(model, query, text, item, wikidata_id)
            text_item_graph_dict['answer'] = _is_not_relevant


            data.append(text_item_graph_dict)
        except Exception as e:
            print(str(e))
            lost.append(json_item)
    return data, lost


def create_text_item_graph_dict(model, query, text, item, wikidata_id):
    '''Use: dict = create_text_item_graph_dict(model, text, item, wikidata_id)
    Pre:
        item is the name to disambiguate
        text is a text that contains the name in a certain context
        wikidata_id is a wikidata id, which can be either the correct or wrong id for the context in text
    Post:
      dict has the information about the graph in a dictionary form,
        including the constructed graph for the item and a vector representing the text
      '''

    text_item_graph_dict = {}
    text_item_graph_dict['text'] = text
    text_item_graph_dict['item'] = item
    text_item_graph_dict['wikidata_id'] = wikidata_id
    text_item_graph_dict['graph'] = get_graph_from_wikidata_id(model, query, wikidata_id, item)
    # text_item_graph_dict['item_vector'] = infer_vector_from_doc(_model, item)
    text_item_graph_dict['item_vector'] = infer_vector_from_vector_nodes(text_item_graph_dict['graph']['vectors'])
    text_item_graph_dict['question_vectors'] = convert_text_into_vector_sequence(model, text)
    text_item_graph_dict['question_mask'] = get_item_mask_for_words(text, item)
    return text_item_graph_dict


def get_graph_from_wikidata_id(model, query, wikidata_id, central_item):
    '''Use: graph = get_graph_from_wikidata_id(model, wikidata_id, central_item)
    Pre
        model is word2vec
        query is a query for wikidata that fetches a subgraph surrounding the wikidata_id, whose entitity has name "central_item"

    Post:
        A json with the data for the entity with the wikidata_id id was fetched.
        The (node1, edge, node2) triplet for the entities/relations connected to the entity was gathered.
        graph is a dict with:
            the adjacency matrix of the graph,
            the vector is glove embeddings for each node and its type (whether it is a vertex or edge).
    '''
    url = 'https://query.wikidata.org/bigdata/namespace/wdq/sparql?query=' + query % wikidata_id


    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
    # From https://www.wikidata.org/wiki/Wikidata:SPARQL_query_service/queries/examples#Cats
    sparql.setQuery(query % wikidata_id)
    sparql.setReturnFormat(JSON)
    data = sparql.query().convert()


    triplets = []
    for item in data['results']['bindings']:
        try:
            from_item = translate_from_url(wikidata_id)
            relation = translate_from_url(item['rel']['value'])
            to_item = translate_from_url(item['item']['value'])
            triplets.append((from_item, relation, to_item))
        except:
            pass
        try:
            from_item = translate_from_url(item['item']['value'])
            relation = translate_from_url(item['rel2']['value'])
            to_item = translate_from_url(item['to_item']['value'])
            triplets.append((from_item, relation, to_item))
        except:
            pass
    triplets = sorted(list(set(triplets)))


    if not triplets:
        raise RuntimeError("This graph contains no suitable triplets.")



    return get_adjacency_matrices_and_vectors_given_triplets(triplets, central_item, model)


def get_adjacency_matrices_and_vectors_given_triplets(triplets, central_item, model):
    '''Use: adj_vect_types = get_adjacency_matrices_and_vectors_given_triplets(triplets, central_item, model)
    Pre:
        triplets is a list of [node1,relation,node2]
        central_item is the name of the entity, which will become the central node in the graph represented by the triplets
        model is models.keyedvectors, such as word2vec: a mapping between keys (such as words) and a vector representation
    Post:
        adj_vect_types is a dict with:
            the adjacency matrix of the graph, the vector is glove embeddings for each node and its type (whether it is a vertex or edge)'''
    g_bw = get_bw_graph(triplets)

    vectors = get_vectors_from_nodes_in_graph(g_bw, model)
    node_types = get_types_from_nodes_in_graph(g_bw)
    nodelist = list(g_bw.nodes())

    print('nodelist, before adding the dummy: ')
    print(nodelist, '\n')
    #try:
    if central_item + '|NODE' not in nodelist:
        # we need to add a fake relation for the name
        # p1559 is "name in native language", which should do
        # should be second item, after the wikidata_id such as q534153
        # (e.g. ['q534153|NODE', 'captain marvel|NODE', 'p1559|EDGE', ... ])
        nodelist.insert(1,central_item + '|NODE')
        # then after it, add the dummy relation
        nodelist.insert(2,'p1559|EDGE')


    # the central item (the name) is in the list of triplets
    central_node_index = nodelist.index(central_item + '|NODE')
    nodelist[central_node_index], nodelist[0] = nodelist[0], nodelist[central_node_index]

    print('\n nodelist, after adding the dummy: ')
    print(nodelist)
    print('\n')
    #except Exception as e:
    #    print('nodelist:', e)
    #    raise e
    A_bw = np.array(nx.to_numpy_matrix(g_bw, nodelist=nodelist))
    return {'A_bw': A_bw,
            'vectors': vectors,
            'types': node_types}










def get_words(text):
    tokenizer = nltk.tokenize.TweetTokenizer()
    words = tokenizer.tokenize(utils.to_unicode(text))
    return words


def capitalize(word):
    return word[0].upper() + word[1:]


def low_case(word):
    return word[0].lower() + word[1:]


def infer_vector_from_word(model, word):
    vector = np.zeros(300)
    try:
        vector = model[word]
    except:
        try:
            vector = model[capitalize(word)]
        except:
            try:
                vector = model[low_case(word)]
            except:
                pass
    return vector


def infer_vector_from_doc(model, text):
    words = get_words(text)
    vector = np.zeros(300)
    for word in words:
        vector += infer_vector_from_word(model, word)
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector /= norm
    return vector


def get_vectors_from_nodes_in_graph(g, model):
    nodes = nx.nodes(g)
    vectors = []
    for node in nodes:
        text = node.replace('_', ' ')
        text = text.split('|')[0]
        vectors.append(infer_vector_from_doc(model, text))
    return np.array(vectors)


def get_types_from_nodes_in_graph(g):
    nodes = nx.nodes(g)
    vectors = []
    for node in nodes:
        texts = node.split('|')
        vector = np.zeros(3)
        if 'NODE' in texts:
            vector[0] = 1.
        if 'EDGE' in texts:
            vector[1] = 1.
        vectors.append(vector)
    return np.array(vectors)


def get_edge_name_with_signature(node_str):
    node_str = node_str.split('|')[0].lower()
    node_str += '|EDGE'
    return node_str


def get_node_name_with_signature(node_str):
    node_str = node_str.split('|')[0].lower()
    node_str += '|NODE'
    return node_str


def add_triplets_to_graph_bw(g, triplets):
    for n1, r, n2 in triplets:
        clean_n1 = get_node_name_with_signature(n1)
        clean_n2 = get_node_name_with_signature(n2)
        clean_r = get_edge_name_with_signature(r)
        g.add_node(clean_n1)
        g.add_node(clean_n2)
        g.add_node(clean_r)
        g.add_edge(clean_n2, clean_r, **{'label': 'to_relation'})
        g.add_edge(clean_r, clean_n1, **{'label': 'to_node'})
    return g
