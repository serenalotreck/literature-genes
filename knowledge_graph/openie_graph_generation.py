"""
One-off script to get OpenIE triples and filter them by DyGIE++ entities.
"""
from os import listdir
from openie import StanfordOpenIE
from tqdm import tqdm
import networkx as nx

print('\nReading in data...')
txt_dir = '../data/ontogpt_input/destol_final/'
docs = []
for f in listdir(txt_dir):
    with open(txt_dir + f) as myf:
        docs.append(myf.read())

print('\nApplying OpenIE...')
triples = []
with StanfordOpenIE() as client:
    for doc in tqdm(docs):
        for triple in client.annotate(doc):
            triples.append(triple)

print('\nReading in DyGIE++ graph...')
dygie_co_graph = nx.read_graphml('../data/kg/all_drought_dt_co_occurrence_graph_02May2024.graphml')
dygiepp_ents = list(dygie_co_graph.nodes())

print('\nFiltering openIE results...')
openie_nodes = []
openie_edges = []
for trip in tqdm(triples):
    if (trip['subject'].lower() in dygiepp_ents) and (trip['object'].lower() in dygiepp_ents):
        openie_nodes.append(trip['subject'].lower())
        openie_nodes.append(trip['object'].lower())
        openie_edges.append((trip['subject'].lower(), trip['object'].lower()))
openie_nodes = list(set(openie_nodes))
openie_edges = list(set(openie_edges))

print(f'There are {len(openie_nodes)} nodes and {len(openie_edges)} edges after filtering by entity.')

print('\nBuilding graph...')
openie_graph = nx.Graph()
_ = openie_graph.add_nodes_from(openie_nodes)
_ = openie_graph.add_edges_from(openie_edges)

print('\nSaving...')
nx.write_graphml(openie_graph, '../data/kg/openIE_filtered_graph_08May2024.graphml')
