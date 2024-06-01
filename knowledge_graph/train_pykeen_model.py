"""
One-off script to train a specific PyKEEN model.

Author: Serena G. Lotreck
"""
import networkx as nx
from pykeen.triples import TriplesFactory
from pykeen.pipeline import pipeline


def get_predicate(row):
    if row.is_drought:
        if row.is_desiccation:
            return 'both'
        else:
            return 'drought'
    else:
        if row.is_desiccation:
            return 'desiccation'


if __name__ == '__main__':
    
    print('\nReading in and formatting data...')
    graph = nx.read_graphml('../data/kg/all_drought_dt_co_occurrence_graph_02May2024.graphml')
    edgelist = nx.to_pandas_edgelist(graph)
    edgelist['predicate'] = edgelist.apply(get_predicate, axis=1)
    triples = edgelist[['source', 'predicate', 'target']].to_numpy()
    print(f'Snapshot of triples: {triples[:5]}')
    tf = TriplesFactory.from_labeled_triples(triples, create_inverse_triples=True)
    training, validation, testing = tf.split([0.8, 0.1, 0.1])
    
    print('\nStarting model training....')
    result = pipeline(
    training=training,
    validation=validation,
    testing=testing,
    stopper='early',
    model='RESCAL',
    training_kwargs=dict(
        num_epochs=25,
        checkpoint_name='dt_rescal.pt',
        checkpoint_frequency=0
        )
    )
