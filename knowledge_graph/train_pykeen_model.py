"""
Train a specific PyKEEN model.

Author: Serena G. Lotreck
"""
import argparse
from os.path import abspath
import networkx as nx
from pykeen.triples import TriplesFactory
from pykeen.pipeline import pipeline
from pykeen.constants import PYKEEN_CHECKPOINTS
from pykeen.losses import BCEWithLogitsLoss
import torch
import functools


def get_predicate(row):
    if row.is_drought:
        if row.is_desiccation:
            return 'both'
        else:
            return 'drought'
    else:
        if row.is_desiccation:
            return 'desiccation'

def main(graph_path, checkpoint_name, data_random_seed, model_random_seed, loss, dt, resume):

    print('\nReading in and formatting data...')
    graph = nx.read_graphml(graph_path)
    edgelist = nx.to_pandas_edgelist(graph)
    if dt:
        edgelist['predicate'] = edgelist.apply(get_predicate, axis=1)
    triples = edgelist[['source', 'predicate', 'target']].to_numpy()
    print(f'Snapshot of triples: {triples[:5]}')
    tf = TriplesFactory.from_labeled_triples(triples,
                                             create_inverse_triples=True)
    if not resume:
        training, validation, testing = tf.split([0.8, 0.1, 0.1],
                random_state=data_random_seed)
        print(f'Training splits performed with random seed {data_random_seed}.')
    else:
        checkpoint = torch.load(PYKEEN_CHECKPOINTS.joinpath(checkpoint_name))
        tf = TriplesFactory.from_labeled_triples(triples,
                                         create_inverse_triples=True,
                                         entity_to_id=checkpoint['entity_to_id_dict'],
                                         relation_to_id=checkpoint['relation_to_id_dict'])
        training, validation, testing = tf.split([0.8, 0.1, 0.1],
                random_state=data_random_seed)
        print(f'Training splits performed with random seed {data_random_seed}.')

    print('\nStarting model training....')
    if loss == 'default':
        result = pipeline(training=training,
                          validation=validation,
                          testing=testing,
                          stopper='early',
                          model='RESCAL',
                          training_kwargs=dict(
                              num_epochs=25,
                              checkpoint_name=checkpoint_name,
                              checkpoint_frequency=0),
                          random_seed=model_random_seed)
        print(f'Model intialized with random seed {model_random_seed}')
    elif loss == 'BCE':
        result = pipeline(training=training,
                          validation=validation,
                          testing=testing,
                          stopper='early',
                          model='RESCAL',
                          model_kwargs=dict(entity_initializer="normal",
                              relation_initializer="normal"),
                          loss=BCEWithLogitsLoss,
                          training_kwargs=dict(
                              num_epochs=25,
                              checkpoint_name=checkpoint_name,
                              checkpoint_frequency=0),
                          random_seed=model_random_seed)
        print(f'Model intialized with random seed {model_random_seed}')

    print('\nDone!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a PyKEEN model')

    parser.add_argument('graph_path', type=str,
            help='Path to .graphml file')
    parser.add_argument('checkpoint_name', type=str,
            help='Filename to model checkpoint, ends in .pt')
    parser.add_argument('-data_random_seed', type=int, default=1234,
            help='Random seed for dataset splits.')
    parser.add_argument('-model_random_seed', type=int, default=5678,
            help='Random seed for model initiation')
    parser.add_argument('-loss', type=str,
            help='The kind of loss to use. Options are "default" and "BCE".')
    parser.add_argument('--dt', action='store_true',
            help='Whether or not to assign desiccation/drought predicates')
    parser.add_argument('--resume', action='store_true',
            help='Whether or not this is resuming from a previous training session')

    args = parser.parse_args()

    args.graph_path = abspath(args.graph_path)

    main(args.graph_path, args.checkpoint_name, args.data_random_seed,
            args.model_random_seed, args.loss, args.dt, args.resume)
