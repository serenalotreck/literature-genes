"""
One-off script to contract the graph based on entity groundings.

Author: Serena G. Lotreck
"""
import networkx as nx
from collections import defaultdict
import taxoniq
from tqdm.notebook import tqdm
import json
import pandas as pd


def contract_groups(graph, groups):
    """
    Contract groups of nodes that may or may not be connected. Combines the number
    of per-doc mentions for edges and nodes, and keeps the oldest year as the first_year_mentioned.

    graph, newtorkx Graph: undirected network containing nodes in group
    groups, dict of list: nodes to coalesce.
    """
    # Convert original graph to node and edgelist
    nodes = graph.nodes(data=True)
    edges = nx.to_pandas_edgelist(graph)

    # Make sure edge years are integers for later use
    edges = edges.astype({'first_year_mentioned': 'int32'})

    # Go through the groups
    nodes_to_add = []
    nodes_to_remove = []
    for grounding, n_list in tqdm(groups.items()):

        try:
            _ = [nodes[n] for n in n_list]
        except KeyError as e:
            print(f'Skipping grounding group {grounding} because of node '
                    f'{e} not being found in graph.')
            continue
        # Get the oldest year for node mentions
        oldest_node_year = min(
            [int(nodes[n]['first_year_mentioned']) for n in n_list])

        # Get total node mentions
        total_node_mentions = sum(
            [nodes[n]['num_doc_mentions_all_time'] for n in n_list])

        # Get all uids of origin
        combined_node_uids = ', '.join(
            [nodes[n]['uids_of_origin'] for n in n_list])

        # Get the formal name that we want to keep for the node
        try:
            formal_name = taxoniq.Taxon(int(grounding)).scientific_name
        except KeyError:
            formal_name = groups[grounding][0]

        # Get the subset of relations that involve these nodes
        edge_subset = edges[(edges['source'].isin(n_list)) |
                            (edges['target'].isin(n_list))].reset_index(
                                drop=True)

        # Replace all nodes with the formal representation to being coalescing
        edges_replaced = edge_subset.replace(to_replace=n_list,
                                             value=formal_name)

        # Combine the values of any edges that are semantically identical after replacement
        # First, get the indices of repeated groups, order-agnostically
        tup_list = [
            tuple(set(tup)) for tup in list(edges_replaced[
                ['source', 'target']].itertuples(index=False, name=None))
            if len(set(tup)) > 1
        ]
        tup_set = set(tup_list)
        rep_idxs = defaultdict(list)
        for i, tup in enumerate(tup_list):
            rep_idxs[tup].append(i)
        # Now, combine the attributes and store in a dict
        edge_replacements = []
        keep_the_same = []
        for edge, idxs in rep_idxs.items():
            if len(idxs) > 1:
                oldest_edge_year = edges_replaced.loc[
                    idxs, 'first_year_mentioned'].min()
                total_edge_mentions = edges_replaced.loc[
                    idxs, 'num_doc_mentions_all_time'].sum()
                is_drought = edges_replaced.loc[idxs, 'is_drought'].any()
                is_desiccation = edges_replaced.loc[idxs,
                                                    'is_desiccation'].any()
                uids_of_origin = ', '.join(
                    edges_replaced.loc[idxs, 'uids_of_origin'])
                edge_replacements.append({
                    'source': edge[0],
                    'target': edge[1],
                    'first_year_mentioned': oldest_edge_year,
                    'num_doc_mentions_all_time': total_edge_mentions,
                    'is_drought': is_drought,
                    'is_desiccation': is_desiccation,
                    'uids_of_origin': uids_of_origin
                })
            elif len(idxs) == 1:
                keep_the_same.extend(idxs)
        # Now drop all indices that had more than one semantic replicate
        edges_replaced_to_drop = edge_subset.loc[~edges_replaced.index.
                                                 isin(keep_the_same)]
        edges = pd.merge(edges,
                         edges_replaced_to_drop,
                         how='outer',
                         indicator=True)
        edges = edges.loc[edges._merge == 'left_only'].drop(columns='_merge')
        # And replace with the combined edges
        edges = pd.concat([edges, pd.DataFrame(edge_replacements)],
                          ignore_index=True)

        # And finally, save the formal name of the new node and its attrs to use later, and add the nodes to remove
        nodes_to_add.append((formal_name, {
            'first_year_mentioned': oldest_node_year,
            'num_doc_mentions_all_time': total_node_mentions,
            'uids_of_origin': combined_node_uids,
            'entity_type': 'Multicellular_organism'
        }))  # Since this was all we could ground
        nodes_to_remove.extend(n_list)

    # Remove old nodes and add new ones
    nodes_processed = [(n, attrs) for n, attrs in nodes
                       if n not in nodes_to_remove]
    for new_node in nodes_to_add:
        nodes_processed.append(new_node)

    # Make new graph from edgelist and nodelist
    new_graph = nx.from_pandas_edgelist(edges,
                                        edge_attr=[
                                            'first_year_mentioned',
                                            'num_doc_mentions_all_time',
                                            'is_drought', 'is_desiccation',
                                            'uids_of_origin'
                                        ])
    _ = new_graph.add_nodes_from(nodes_processed)

    return new_graph


if __name__ == "__main__":

    print('\nReading in the graph...')
    graph = nx.read_graphml(
        '../data/kg/all_drought_dt_co_occurrence_graph_02May2024.graphml')

    print('\nReading in the entity groundings...')
    with open(
            '../data/kg/full_graph_multicellular_ents_GROUNDED_02May2024.json'
    ) as f:
        grounded_ents = json.load(f)
        print(
            f'There are {len(grounded_ents)} that received a grounding, and {len(set(grounded_ents.values()))} unique groundings.'
        )

    print('\nFormatting groundings...')
    groundings_to_ents = defaultdict(list)
    for ent, grd in grounded_ents.items():
        groundings_to_ents[grd].append(ent)

    print('\nContracting graph...')
    contracted_graph = contract_groups(graph, groundings_to_ents)

    print('\nSaving graph output...')
    nx.write_graphml(
        contracted_graph,
        '../data/kg/dygiepp_co-occurrence_CONTRACTED_grounded_graph_13May2024.graphml'
    )
