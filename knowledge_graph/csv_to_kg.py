"""
Turns KGX-compliant CSVs into a graphml file.

Author: Serena G. Lotreck
"""
import argparse
from os.path import abspath
from os import listdir
import pandas as pd
import jsonlines
import networkx as nx

def map_doc_keys(ent_df, rel_df, input_dir):
    """
    Map the provided_by ID's to the original doc ID's. Assumes the order was
    not shuffled by OntoGPT.

    TODO confirm that order is not shuffled after implementing batching in
    OntoGPT.

    parameters:
        ent_df, df: entity dataframe from OntoGPT
        rel_df, df: relation dataframe from OntoGPT
        input_dir, str: path to directory used as input to OntoGPT
    """
    id_map = {rid: orig_id.split('.')[0] for rid, orig_id in zip(ent_df.provided_by.unique(), listdir(input_dir))}
    ent_df['UID'] = ent_df['provided_by'].map(id_map)
    rel_df['UID'] = rel_df['provided_by'].map(id_map)

    return ent_df, rel_df


def main(dset_path, ent_df, rel_df, ontogpt_input_dir, dset1_name, dset2_name,
        out_loc, out_prefix):

    # Read in data
    ent_df = pd.read_csv(ent_df)
    rel_df = pd.read_csv(rel_df)

    # Read in original dataset with desiccation attributes, map by UID
    with jsonlines.open(dset_path) as reader:
        dset = [obj for obj in reader]
    mapped_dset = {p['UID']: {dset1_name: p[dset1_name], dset2_name:
        p[dset2_name]}

    # Map UIDs
    ent_df, rel_df = map_doc_keys(ent_df, rel_df, ontogpt_input_dir)

    # Add original dataset attributes to entities

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Format KG')

    parser.add_argument('dset_path', type=str,
            help='Path to jsonl dataset with additional node and edge attrs')
    parser.add_argument('ent_df', type=str,
            help='Path to KGX-compliant entity df')
    parser.add_argument('rel_df', type=str,
            help='Path to KGX-compliant relation df')
    parser.add_argument('ontogpt_input_dir', type=str,
            help='Path to directory used for input to OntoGPT')
    parser.add_argument('dset1_name', type=str,
            help='String used as an attribute to identify papers from dset1')
    parser.add_argument('dset2_name', type=str,
            help='String used as an attribute to identify papers from dset2')
    parser.add_argument('out_loc', type=str,
            help='Path to directory to save output')
    parser.add_argument('out_prefix', type=str,
            help='String to prepend to save file name')

    args = parser.parse_args()

    args.dset_path = abspath(args.dset_path)
    args.ent_df = abspath(args.ent_df)
    args.rel_df = abspath(args.rel_df)
    args.ontogpt_input_dir = abspath(args.ontogpt_input_dir)
    args.out_loc = abspath(args.out_loc)

    main(args.dset_path args.ent_df, args.rel_df, args.ontogpt_input_dir,
            args.dset1_name, args.dset2_name, args.out_loc, args.out_prefix)


