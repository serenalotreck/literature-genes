"""
One-off script to map the arbistrary OntoGPT document IDs to their actual
document keys.

Author: Serena G. Lotreck
"""
import json
import yaml
from os import listdir
from os.path import splitext
from tqdm import tqdm


dirname = '../data/ontogpt_input/destol_final'
text2doc_key = {}
for f in listdir(dirname):
    doc_key = splitext(f)[0]
    with open(f'{dirname}/{f}') as f:
        doc_text = f.read()
        text2doc_key[doc_text] = doc_key

onto_id2doc_key = {}
with open('../data/ontogpt_output/destol_all_slim_13May2024/output.txt') as f:
    ontogpt_output = yaml.safe_load_all(f)
    for onto_doc in tqdm(ontogpt_output):
        onto_text = onto_doc['input_text']
        onto_id = onto_doc['extracted_object']['id']
        try:
            dyg_id = text2doc_key[onto_text]
            onto_id2doc_key[onto_id] = dyg_id
        except KeyError:
            print(f'No match found for document with OntoID {onto_id}.')

with open('../data/ontogpt_output/destol_onto_id_to_dygiepp_doc_key_23May2024.json', 'w') as f:
    json.dump(onto_id2doc_key, f)
