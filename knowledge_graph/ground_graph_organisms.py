"""
One-off script to ground Multicellular Organisms from the graph to Taxonomy.
Uses file written out in entity_characterization.ipynb.
"""
from taxonerd import TaxoNERD
import sys
sys.path.append('../../desiccation-network/desiccation_network/build_citation_network/')
from classify_papers import make_ent_docs
import json
import time


print('\nReading in data...')
with open('../data/kg/full_graph_multicellular_ents_02May2024.txt') as f:
    ents = [line.strip() for line in f.readlines()]
num_chars = len(" ".join(ents))
print(f'There are {num_chars} characters of Multicellular_organism entities, '
      f'which will require {num_chars//10000 + 1} spacy documents.')

print('\nInitializing TaxoNERD model...')
taxonerd = TaxoNERD(prefer_gpu=True)
nlp = taxonerd.load(model="en_core_eco_biobert")

print('\nMaking spacy docs...')
multicellular_docs = make_ent_docs(ents, nlp)

print('\nPerforming linking...')
species_ids = {}
for i, doc in enumerate(multicellular_docs):
    start = time.time()
    doc = linker(doc)
    print(f'Time to apply linker on doc {i}: {time.time() - start: .2f}')
    for ent in doc.ents:
        ent_id = ent._.kb_ents[0][0].split(':')[1]
        species_ids[ent.text] = ent_id

print('\nSaving...')
with open('../data/kg/full_graph_multicellular_ents_GROUNDED_02May2024.json', 'w') as f:
    json.dump(species_ids, f)
