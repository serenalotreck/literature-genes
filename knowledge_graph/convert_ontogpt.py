"""
One-off script to convert the YAML output of OntoGPT to the entity and relation
dataframes needed for downstream use.

Author: Serena G. Lotreck
"""
import sys
sys.path.append('../../ontogpt/src/ontogpt/io')
from csv_wrapper import parse_yaml_predictions

print('\nReading in YAML...')
slim_ent_df, slim_rel_df = parse_yaml_predictions('../data/ontogpt_output/destol_all_slim_13May2024/output.txt', 'schema/desiccation.yaml')

print('\nSaving dataframes...')
slim_ent_df.to_csv('../data/kg/ontogpt_slim_ent_df_20May2024.csv', index=False)
slim_rel_df.to_csv('../data/kg/ontogpt_slim_rel_df_20May2024.csv', index=False)

print('\nDone!')
