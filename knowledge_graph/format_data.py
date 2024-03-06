"""
Format a WOS result jsonl file for input to OntoGPT

Author: Serena G. Lotreck
"""
import argparse
from os.path import abspath
import jsonlines
from tqdm import tqdm


def main(wos_jsonl, output_dir):
    
    # Read in data
    print('\nReading in data...')
    with jsonlines.open(wos_jsonl) as reader:
        data = [obj for obj in reader]
    
    # Get text
    print('\nProcessing text...')
    docs = {}
    no_uid = 0
    no_abstract = []
    for paper in tqdm(data):
        try:
            uid = paper['UID']
        except KeyError:
            no_uid += 1
            continue
        text = paper['title']
        try:
            text += ' | ' + paper['abstract']
        except KeyError:
            no_abstract.append(uid)
            continue
        docs[uid] = text
    
    print(f'Of the provided {len(data)} papers, {len(docs)} were successfully formatted.')
    print(f'Of those missing, {no_uid} were dropped because they had no UID, and {len(no_abstract)} were dropped because they did not have an abstract.')
    
    # Save
    print('\nSaving documents...')
    for uid, doc in tqdm(docs.items()):
        savepath = f'{output_dir}/{uid}.txt'
        with open(savepath, 'w') as myf:
            myf.write(doc)
            
    print('\nDone!')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Format data')
    
    parser.add_argument('wos_jsonl', type=str,
            help='Path to jsonl file with WOS-sourced abstracts')
    parser.add_argument('output_dir', type=str,
            help='Path to a dir to dump individual abstract files')
    
    args = parser.parse_args()
    
    args.wos_jsonl = abspath(args.wos_jsonl)
    args.output_dir = abspath(args.output_dir)
    
    main(args.wos_jsonl, args.output_dir)