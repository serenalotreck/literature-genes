"""
Format Semantic Scholar search output for input to PubTator.

Author: Serena G. Lotreck
"""
import argparse
from os.path import abspath
import jsonlines
from tqdm import tqdm
import json


def format_ontogpt_input(flat_papers):
    """
    Format jsonl docs into strings for input into OntoGPT.

    parameters:
        flat_papers, dict: papers to format

    returns:
        ontogpt_papers, dict: formatted papers
    """
    ontogpt_papers = {}
    nonepapers = 0
    for pid, p in tqdm(flat_papers.items()):
        if pid is not None:
            try:
                if p['abstract'] is not None:
                    paper_text = p['title'] + ' ' + p['abstract']
                else:
                    paper_text = 'TITLE_ONLY ' + p['title']
            except KeyError:
                paper_text = 'TITLE_ONLY ' + p['title']
            ontogpt_papers[pid] = paper_text
        else:
            nonepapers += 1
    print(f'{nonepapers} papers were dropped because their  paperId was None')

    return ontogpt_papers


def format_pubtator_input(flat_papers):
    """
    Format jsonl docs into jsons for input into PubTator.

    parameters:
        flat_papers, dict: papers to format

    returns:
        pubtator_papers, dict: papers in pubtator format
    """
    pubtator_papers = {}
    nonepapers = 0
    for pid, p in tqdm(flat_papers.items()):
        if pid is not None:
            try:
                if p['abstract'] is not None:
                    paper_text = p['title'] + ' ' + p['abstract']
                else:
                    paper_text = 'TITLE_ONLY ' + p['title']
            except KeyError:
                paper_text = 'TITLE_ONLY ' + p['title']
            doc_json = {'sourcedb': 'user',
                    'sourceid': pid,
                    'text': paper_text}
            pubtator_papers[pid] = doc_json
        else:
            nonepapers += 1
    print(f'{nonepapers} papers were dropped because their  paperId was None')

    return pubtator_papers


def main(ie_type, data_to_format, input_dir_loc):

    # Read in the data
    print('\nReading in data...')
    with jsonlines.open(data_to_format) as reader:
        papers = []
        for obj in reader:
            papers.append(obj)

    # Determine if nested or not; if nested, flatten
    print('\nChecking if the data is nested...')
    try:
        flat_papers = {}
        for p in papers:
             flat_papers[p['paperId']] = p
             for r in p['references']:
                 flat_papers[r['paperId']] = r
        print('Data is nested!')
    except KeyError:
        flat_papers = {p['paperId']: p for p in papers}
        print('Data is not nested!')

    # Format as json
    print('\nFormatting papers as json...')
    if ie_type == 'pubtator':
        formatted_papers = format_pubtator_input(flat_papers)
    elif ie_type == 'ontogpt':
        formatted_papers = format_ontogpt_input(flat_papers)

    # Save
    print('\nSaving papers...')
    if ie_type == 'pubtator':
        for pid, paper_json in tqdm(formatted_papers.items()):
            with open(f'{input_dir_loc}/{pid}.json', 'w') as myf:
                json.dump(paper_json, myf)
    elif ie_type == 'ontogpt':
        for pid, paper_text in tqdm(formatted_papers.items()):
            with open(f'{input_dir_loc}/{pid}.txt', 'w') as myf:
                myf.write(paper_text)

    print('\nDone!')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Format PubTator input')

    parser.add_argument('ie_type', type=str,
            help='Options are "pubtator" or "ontogpt".')
    parser.add_argument('data_to_format', type=str,
            help='Path to .jsonl dataset to format. Can be flattened or have '
            'nested references.')
    parser.add_argument('input_dir_loc', type=str,
            help='Path to input directory to save files.')


    args = parser.parse_args()

    args.data_to_format = abspath(args.data_to_format)
    args.input_dir_loc = abspath(args.input_dir_loc)

    main(args.ie_type, args.data_to_format, args.input_dir_loc)
