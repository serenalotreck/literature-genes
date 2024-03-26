"""
Combine datasets, assigning attributes for original dataset membership.

Author: Serena G. Lotreck
"""
import argparse
from os.path import abspath
import jsonlines


def main(dataset1_path, dataset1_name, dataset2_path, dataset2_name, savepath):

    # Read in both datasets
    print('\nReading in dataset...')
    datasets = {}
    for dset_name, dset_path in {dataset1_name: dataset1_path, dataset2_name:
            dataset2_path}.items():
        with jsonlines.open(dset_path) as reader:
            dset = [obj for obj in reader]
            datasets[dset_name] = dset
    print(f'There are two datasets named {dataset1_name} and {dataset2_name}. '
            f'{dataset1_name} has {len(datasets[dataset1_name])} documents and '
            f'{dataset1_name} has {len(datasets[dataset2_name])} documents.')

    # Check for overlaps
    dset1_uids = [d['UID'] for d in datasets[dataset1_name]]
    dset2_uids = [d['UID'] for d in datasets[dataset2_name]]
    overlap = list(set(dset1_uids).intersection(set(dset2_uids)))
    print(f'There are {len(overlap)} documents between the two datasets.')

    # Assign attributes and combine
    print('\nCombining...')
    attr1_name = f'is_{dataset1_name}'
    attr2_name = f'is_{dataset2_name}'
    combined_dset = []
    for doc in datasets[dataset1_name]:
        doc[attr1_name] = True
        if doc['UID'] not in overlap:
            doc[attr2_name] = False
        else:
            doc[attr2_name] = True
        combined_dset.append(doc)
    for doc in datasets[dataset2_name]:
        if doc['UID'] in overlap:
            continue
        else:
            doc[attr2_name] = True
            doc[attr1_name] = False
            combined_dset.append(doc)
    print(f'There are {len(combined_dset)} documents in the final dataset.')

    # Save
    print('\nSaving...')
    with jsonlines.open(savepath, 'w') as writer:
        writer.write_all(combined_dset)
    print(f'Saved combined dataset as {savepath}')

    print('\nDone!')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Combine datasets')

    parser.add_argument('dataset1_path', type=str,
            help='Path to first dataset')
    parser.add_argument('dataset1_name', type=str,
            help='String to use for dset1 attribute name')
    parser.add_argument('dataset2_path', type=str,
            help='Path to second dataset')
    parser.add_argument('dataset2_name', type=str,
            help='String to use for dset2 attribute name')
    parser.add_argument('savepath', type=str,
            help='Full filepath to save output')

    args = parser.parse_args()

    args.dataset1_path = abspath(args.dataset1_path)
    args.dataset2_path = abspath(args.dataset2_path)
    args.savepath = abspath(args.savepath)

    main(args.dataset1_path, args.dataset1_name, args.dataset2_path, args.dataset2_name,
            args.savepath)
