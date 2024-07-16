"""
Script to train and evaluate a random forest model based on an upstream RESCAL
model. For a given set of RESCAL models, three sampling strategies will be used
and tested on an identical test set for RF models. Therefore, the number of RF
models trained is the number of RESCAL models provided times 3.

Author: Serena G. Lotreck
"""
import argparse
from os.path import abspath

import networkx as nx
from pykeen.triples import TriplesFactory
from pykeen.pipeline import pipeline
from pykeen.losses import BCEWithLogitsLoss
from pykeen.constants import PYKEEN_CHECKPOINTS
import torch
from pykeen.models import RESCAL
from itertools import combinations
from tqdm import tqdm
import json
import pandas as pd
from collections import Counter, defaultdict
from scipy.stats import randint
from sklearn.preprocessing import LabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, roc_auc_score, precision_recall_curve, PrecisionRecallDisplay, average_precision_score, RocCurveDisplay, auc, roc_curve
from sklearn.decomposition import PCA
import pickle
from random import choice, sample
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['pdf.fonttype'] = 42
import numpy as np
from itertools import cycle


def auc_scores(y_onehot_test, y_score, label_map):
    """
    Get the ROC and PR curves, plus the AUROC and AP metrics for each class, and
    averaged for multiclass.
    """
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    prec = dict()
    rec = dict()
    prc_auc = dict()
    n_classes = len(label_map)
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_onehot_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        prec[i], rec[i], _ = precision_recall_curve(y_onehot_test[:, i],
                                                    y_score[:, i])
        prc_auc[i] = average_precision_score(y_onehot_test[:, i], y_score[:,
                                                                          i])

    # ROC
    fpr_grid = np.linspace(0.0, 1.0, 1000)
    mean_tpr = np.zeros_like(fpr_grid)
    for i in range(n_classes):
        mean_tpr += np.interp(fpr_grid, fpr[i], tpr[i])  # linear interpolation
    # Average it and compute AUC
    mean_tpr /= n_classes
    fpr["macro"] = fpr_grid
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # PRC
    prec["micro"], rec["micro"], _ = precision_recall_curve(
        y_onehot_test.ravel(), y_score.ravel())
    prc_auc["micro"] = average_precision_score(y_onehot_test,
                                               y_score,
                                               average="micro")
    prc_auc["macro"] = average_precision_score(y_onehot_test,
                                               y_score,
                                               average="macro")

    return fpr, tpr, roc_auc, rec, prec, prc_auc, n_classes


def evaluate_model(train_df, test_df, best_rf, label_map, dataset, outloc, outprefix,
                   short_rescname, samp_strat):
    """
    Evaluate model and generate visualizations.

    parameters:
        train_df, pandas df: train set with features
        test_df, pandas df: test set with features
        best_rf, sklearn RandomForestClassifier
        label_map, dict: label map
        outloc, str: path to save
        outprefix, str: prefix to append to save name
        short_rescname, str: name of the RESCAL model we're on
        samp_strat, str: name of sampling strategy we're on
    """
    # Format data
    X_test = test_df.drop(columns='label').to_numpy()
    y_test = test_df['label'].to_numpy()
    y_pred = best_rf.predict(X_test)
    X_train = train_df.drop(columns='label').to_numpy()
    y_train = train_df['label'].to_numpy()

    # Make savename prefix
    figure_save_prefix = f'{outloc}/{outprefix}_{short_rescname}_{samp_strat}'

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    if dataset == 'dt':
        ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=['negative', 'desiccation', 'drought',
                            'both']).plot()
    elif dataset == 'gpe':
        ConfusionMatrixDisplay(confusion_matrix=gpe_cm).plot()
    plt.savefig(f'{figure_save_prefix}_confusion_matrix.pdf',
                transparent='True',
                format='pdf',
                dpi=600,
                bbox_inches='tight')

    # Numeric scores
    f1 = f1_score(y_pred, y_test, average='macro')
    aucroc_score = roc_auc_score(y_test,
                                 best_rf.predict_proba(X_test),
                                 average='macro',
                                 multi_class='ovr')
    print(
        f'\nModel {short_rescname}/{samp_strat} has a macro-averaged F1 score of {f1:.2f} and a one-v-rest macro-averaged AUROC of {aucroc_score:.2f}'
    )

    # AUROC curves
    y_score = best_rf.predict_proba(X_test)
    label_binarizer = LabelBinarizer().fit(y_train)
    y_onehot_test = label_binarizer.transform(y_test)
    y_onehot_test.shape  # (n_samples, n_classes)
    fpr, tpr, roc_auc, rec, prec, prc_auc, n_classes = auc_scores(
        y_onehot_test, y_score, label_map)
    reverse_label_map = {v: k for k, v in label_map.items()}
    fig, ax = plt.subplots(figsize=(6, 6))

    plt.plot(
        fpr["macro"],
        tpr["macro"],
        label=f"macro-average ROC curve (AUC = {roc_auc['macro']:.2f})",
        color="deeppink",
        linestyle="dashdot",
    )

    colors = cycle(["aqua", "darkorange", "cornflowerblue", "springgreen"])
    for class_id, color in zip(range(n_classes), colors):
        RocCurveDisplay.from_predictions(
            y_onehot_test[:, class_id],
            y_score[:, class_id],
            name=f"ROC curve for {reverse_label_map[class_id]}",
            color=color,
            ax=ax,
            plot_chance_level=(class_id == 3),
        )
    _ = ax.set_xlabel("False Positive Rate")
    _ = ax.set_ylabel("True Positive Rate")
    ax.legend(loc=(1.1, 0))
    plt.savefig(f'{figure_save_prefix}_AUROC_curves.pdf',
                transparent='True',
                format='pdf',
                dpi=600,
                bbox_inches='tight')


def generate_features(model, checkpoint, dataset, node_pairs, training_tf):
    """
    Generate the feature matrix for a given set of node pairs.

    parameters:
        moodel, RESCAL model: model to use
        checkpoint, PYKEEN CHECKPOINT: checkpoint for model
        dataset, str: "gpe" or "dt"
        node_pairs, list of tuple: pairs for which to generate a feature set

    returns:
        node_df, pandas df: feature table,
        label_map, dict: label map
    """
    node_reps = model.entity_representations[0]()
    ent_map = checkpoint['entity_to_id_dict']

    if dataset == 'dt':
        label_map = {'desiccation': 1, 'drought': 2, 'both': 3, 'negative': 0}
    elif dataset == 'gpe':
        trips = training_tf.triples
        train_ent_types = Counter([t[1] for t in trips])
        labels = [k for k, v in train_ent_types.items() if v > 2000
                  ] + ['negative']
        label_map = {k: i for i, k in enumerate(labels)}

    node_pairs_flattened = {k: v for k, v in node_pairs['positives'].items()}
    try:
        node_pairs_flattened['negative'] = [
            t for t_list in node_pairs['negatives'].values() for t in t_list
        ]
    except AttributeError:
        node_pairs_flattened['negative'] = node_pairs['negatives']


    all_node_feat_dfs = []
    for lab, pair_list in node_pairs_flattened.items():
        class_feats = {}
        for pair in pair_list:
            feats = node_reps[ent_map[pair[0]]].tolist()
            feats.extend(node_reps[ent_map[pair[1]]].tolist())
            class_feats[pair] = feats
        class_df = pd.DataFrame.from_dict(class_feats, orient='index')
        class_df['label'] = label_map[lab]
        all_node_feat_dfs.append(class_df)
    node_df = pd.concat(all_node_feat_dfs).sample(frac=1)  # Shuffle the data

    return node_df, label_map


def get_class_pairs(semantic_trips,
                    num_inst=2000,
                    types_to_exclude=None,
                    sampling_method='corrupt',
                    node_reps=None,
                    ent_map=None,
                    neg_balance='total',
                    prev_positives=None):
    """
    Get triples for each of 'desiccation', 'drought', 'both', 'negative'.

    parameters:
        semantic_trips, list of list: triples
        num_inst, int: number of instances to get for each class.
        types_to_exclude, list of str: relation types that should be ignored.
        sampling_method, str: 'corrupt', 'random', 'distance'
        node_reps, dict: node representations, only required if sampling_method == 'embedding'
        ent_map, dict: mapping from entity names to id's, only required if sampling_method == 'embedding'
        neg_balance, str: 'total' gives n_classes*num_inst negative samples, 'one' gives num_inst number
            of negative samples
        prev_positives, dict: if the dataset should only change in the negative set, pass the
            positive set of pairs here to maintain the same positive instances.

    returns:
        data, dict: keys are class names, values are pairs for the class.
    """
    types_to_exclude = [] if types_to_exclude is None else types_to_exclude

    # If any of the types have less than the required instances, check here
    type_counts = Counter([t[1] for t in semantic_trips])
    too_few = {
        k: v
        for k, v in type_counts.items()
        if (v < num_inst) and (k not in types_to_exclude)
    }
    if len(too_few) > 0:
        print(
            f'{len(too_few)} entity types have less than the requested number of instances. They will '
            'be returned with the number that they have.')
    length_check = {}

    # Get positive instances
    if prev_positives is None:
        print('Sampling new positives...')
        positives = defaultdict(list)
        for trip in semantic_trips:
            trip_type = trip[1]
            if trip_type not in types_to_exclude:
                if len(positives[trip_type]) < num_inst:
                    positives[trip_type].append((trip[0], trip[2]))
                else:
                    if len(too_few) == 0:
                        if sum([len(v) for v in positives.values()
                                ]) == num_inst * len(positives.keys()):
                            break
                    else:
                        short_finished = [
                            True if len(positives[k]) == too_few[k] else False
                            for k in too_few.keys()
                        ]
                        long_finished = [
                            True if (k not in too_few.keys()) and
                            (len(positives[k]) == num_inst) else False
                            for k in positives.keys()
                        ]
                        if all(short_finished) and all(long_finished):
                            break
    else:
        print('Using previous positives...')
        positives = prev_positives

    # Generate negative instances
    all_ents = list(
        set([trip[0] for trip in semantic_trips] +
            [trip[2] for trip in semantic_trips]))
    if sampling_method == 'random':
        negs = []
        for pair in combinations(all_ents, 2):
            not_pos = True
            for r_type, pairs in positives.items():
                if pair in pairs:
                    not_pos = False
                elif (pair[1], pair[0]) in pairs:
                    not_pos = False
            if not_pos:
                negs.append(pair)
            if neg_balance == 'one':
                if len(negs) == num_inst:
                    break
            elif neg_balance == 'total':
                if len(negs) == sum([len(v) for v in positives.values()]):
                    break

    elif sampling_method == 'corrupt':  # Currently implemented for tail-only corruption
        negs = defaultdict(list)
        for trip_type, pos_trips in tqdm(positives.items()):
            for pos_t in pos_trips:
                head = pos_t[0]
                old_tail = pos_t[1]
                is_neg = False
                while not is_neg:
                    new_tail = choice(all_ents)
                    if ((head, new_tail) not in pos_trips) and (
                        (head, new_tail) not in negs[trip_type]):
                        is_neg = True
                neg_trip = (head, new_tail)
                negs[trip_type].append(neg_trip)
                if neg_balance == 'one':
                    if len(negs[trip_type]) == len(pos_trips) // len(positives):
                        if sum([len(ts) for ts in negs.values()]) == len(pos_trips) - 1:
                            continue # Get the last instance to make the balance equal
                        else:
                            break
                    if sum([len(ts) for ts in negs.values()]) == len(pos_trips):
                        break
                elif neg_balance == 'total':
                    if len(negs) == sum([len(v) for v in positives.values()]):
                        break

    elif sampling_method == 'embedding':
        rev_ent_map = {v: k for k, v in ent_map.items()}
        negs = defaultdict(list)
        for trip_type, pos_trips in tqdm(positives.items()):
            # For each triple
            num_class_negs = 0
            for pos_t in pos_trips:
                # Randomly sample 50 possible replacement tails
                possible_tails = sample(all_ents, 500)
                # Remove any that are the same as the current tail
                possible_tails = [t for t in possible_tails if t != pos_t[1]]
                # Get embedding subset for these tails
                orig_idxs = [ent_map[t] for t in possible_tails]
                tail_reps = node_reps[orig_idxs]
                orig_tail_rep = node_reps[ent_map[pos_t[1]]].detach().numpy()
                # Calculate euclidean distance between original and possible new tails
                euc_dist = cdist([orig_tail_rep], tail_reps.detach().numpy())
                # Calculate the softmax probabilities on these distances
                prob_mat = softmax(1 / euc_dist)
                # Get the top five and choose one at random
                top_5_flat_ind = np.argpartition(prob_mat.flatten(), -5)[-5:]
                chosen_checked = False
                while not chosen_checked:
                    chosen_flat_ind = choice(top_5_flat_ind)
                    # Convert back to the entities that generated this score
                    chosen_tail_actual_ind = orig_idxs[chosen_flat_ind]
                    chosen_tail_semantic = rev_ent_map[chosen_tail_actual_ind]
                    # Check if negative triple already isin negative set, if so choose again
                    neg_trip = (pos_t[0], chosen_tail_semantic)
                    if neg_trip not in negs[trip_type]:
                        chosen_checked = True
                negs[trip_type].append(neg_trip)
                num_class_negs += 1
                if neg_balance == 'one':
                    if num_class_negs == len(pos_trips) // len(positives):
                        if sum([len(ts) for ts in negs.values()]) == len(pos_trips) - 1:
                            continue # Get the last instance to make the balance equal
                        else:
                            break
                    if sum([len(ts) for ts in negs.values()]) == len(pos_trips):
                        break

    data = {'positives': positives, 'negatives': negs}

    return data


def get_predicate(row):
    if row.is_drought:
        if row.is_desiccation:
            return 'both'
        else:
            return 'drought'
    else:
        if row.is_desiccation:
            return 'desiccation'


def load_rescal(rescal_ckpt_name, graph_path, dataset, model_random_seed,
                data_random_seed):
    """
    Load a RESCAL model
    """
    checkpoint = torch.load(PYKEEN_CHECKPOINTS.joinpath(rescal_ckpt_name),
                            map_location=torch.device('cpu'))
    graph = nx.read_graphml(graph_path)
    edgelist = nx.to_pandas_edgelist(graph)
    if dataset == 'dt':
        edgelist['predicate'] = edgelist.apply(get_predicate, axis=1)
    triples = edgelist[['source', 'predicate', 'target']].to_numpy()
    print(f'Snapshot of triples: {triples[:5]}')
    tf = TriplesFactory.from_labeled_triples(
        triples,
        create_inverse_triples=True,
        entity_to_id=checkpoint['entity_to_id_dict'],
        relation_to_id=checkpoint['relation_to_id_dict'])
    training, validation, testing = tf.split([0.8, 0.1, 0.1],
                                             random_state=data_random_seed)
    my_model = RESCAL(triples_factory=training, random_seed=model_random_seed)
    my_model.load_state_dict(checkpoint['model_state_dict'])

    return training, testing, checkpoint, my_model


def main(rescal_ckpt_names, graph_path, dataset, model_random_seed,
         data_random_seed, num_test_inst, num_train_inst, outloc, outprefix):

    # Load model and training splits
    print('\nLoading RESCAL model and training splits...')
    rescal_models = defaultdict(dict)
    for ckpt in rescal_ckpt_names:
        training, testing, checkpoint, model = load_rescal(
            ckpt, graph_path, dataset, model_random_seed, data_random_seed)
        rescal_models[ckpt]['training'] = training
        rescal_models[ckpt]['testing'] = testing
        rescal_models[ckpt]['checkpoint'] = checkpoint
        rescal_models[ckpt]['model'] = model

    # Assert that all RESCAL models were trained on the same splits
    for k in ['training', 'testing']:
        allsets = [v[k] for v in rescal_models.values()]
        assert all(
                i == allsets[0] for i in allsets
            ), f"Not all RESCAL models have the same splits for {k}, please try again"

    # Get the one test set to be used for all models
    print('\nGetting the test set to be used for all models, random sampling '
          'strategy will be used for negatives.')
    rescal_test_trips = list(rescal_models.values())[0]['testing'].triples
    test_pairs = get_class_pairs(rescal_test_trips,
                                 num_inst=num_test_inst,
                                 sampling_method='random',
                                 neg_balance='one')
    counted_pos = {k:len(v) for k,v in test_pairs["positives"].items()}
    test_classes = [k for k in test_pairs.keys()] + ['negative']

    # Get the training set for each sampling strategy
    print('\nGenerating positive training set...')
    rescal_train_trips = list(rescal_models.values())[0]['training'].triples
    if dataset == 'gpe':
        train_ent_types = Counter([t[1] for t in rescal_train_trips])
        types_to_exclude = [i for i in train_ent_types if i not in
                test_classes]
    else:
        types_to_exclude = None
    random_train_pairs = get_class_pairs(rescal_train_trips,
                                         num_inst=num_train_inst,
                                         types_to_exclude=types_to_exclude,
                                         sampling_method='random',
                                         neg_balance='total')
    positives_to_reuse = random_train_pairs['positives']

    # Pull the training negatives and train model for each pairing
    print(
        '\nPulling negative samples and training for each RESCAL/RF comboination...'
    )
    for resc_name, components in rescal_models.items():
        short_rescname = resc_name.split('.')[0]
        for samp_strat in ['random', 'corrupt', 'embedding']:
            print(f'\nOn combination {resc_name} + {samp_strat}')
            if samp_strat == 'embedding':
                train_pairs = get_class_pairs(
                    train_trips_semantic,
                    num_inst=2000,
                    types_to_exclude=types_to_exclude,
                    sampling_method=samp_strat,
                    node_reps=components['model'].entity_representations[0](),
                    ent_map=components['checkpoint']['entity_to_id_dict'],
                    neg_balance='total',
                    prev_positives=positives_to_reuse)
            else:
                train_pairs = get_class_pairs(
                    rescal_train_trips,
                    num_inst=num_train_inst,
                    types_to_exclude=types_to_exclude,
                    sampling_method='random',
                    neg_balance='total',
                    prev_positives=positives_to_reuse)
            train_df, label_map = generate_features(components['model'],
                                         components['checkpoint'], dataset,
                                         train_pairs, components['training'])
            test_df, _ = generate_features(components['model'],
                                        components['checkpoint'], dataset,
                                        test_pairs, components['testing'])
            train_savename = f'{outloc}/{outprefix}_{short_rescname}_{samp_strat}_training_set.csv'
            test_savename = f'{outloc}/{outprefix}_{short_rescname}_{samp_strat}_testing_set.csv'
            train_df.to_csv(train_savename)
            test_df.to_csv(test_savename)
            print(f'Saved training and test sets to {outloc}.')

            print('Training model...')
            X_train = train_df.drop(columns='label').to_numpy()
            y_train = train_df['label'].to_numpy()
            params_to_test = {
                'n_estimators': randint(100, 500),
                'max_depth': randint(1, 50),
                'criterion': ['gini', 'entropy', 'log_loss']
            }

            rf = RandomForestClassifier(verbose=1)
            rand_search = RandomizedSearchCV(
                rf, param_distributions=params_to_test, cv=5, verbose=1)
            rand_search.fit(X_train, y_train)
            best_rf = rand_search.best_estimator_
            print('Best hyperparameters:', rand_search.best_params_)
            model_savename = f'{outloc}/{outprefix}_{short_rescname}_{samp_strat}_RF_model.pk'
            with open(model_savename, 'wb') as f:
                pickle.dump(best_rf, f)
            print(f'Saved model as {model_savename}')

            print('\nEvaluating model...')
            evaluate_model(train_df, test_df, best_rf, label_map, dataset, outloc, outprefix,
                           short_rescname, samp_strat)

    print('\nDone!')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train and eval RF')

    parser.add_argument(
        '-rescal_ckpt_names',
        nargs='+',
        help='The .pt checkpoint names for RESCAL models to test')
    parser.add_argument(
        '-graph_path',
        type=str,
        help='Path to networkx graphml file from which to derive triples')
    parser.add_argument(
        '-dataset',
        type=str,
        help='Which dataset is being used, options are "dt" or "gpe"')
    parser.add_argument(
        '-model_random_seed',
        type=int,
        default=5678,
        help='Random seed for model instantiation, must be the same as the '
        'RESCAL models were trained with, and all RESCAL models must have '
        'the same random seed')
    parser.add_argument(
        '-data_random_seed',
        type=int,
        default=5678,
        help='Random seed for dataset splitting, must be the same as the '
        'RESCAL models were trained with, and all RESCAL models must have '
        'the same random seed')
    parser.add_argument(
        '-num_test_inst',
        type=int,
        default=1000,
        help='Numebr of instances for each class in the test set for RF')
    parser.add_argument(
        '-num_train_inst',
        type=int,
        default=2000,
        help='Numebr of instances for each class in the train set for RF')
    parser.add_argument('-outloc', type=str, help='Path to save outputs')
    parser.add_argument('-outprefix',
                        type=str,
                        help='String to prepend to output filenames')

    args = parser.parse_args()

    args.graph_path = abspath(args.graph_path)
    args.outloc = abspath(args.outloc)

    main(args.rescal_ckpt_names, args.graph_path, args.dataset,
         args.model_random_seed, args.data_random_seed, args.num_test_inst,
         args.num_train_inst, args.outloc, args.outprefix)
