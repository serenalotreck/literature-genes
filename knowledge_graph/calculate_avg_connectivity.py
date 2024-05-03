"""
One-off script to calculate graph connectivity over time.
"""
import jsonlines
import networkx as nx
from tqdm import tqdm
import time
import json
from collections import defaultdict, Counter

print('\nReading in data...')
with jsonlines.open('../data/wos_files/drought_and_desiccation_combined_22Mar2024.jsonl') as reader:
    papers = [obj for obj in reader]
dygiepp_graph = nx.read_graphml('../data/kg/all_drought_dt_co_occurrence_graph_29Apr2024.graphml')

print('\nWrangling...')
uids_in_graph = set([
    uid for n, attrs in dygiepp_graph.nodes(data=True)
    for uid in attrs['uids_of_origin'].split(', ')
])
print(f'There are {len(uids_in_graph)} uids in the graph.')
paper_years = defaultdict(list)
for paper in papers:
    if paper['UID'] in uids_in_graph:
        if paper['is_desiccation']:
            if paper['is_drought']:
                paper_years['both'].append(int(paper['year']))
            else:
                paper_years['desiccation'].append(int(paper['year']))
        else:
            if paper['is_drought']:
                paper_years['drought'].append(int(paper['year']))
percent_counts = {cat: len(years)/5 for cat, years in paper_years.items()}
paper_years = {cat: dict(sorted(Counter(years).items(), key=lambda x:x[0])) for cat, years in paper_years.items()}

percent_cut_years = {}
for cat, year_counts in paper_years.items():
    cut_counts = [percent_counts[cat]*(i + 1) for i in range(5)]
    cut_years = {}
    cut_actual_counts = {}
    running_sum = 0
    for year, paper_count in year_counts.items():
        running_sum += paper_count
        if (running_sum >= cut_counts[0]) and (0 not in cut_years.keys()):
            cut_years[0] = year
            cut_actual_counts[0] = sum([v for y, v in year_counts.items() if int(y) <= year])
        elif (running_sum >= cut_counts[1]) and (1 not in cut_years.keys()):
            cut_years[1] = year
            cut_actual_counts[1] = sum([v for y, v in year_counts.items() if int(y) <= year])
        elif (running_sum >= cut_counts[2]) and (2 not in cut_years.keys()):
            cut_years[2] = year
            cut_actual_counts[2] = sum([v for y, v in year_counts.items() if int(y) <= year])
        elif (running_sum >= cut_counts[3]) and (3 not in cut_years.keys()):
            cut_years[3] = year
            cut_actual_counts[3] = sum([v for y, v in year_counts.items() if int(y) <= year])
        elif (running_sum >= cut_counts[4]) and (4 not in cut_years.keys()):
            cut_years[4] = year
            cut_actual_counts[4] = sum([v for y, v in year_counts.items() if int(y) <= year])
    percent_cut_years[cat] = dict(zip(cut_years.values(), cut_actual_counts.values()))

print('\nCalculating connectivity..')
connectivity_at_cuts = {}
for cat, cut_years in percent_cut_years.items():
    connects = {}
    for cut_year in tqdm(cut_years.keys()):
        per_year_graph = nx.Graph()
        if cat == 'desiccation':
            new_nodes = [(n, attrs) for n, attrs in dygiepp_graph.nodes(data=True)
                                   if (int(attrs['first_year_mentioned']) <= cut_year)
                        and (attrs['is_desiccation'] and not attrs['is_drought'])]
            new_edges = [(e1, e2, attrs) for e1, e2, attrs in dygiepp_graph.edges(data=True)
                                   if int(attrs['first_year_mentioned']) <= cut_year
                        and (attrs['is_desiccation'] and not attrs['is_drought'])]
        elif cat == 'drought':
            new_nodes = [(n, attrs) for n, attrs in dygiepp_graph.nodes(data=True)
                                   if (int(attrs['first_year_mentioned']) <= cut_year)
                        and (attrs['is_drought'] and not attrs['is_desiccation'])]
            new_edges = [(e1, e2, attrs) for e1, e2, attrs in dygiepp_graph.edges(data=True)
                                   if int(attrs['first_year_mentioned']) <= cut_year
                        and (attrs['is_drought'] and not attrs['is_desiccation'])]
        elif cat == 'both':
            new_nodes = [(n, attrs) for n, attrs in dygiepp_graph.nodes(data=True)
                                   if (int(attrs['first_year_mentioned']) <= cut_year)
                        and (attrs['is_drought'] and attrs['is_desiccation'])]
            new_edges = [(e1, e2, attrs) for e1, e2, attrs in dygiepp_graph.edges(data=True)
                                   if int(attrs['first_year_mentioned']) <= cut_year
                        and (attrs['is_drought'] and attrs['is_desiccation'])]
        _ = per_year_graph.add_nodes_from(new_nodes)
        _ = per_year_graph.add_edges_from(new_edges)
        start = time.time()
        print(f'Starting average connectivity calculation for category {cat} in cut year {cut_year}.')
        avg_connect = nx.average_node_connectivity(per_year_graph)
        print(f'Took {time.time() - start} seconds to run the average connectivity calculation.')
        connects[cut_year] = avg_connect
    connectivity_at_cuts[cat] = connects

with open('../data/kg/time_and_category_connectivity.json', 'w') as f:
    json.dump(connectivity_at_cuts, f)
