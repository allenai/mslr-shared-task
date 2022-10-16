import os
import csv
import argparse
from datetime import date
from collections import defaultdict
from copy import copy
import random

"""
Example command to run:
python sample_submissions_for_annotation.py -d Cochrane -s submissions/combined_submissions.csv -t submissions/test-targets-cochrane.csv -o submissions/sample_for_annotation/ -n 100 -p 50
"""

SKIP_EXP = {'01G99JJWDSQNQDKMK5ZGRQD4RK'}
INCL_EXP = {'01G4NE2DDS5G6Q047M97PX7SGV'}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sample MSLR submissions for annotation')
    parser.add_argument(
        '--datasets', '-d',
        help='which subtask datasets to sample from (options: Cochrane, MS2), separate with comma if both',       
        required=True)
    parser.add_argument(
        '--submissions', '-s',
        help='File containing all submissions to sample from (output of download_beaker_results.py',
        required=True)
    parser.add_argument(
        '--targets', '-t',
        help='File containing all target summaries',
        required=True)
    parser.add_argument(
        '--output', '-o',
        help='Directory to output sample',
        required=True)
    parser.add_argument(
        '--number_to_sample', '-n',
        help='Number of instances to sample per submission (default=100)',
        default=100)
    parser.add_argument(
        '--number_overlap', '-p',
        help='Number of sampled instances that overlap between all submissions to that subtask (default=50)',
        default=50)

    args = parser.parse_args()
    subtasks = args.datasets.split(',')
    submission_file = args.submissions
    target_file = args.targets
    output_dir = args.output
    num_per_system = args.number_to_sample
    num_overlap = args.number_overlap

    os.makedirs(output_dir, exist_ok=True)
    assert os.path.exists(submission_file)
    assert os.path.exists(target_file)
    assert num_overlap <= num_per_system

    # form output datafile
    today = date.today()
    datestr = today.strftime("%Y%m%d")
    output_file_1 = os.path.join(output_dir, f"{'-'.join(subtasks)}_n{num_per_system}_o{num_overlap}_{datestr}_split1.csv")
    output_file_2 = os.path.join(output_dir, f"{'-'.join(subtasks)}_n{num_per_system}_o{num_overlap}_{datestr}_split2.csv")
    if os.path.exists(output_file_1) or os.path.exists(output_file_2):
        print('ERROR: output files already exists!')

    # read all submissions and keep those that match subtasks of interest
    submissions_to_sample = []
    with open(submission_file, 'r') as f:
        reader = csv.DictReader(f, delimiter=',')
        for row in reader:
            if row['subtask'] in subtasks and ((row['Type'] == 'model' and row['Submission name'] and row['Beaker experiment ID'] not in SKIP_EXP) or row['Beaker experiment ID'] in INCL_EXP):
                submissions_to_sample.append(row)

    # read targets
    targets = dict()
    with open(target_file, 'r') as f:
        reader = csv.DictReader(f, delimiter=',')
        for row in reader:
            targets[row['ReviewID']] = row['Target']

    # all systems
    all_exp_ids = [entry['Beaker experiment ID'] for entry in submissions_to_sample]
    all_ds_paths = [entry['ds_path'] for entry in submissions_to_sample]

    # read all data
    data = defaultdict(dict)
    for entry in submissions_to_sample:
        exp_id = entry['Beaker experiment ID']
        dset_file = entry['ds_path']
        with open(dset_file, 'r') as f:
            reader = csv.DictReader(f, delimiter=',')
            for row in reader:
                data[row['ReviewID']][exp_id] = row['Generated']
    
    # convert to list
    datarows = []
    for rev_id, target in targets.items():
        generated_by_exp = data[rev_id]
        generated_texts = [generated_by_exp.get(exp_id, '') for exp_id in all_exp_ids]
        datarows.append([rev_id] + [target] + generated_texts)

    all_output_file = os.path.join(output_dir, f"{'-'.join(subtasks)}_all_{datestr}.csv")
    headers = ['ReviewID', 'Target'] + all_exp_ids
    with open(all_output_file, 'w') as outf:
        writer = csv.writer(outf, delimiter=',', quotechar='"')
        writer.writerow(headers)
        for row in datarows:
            writer.writerow(row)
    
    # randomize and sample
    sampled_data = []
    num_uniq_per_sys = num_per_system - num_overlap
    random.shuffle(datarows)
    overlapping_sample = datarows[:num_overlap]
    for row in overlapping_sample:
        generated = row[2:]
        for exp_id, gen in zip(all_exp_ids, generated):
            sampled_data.append([exp_id, row[0], row[1], gen])
    remainder = datarows[num_overlap:]
    for ind, exp_id in enumerate(all_exp_ids):
        sys_sample = random.sample(remainder, num_uniq_per_sys)
        for row in sys_sample:
            gen = row[2 + ind]
            sampled_data.append([exp_id, row[0], row[1], gen])

    # split into two annotation tasks
    sampled_rev_ids = set([row[1] for row in sampled_data])
    double_annotation_rev_ids = random.sample(sampled_rev_ids, 20)
    double_annotation_split = [row for row in sampled_data if row[1] in double_annotation_rev_ids]
    single_annotation_split = [row for row in sampled_data if row[1] not in double_annotation_rev_ids]

    # randomize order but keep rev id groups
    random.shuffle(double_annotation_split)
    double_annotation_split.sort(key=lambda x: x[1])

    random.shuffle(single_annotation_split)
    single_annotation_split.sort(key=lambda x: x[1])

    # split single annotation section into 2
    midpoint = int(len(single_annotation_split) / 2)
    split1 = single_annotation_split[:midpoint]
    split2 = single_annotation_split[midpoint:]

    headers = ['ExpID', 'ReviewID', 'Target Summary', 'Generated Summary']

    with open(output_file_1, 'w') as outf:
        writer = csv.writer(outf, delimiter=',', quotechar='"')
        writer.writerow(headers)
        for row in split1[:10]:
            writer.writerow(row)
        for row in double_annotation_split:
            writer.writerow(row)
        for row in split1[10:]:
            writer.writerow(row)

    with open(output_file_2, 'w') as outf:
        writer = csv.writer(outf, delimiter=',', quotechar='"')
        writer.writerow(headers)
        for row in split2[:10]:
            writer.writerow(row)
        for row in double_annotation_split:
            writer.writerow(row)
        for row in split2[10:]:
            writer.writerow(row)

    print('done.')
    






    