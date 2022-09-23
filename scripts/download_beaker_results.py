import os
import sys
import csv
import subprocess
import shutil
import yaml
from pprint import pprint


# Submissions from jetty app
COCHRANE_SUBMISSIONS_FILE = 'submissions/MSLR_Cochrane_leaderboard_submissions.csv'
MS2_SUBMISSIONS_FILE = 'submissions/MSLR_MS2_leaderboard_submissions.csv'

OUTPUT_DIR = 'submissions/predictions/'
OUTPUT_FILE = 'submissions/combined_submissions.csv'

if __name__ == '__main__':
    # read submissions
    submissions = []
    with open(COCHRANE_SUBMISSIONS_FILE, 'r') as f:
        reader = csv.DictReader(f, delimiter=',')
        for row in reader:
            row['subtask'] = 'Cochrane'
            submissions.append(row)

    with open(MS2_SUBMISSIONS_FILE, 'r') as f:
        reader = csv.DictReader(f, delimiter=',')
        for row in reader:
            row['subtask'] = 'MS2'
            submissions.append(row)

    # download corresponding dataset from beaker for each submission
    for entry in submissions:
        print(f"{entry['subtask']} {entry['Submission name']}")
        exp_id = entry['Beaker experiment ID']
        beaker_args = ['beaker', 'experiment', 'get', exp_id]
        exp_output = subprocess.check_output(' '.join(beaker_args), stderr=subprocess.STDOUT, shell=True)
        exp_info = exp_output.decode('utf-8').split()
        entry['ds_id'] = ''
        entry['ds_path'] = ''
        if exp_info[-1] == 'succeeded':
            beaker_args = ['beaker', 'experiment', 'spec', exp_id]
            spec_output = subprocess.check_output(' '.join(beaker_args), stderr=subprocess.STDOUT, shell=True)
            spec_info = yaml.load(spec_output.decode('utf-8'), Loader=yaml.Loader)
            ds_id = spec_info.get('tasks')[0].get('datasets')[0].get('source').get('beaker')
            if ds_id:
                out_dir = os.path.join(OUTPUT_DIR, f'{ds_id}')
                os.makedirs(out_dir, exist_ok=True)
                beaker_args = ['beaker', 'dataset', 'fetch', ds_id, '-o', out_dir]
                subprocess.run(beaker_args)
                print(f'{ds_id} downloaded to {out_dir}')
                entry['ds_id'] = ds_id
                entry['ds_path'] = os.path.join(out_dir, 'predictions.csv')

    # write all submissions along with dataset files to disk
    with open(OUTPUT_FILE, 'w') as outf:
        writer = csv.DictWriter(outf, delimiter=',', quotechar='|', fieldnames=submissions[0].keys())
        writer.writeheader()
        for entry in submissions:
            writer.writerow(entry)

    print('done.')

