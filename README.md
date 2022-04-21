# MSLR Shared Task

The Multidocument Summarization for Literature Review (MSLR) Shared Task aims to study how medical evidence from different clinical studies are summarized in literature reviews. Reviews provide the highest quality of evidence for clinical care, but are expensive to produce manually. (Semi-)automation via NLP may facilitate faster evidence synthesis without sacrificing rigor. The MSLR shared task uses two datasets to assess the current state of multidocument summarization for this task, and to encourage the development of modeling contributions, scaffolding tasks, methods for model interpretability, and improved automated evaluation methods in this domain.

This task will be co-located at the [Scholarly Document Processing Workshop](https://sdproc.org/2022/) at [COLING 2022](https://coling2022.org/), to be held October 16-17, 2022, Online and in Gyeongju, South Korea. We invite submissions to our [leaderboard](#leaderboard) and [paper submissions](#submission-instructions) to the shared task. 

## Timeline

**Submission deadline:** July 11, 2022

**Acceptance notifications:** August 22, 2022

**Source files and Camera-ready PDFs:** September 5, 2022

**Workshop:** October 16-17, 2022 (Online and in Gyeongju, South Korea)

## Dataset Access

The MSLR2022 Shared Task uses two datasets, the MS^2 dataset and the Cochrane dataset. Inputs and target summaries for both datasets are formatted the same way and separated into train/dev/test splits. The MS^2 dataset is much larger, while the Cochrane dataset is smaller but contains cleaner data derived from Cochrane. Additionally, the MS^2 dataset includes something we refer to as Reviews-Info, which is a piece of background text derived from the review that can be used as an optional input during summarization. 

Download link: [here](https://ai2-s2-mslr.s3.us-west-2.amazonaws.com/mslr_data.tar.gz) (253 Mb; md5: `03cf9d`, sha1: `a8f1fa`)

```
wget https://ai2-s2-mslr.s3.us-west-2.amazonaws.com/mslr_data.tar.gz
tar -xvf mslr_data.tar.gz
```

This creates a data directory with two subdirectories corresponding to the two datasets below. See below for contents.

### MS^2 Dataset

This dataset consists of around 20K reviews and 470K studies collected from PubMed. For details on dataset contents and construction, please read the [MS^2 paper](https://arxiv.org/pdf/2104.06486.pdf).

The `mslr_data/ms2/` subdirectory should contain the following 8 files:

|             | Train       | Dev         | Test        |
| ----------- | ----------- | ----------- | ----------- |
| Inputs      | train-inputs.csv (sha1:`ca4852`)       | dev-inputs.csv (sha1:`a18022`)   | test-inputs.csv (sha1:`daaf87`)    |
| Targets     | train-targets.csv (sha1:`0da215`)      | dev-targets.csv (sha1:`133f80`)  |                       |
| Review information (Optional) | train-reviews-info.csv (sha1:`2311ca`) | dev-reviews-info.csv (sha1:`ff86d9`) | test-reviews-info.csv (sha1:`44f162`) |

### Cochrane Dataset

This is a dataset of 4.5K reviews collected from Cochrane systematic reviews. For details on dataset contents and construction, please read the [AMIA paper](https://arxiv.org/pdf/2008.11293.pdf).

The `mslr_data/cochrane/` subdirectory should contain the following 5 files:

|             | Train       | Dev         | Test        |
| ----------- | ----------- | ----------- | ----------- |
| Inputs      | train-inputs.csv (sha1:`b0b8f3`)       | dev-inputs.csv (sha1:`7dbb4e`)   | test-inputs.csv (sha1:`339e93`)    |
| Targets     | train-targets.csv (sha1:`7aa2e4`)      | dev-targets.csv (sha1:`70e1ee`)  |                       |

### Data Structure

Inputs are CSV files with the following columns:
* index: row number (ignore)
* ReviewID
* PMID (Pubmed ID)
* Title
* Abstract

Targets are CSV files with the following columns:
* index: row number (ignore)
* ReviewID
* Target (the summary to be generated)

Reviews-Info (only available for MS^2) are CSV files with the following columns:
* index: row number (ignore)
* ReviewID
* Background (the background information associated with the review; can be used optionally as input)

## Evaluation

Each submission to this shared task will be judged against gold review summaries on ROUGE score and by the evidence-inference-based divergence metric defined in the [MS^2 paper](https://arxiv.org/pdf/2104.06486.pdf). The evaluation script is available at `evaluator/evaluator.py`. 

The format of the predictions is expected to be a CSV file with the following columns:
* index: row number (ignored)
* ReviewID: same review id as in the input and target files
* Generated: containing the generated summary

To ensure that our leaderboard will correctly assess your submission, you may want to first test your the evaluator on your outputs for the dev set. 

To run the evaluator script on your predictions, first clone this repo:

```
git clone git@github.com:allenai/mslr-shared-task.git
```

Then setup your environment using [conda](https://docs.conda.io/en/latest/miniconda.html#latest-miniconda-installer-links):

```
conda env create -f environment.yml
conda activate mslr
conda install -c conda-forge jsonnet
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.4.0/en_core_sci_sm-0.4.0.tar.gz

# from the base directory of the repository, run:
python setup.py develop
```

Download the evidence inference models:

```
cd models/
wget https://ai2-s2-ms2.s3-us-west-2.amazonaws.com/evidence_inference_models.zip
unzip -q evidence_inference_models.zip
```

To test the evaluation script, try:

```
python evaluator/evaluator.py \
  --targets evaluator/test-targets.csv \
  --predictions evaluator/test-predictions.csv \
  --output output/metrics.json \
  --ei_param_file models/bert_pipeline_8samples.json \
  --ei_model_dir models/evidence_inference_models/ 
```

Once you are ready to submit, you can validate the evaluation on your predictions for the dev set (`dev-predictions.csv`):

```
python evaluator/evaluator.py \
  --targets path_to/dev-targets.csv \
  --predictions path_to/dev-predictions.csv \
  --output output/metrics.json \
  --ei_param_file models/bert_pipeline_8samples.json \
  --ei_model_dir models/evidence_inference_models/ 
```

When this script finishes, it will output metrics to `output/metrics.json` or another specified output file. The evaluator script can take several hours to run on CPU so please be patient.

Evaluating generated text is notoriously challenging. **In addition to automated metrics, a sample of summaries from select submissions to the leaderboards will also be subject to human evaluation to measure consistency with the target.** These human evaluation results will be shared with participants, and will be released at the end of this shared task to facilitate the development of multi-document summarization models and better automated summarization evaluation metrics. Participants will be invited to contribute to the submission of a dataset paper.

## Leaderboard

WIP: Information on the leaderboard will be added soon! For now, please refer to the [Evaluation section](#evaluation).

## Paper Submission Instructions

Participants are invited to submit a paper describing their shared task contribution. Submissions will be subject to peer-review. Accepted papers will be presented by the authors at the SDP workshop either as a talk or a poster. All accepted papers will be published in the workshop proceedings in the ACL Anthology (proceedings from previous years can be found [here](https://aclanthology.org/venues/sdp/)).

Submissions must be in PDF format and anonymized for review. All submissions must be written in English and follow the [COLING 2022 formatting requirements](https://coling2022.org/Cpapers).

Paper submissions can include up to 9 pages of content, plus unlimited references. Final versions of accepted papers will be allowed 1 additional page of content so that reviewer comments can be taken into account.

More details about submissions will be made available on our workshop [website](http://www.sdproc.org/). 

## Contact Us

You can reach the organizers by emailing mslr-organizers@googlegroups.com 

To receive updates , please join our mailing list:   https://groups.google.com/g/mslr-info or email lucyw@allenai.org to be added to our Slack workspace.

## Organizing Team

* Lucy Lu Wang, AI2
* Jay DeYoung, Northeastern University
* Byron Wallace, Northeastern University

## References

**DeYoung, Jay, Iz Beltagy, Madeleine van Zuylen, Bailey Kuehl and Lucy Lu Wang. "MS2: A Dataset for Multi-Document Summarization of Medical Studies." EMNLP (2021).**

```bibtex
@inproceedings{DeYoung2021MS2MS,
  title={MSˆ2: Multi-Document Summarization of Medical Studies},
  author={Jay DeYoung and Iz Beltagy and Madeleine van Zuylen and Bailey Kuehl and Lucy Lu Wang},
  booktitle={EMNLP},
  year={2021}
}
```

**Byron C. Wallace, Sayantani Saha, Frank Soboczenski, and Iain James Marshall. (2020). "Generating (factual?) narrative summaries of RCTs: Experiments with neural multi-document summarization." AMIA Annual Symposium.**

```bibtex
@article{Wallace2020GeneratingN,
  title={Generating (Factual?) Narrative Summaries of RCTs: Experiments with Neural Multi-Document Summarization},
  author={Byron C. Wallace and Sayantani Saha and Frank Soboczenski and Iain James Marshall},
  journal={AMIA Annual Symposium},
  year={2020},
  volume={abs/2008.11293}
}
```

