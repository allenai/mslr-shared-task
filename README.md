<p align="center">
  <img src="https://leaderboard.allenai.org/assets/images/leaderboard/mslr/logo.svg" alt="Logo of MSLR shared task" width="20%">
</p>

# MSLR Shared Task 2022<br>Multidocument Summarization for Literature Review

The Multidocument Summarization for Literature Review (MSLR) Shared Task aims to study how medical evidence from different clinical studies are summarized in literature reviews. Reviews provide the highest quality of evidence for clinical care, but are expensive to produce manually. (Semi-)automation via NLP may facilitate faster evidence synthesis without sacrificing rigor. The MSLR shared task uses two datasets to assess the current state of multidocument summarization for this task, and to encourage the development of modeling contributions, scaffolding tasks, methods for model interpretability, and improved automated evaluation methods in this domain. 

This task is co-located at the [Scholarly Document Processing Workshop](https://sdproc.org/2022/) at [COLING 2022](https://coling2022.org/), held October 16-17, 2022, Online and in Gyeongju, South Korea.

## Timeline

**Workshop:** October 16-17, 2022 (Online and in Gyeongju, South Korea)

## Dataset Access

The MSLR2022 Shared Task uses two datasets, the MS^2 dataset and the Cochrane dataset. Inputs and target summaries for both datasets are formatted the same way and separated into train/dev/test splits. The MS^2 dataset is much larger, while the Cochrane dataset is smaller but contains cleaner data derived from Cochrane. Additionally, the MS^2 dataset includes something we refer to as Reviews-Info, which is a piece of background text derived from the review that can be used as an optional input during summarization. 

The dataset is available through Huggingface datasets: [https://huggingface.co/datasets/allenai/mslr2022](https://huggingface.co/datasets/allenai/mslr2022) 

You can also acquire it directly at the following link:

Download link: [here](https://ai2-s2-mslr.s3.us-west-2.amazonaws.com/mslr_data.tar.gz) (253 Mb; md5: `d1ae52`, sha1: `3ba174`)

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
| Targets     | train-targets.csv (sha1:`417a18`)      | dev-targets.csv (sha1:`4baf55`)  |                       |
| Review information (Optional) | train-reviews-info.csv (sha1:`da1a1c`) | dev-reviews-info.csv (sha1:`60cc60`) | test-reviews-info.csv (sha1:`6a9c1e`) |

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
* ReviewID (Pubmed ID of the review)
* PMID (Pubmed ID of the input study)
* Title of input study
* Abstract of input study

Targets are CSV files with the following columns:
* index: row number (ignore)
* ReviewID (Pubmed ID of the review)
* Target (the target summary, extracted from the review)

Reviews-Info (only available for MS^2) are CSV files with the following columns:
* index: row number (ignore)
* ReviewID (Pubmed ID of the review)
* Background (the background information associated with the review; can be used optionally as input)

## Evaluation

Each submission to this shared task will be judged against gold review summaries on [ROUGE score](https://aclanthology.org/W04-1013/), [BERTscore](https://arxiv.org/pdf/1904.09675.pdf), and by the evidence-inference-based divergence metric defined in the [MS^2 paper](https://arxiv.org/pdf/2104.06486.pdf). The evaluation script is available at `evaluator/evaluator.py`. 

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

Evaluating generated text is notoriously challenging. To facilitate the development of better automated summarization evaluation metrics for this task, we may perform human evaluation on some of the generated summaries submitted to the MSLR leaderboards. We aim to share these human evaluation results with the public when they are complete.

## Leaderboard

Once you are ready to submit, you can find the task leaderboards on the [AI2 Leaderboard Page](https://leaderboard.allenai.org/):

**MS^2 Subtask**: [https://leaderboard.allenai.org/mslr-ms2/submissions/public](https://leaderboard.allenai.org/mslr-ms2/submissions/public)

**Cochrane Subtask**: [https://leaderboard.allenai.org/mslr-cochrane/submissions/public](https://leaderboard.allenai.org/mslr-cochrane/submissions/public)

You will need to create an account to submit results. Before submitting, please confirm that you are submitting to the correct subtask! Evaluation may take several hours (especially for the MS^2 dataset). If evaluation completes successfully, you can return to the leaderboard page to publish your results. 

If evaluation fails, you will receive an error. The same evaluation script as above is used in the leaderboard so please try to debug first by following the instructions in [the evaluation section](https://github.com/allenai/mslr-shared-task#evaluation). If you are able to get results with the evaluation script but not in the leaderboard, please contact [lucyw@allenai.org](mailto:lucyw@allenai.org).

## MSLR 2022 Accepted Papers

[**Overview of MSLR2022: A Shared Task on Multi-document Summarization for Literature Reviews**](https://aclanthology.org/2022.sdp-1.20/)  
Lucy Lu Wang, Jay DeYoung, Byron Wallace

[**LED down the rabbit hole: exploring the potential of global attention for biomedical multi-document summarisation**](https://aclanthology.org/2022.sdp-1.21/)  
Yulia Otmakhova, Thinh Hung Truong, Timothy Baldwin, Trevor Cohn, Karin Verspoor, Jey Han Lau

[**Evaluating Pre-Trained Language Models on Multi-Document Summarization for Literature Reviews**](https://aclanthology.org/2022.sdp-1.22/)  
Benjamin Yu

[**Exploring the limits of a base BART for multi-document summarization in the medical domain**](https://aclanthology.org/2022.sdp-1.23/)  
Ishmael Obonyo, Silvia Casola, Horacio Saggion

[**Abstractive Approaches To Multidocument Summarization Of Medical Literature Reviews**](https://aclanthology.org/2022.sdp-1.24/)  
Rahul Tangsali, Aditya Jagdish Vyawahare, Aditya Vyankatesh Mandke, Onkar Rupesh Litake, Dipali Dattatray Kadam

[**An Extractive-Abstractive Approach for Multi-document Summarization of Scientific Articles for Literature Review**](https://aclanthology.org/2022.sdp-1.25/)  
Kartik Shinde, Trinita Roy, Tirthankar Ghosal

## Contact Us

You can reach the organizers by emailing mslr-organizers@googlegroups.com 

To receive updates , please join our mailing list:   https://groups.google.com/g/mslr-info or email lucyw@allenai.org to be added to our Slack workspace.

## Organizing Team

* [Lucy Lu Wang](https://llwang.net/), University of Washington and Allen Institute for AI (AI2)
* [Jay DeYoung](https://www.khoury.northeastern.edu/people/jay-deyoung/), Northeastern University
* [Byron Wallace](https://www.byronwallace.com/), Northeastern University

## References

[**MS2: A Dataset for Multi-Document Summarization of Medical Studies**](https://aclanthology.org/2021.emnlp-main.594/)  
Jay Deyoung, Iz Beltagy, Madeleine van Zuylen, Bailey Kuehl, Lucy Lu Wang

```bibtex
@inproceedings{DeYoung2021MS2MS,
  title={MSˆ2: Multi-Document Summarization of Medical Studies},
  author={Jay DeYoung and Iz Beltagy and Madeleine van Zuylen and Bailey Kuehl and Lucy Lu Wang},
  booktitle={EMNLP},
  year={2021}
}
```

[**Generating (factual?) narrative summaries of RCTs: Experiments with neural multi-document summarization**](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8378607/)  
Byron C. Wallace, Sayantani Saha, Frank Soboczenski, Iain James Marshall

```bibtex
@article{Wallace2020GeneratingN,
  title={Generating (Factual?) Narrative Summaries of RCTs: Experiments with Neural Multi-Document Summarization},
  author={Byron C. Wallace and Sayantani Saha and Frank Soboczenski and Iain James Marshall},
  journal={AMIA Annual Symposium},
  year={2020},
  volume={abs/2008.11293}
}
```

