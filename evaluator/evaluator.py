#!/usr/bin/env python3

import os
import sys
import json
import csv
import logging
import argparse
from typing import *
import torch
import bert_score
import warnings

from ms2.models.utils import rouge_scores
from ms2.models.evidence_inference_models import initialize_models
from ms2.evaluation.utils import clean, entailment_scores
from ms2.utils import (
    get_tokenizer,
    Review,
    Study,
    Reference,
    TargetReference,
    TargetSummary,
    EXTRA_TOKENS,
    START_BACKGROUND,
    END_BACKGROUND,
    START_REFERENCE,
    END_REFERENCE,
    SEP_TOKEN,
)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def read_targets(target_file: str) -> Dict[str, Dict]:
    logging.info(f"Loading targets...")
    assert os.path.exists(target_file)
    target_dict = dict()
    with open(target_file, 'r') as f:
        reader = csv.DictReader(f, delimiter=',', quotechar='"')
        for entry in reader:
            target_dict[entry["ReviewID"]] = {
                "target": entry["Target"],
                "preface": entry.get("Background", "")
            }
    if len(target_dict) == 0:
        logging.error(f"No summaries found in file {target_file}")
        sys.exit(-1)
    logging.info(f"{len(target_dict)} target summaries loaded.")
    return target_dict


def read_predictions(pred_file: str) -> Dict[str, str]:
    logging.info(f"Loading predictions...")
    assert os.path.exists(pred_file)
    prediction_dict = dict()
    with open(pred_file, 'r') as f:
        reader = csv.DictReader(f, delimiter=',', quotechar='"')
        for entry in reader:
            prediction_dict[entry["ReviewID"]] = entry["Generated"]
    if len(prediction_dict) == 0:
        logging.error(f"No summaries found in file {pred_file}")
        sys.exit(-1)
    logging.info(f"{len(prediction_dict)} generated summaries loaded.")
    return prediction_dict


def calculate_rouge(targets: Dict[str, Dict], generated: Dict[str, str]) -> Dict:
    """
    Calculate ROUGE scores
    :param targets:
    :param generated:
    :return:
    """
    logging.info(f"Computing ROUGE scores...")
    docids = list(targets.keys())
    target_texts = [[targets[docid]['target']] for docid in docids]
    generated_texts = [[generated.get(docid, '')] for docid in docids]

    # rouge scoring
    tokenizer = get_tokenizer('facebook/bart-base')
    rouge_results = rouge_scores(generated_texts, target_texts, tokenizer, use_aggregator=True)
    return rouge_results


def calculate_bertscore(targets: Dict[str, Dict], generated: Dict[str, str], model_type="roberta-large") -> Dict:
    """
    Calculate BERTscore
    :param targets:
    :param generated:
    :return:
    """
    logging.info(f"Computing BERTscore...")
    docids = list(targets.keys())
    target_texts = [targets[docid]['target'] for docid in docids]
    generated_texts = [generated.get(docid, '') for docid in docids]

    # BERTscore
    bs_ps, bs_rs, bs_fs = bert_score.score(generated_texts, target_texts, model_type=model_type)
    return {
        "bs_ps": bs_ps,
        "bs_rs": bs_rs,
        "bs_fs": bs_fs
    }


def calculate_evidence_inference_divergence(
        targets: Dict[str, Dict],
        generated: Dict[str, str],
        ei_param_file: str,
        ei_model_dir: str,
        ei_use_unconditional: bool = False
) -> Dict:
    """
    Calculate Evidence Inference Divergence
    :param targets:
    :param generated:
    :param ei_param_file:
    :param ei_model_dir:
    :param ei_use_unconditional:
    :return:
    """
    logging.info(f"Computing Delta Evidence Inference scores...")
    docids = list(targets.keys())
    target_texts = [targets[docid]['target'] for docid in docids]
    preface_texts = [targets[docid]['preface'] for docid in docids]
    generated_texts = [generated.get(docid, '') for docid in docids]

    generated_texts = list(map(clean, generated_texts))
    target_texts = list(map(clean, target_texts))

    # evidence inference scoring
    with open(ei_param_file, 'r') as inf:
        params = json.loads(inf.read())
    _, evidence_inference_classifier, _, _, _, evidence_inference_tokenizer = initialize_models(params)
    if ei_use_unconditional:
        classifier_file = os.path.join(ei_model_dir, 'unconditioned_evidence_classifier', 'unconditioned_evidence_classifier.pt')
    else:
        classifier_file = os.path.join(ei_model_dir, 'evidence_classifier', 'evidence_classifier.pt')
    #evidence_inference_classifier.load_state_dict(torch.load(classifier_file))
    # pooler parameters are added by default in an older transformers, so we have to ignore that those are uninitialized.
    evidence_inference_classifier.load_state_dict(torch.load(classifier_file, map_location=device), strict=False)
    if torch.cuda.is_available():
        evidence_inference_classifier.cuda()

    entailment_results = entailment_scores(
        evidence_inference_classifier, evidence_inference_tokenizer,
        generated_texts, target_texts, preface_texts,
        use_ios=not ei_use_unconditional
    )

    return entailment_results


def calculate_metrics(
        targets: Dict[str, Dict],
        generated: Dict[str, str],
        ei_param_file: str,
        ei_model_dir: str,
        ei_use_unconditional: bool = False
) -> Dict:
    """
    Calculate all metrics
    :param targets:
    :param generated:
    :param ei_param_file:
    :param ei_model_dir:
    :param ei_use_unconditional:
    :return:
    """
    metrics = dict()

    # ROUGE scores
    rouge_scores = calculate_rouge(
        targets=targets,
        generated=generated
    )

    # BERTscore
    bertscores = calculate_bertscore(
        targets=targets,
        generated=generated
    )
    bertscore_avg_p = torch.mean(bertscores['bs_ps']).item()
    bertscore_avg_r = torch.mean(bertscores['bs_rs']).item()
    bertscore_avg_f = torch.mean(bertscores['bs_fs']).item()

    bertscore_std_p = torch.std(bertscores['bs_ps']).item()
    bertscore_std_r = torch.std(bertscores['bs_rs']).item()
    bertscore_std_f = torch.std(bertscores['bs_fs']).item()

    # EI divergence
    delta_ei = calculate_evidence_inference_divergence(
        targets=targets,
        generated=generated,
        ei_param_file=ei_param_file,
        ei_model_dir=ei_model_dir,
        ei_use_unconditional=ei_use_unconditional
    )
    delta_ei_micro_avg = delta_ei['f1_score'].get('micro avg')
    delta_ei_macro_avg = delta_ei['f1_score'].get('macro avg')

    # TODO: Add other automated evaluation metrics

    metrics = {
        "rouge": rouge_scores,
        "rouge1": rouge_scores['rouge1'].mid.fmeasure,
        "rouge2": rouge_scores['rouge2'].mid.fmeasure,
        "rougeL": rouge_scores['rougeL'].mid.fmeasure,
        "rougeLsum": rouge_scores['rougeLsum'].mid.fmeasure,
        "bertscore_avg_p": bertscore_avg_p,
        "bertscore_avg_r": bertscore_avg_r,
        "bertscore_avg_f": bertscore_avg_f,
        "bertscore_std_p": bertscore_std_p,
        "bertscore_std_r": bertscore_std_r,
        "bertscore_std_f": bertscore_std_f,
        "delta_ei": delta_ei,
        "delta_ei_avg": delta_ei.get('average'),
        "delta_ei_std": delta_ei.get('std'),
        "accuracy": delta_ei['f1_score'].get('accuracy', None),
        "delta_ei_macro_p": delta_ei_macro_avg.get('precision') if delta_ei_macro_avg else None,
        "delta_ei_macro_r": delta_ei_macro_avg.get('recall') if delta_ei_macro_avg else None,
        "delta_ei_macro_f1": delta_ei_macro_avg.get('f1-score') if delta_ei_macro_avg else None,
        "delta_ei_micro_p": delta_ei_micro_avg.get('precision') if delta_ei_micro_avg else None,
        "delta_ei_micro_r": delta_ei_micro_avg.get('recall') if delta_ei_micro_avg else None,
        "delta_ei_micro_f1": delta_ei_micro_avg.get('f1-score') if delta_ei_micro_avg else None
    }

    return metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate predictions for MS2')
    parser.add_argument(
        '--targets', '-t',
        help='Filename containing target summaries and references',
        required=True)
    parser.add_argument(
        '--predictions', '-p',
        help="Filename of model generated summaries",
        required=True)
    parser.add_argument(
        '--output', '-o',
        help='Output file',
        required=True)

    # TODO: change model-based metrics to use param file
    parser.add_argument(
        '--ei_param_file',
        help='Path to Evidence Inference parameter file',
        required=True
    )
    parser.add_argument(
        '--ei_model_dir',
        help='Path to Evidence Inference models',
        required=True
    )
    parser.add_argument(
        '--ei_use_unconditional',
        help='Whether to use unconditional EI model',
        action='store_true'
    )

    args = parser.parse_args()
    targets = read_targets(args.targets)
    predictions = read_predictions(args.predictions)

    # check sufficient overlap
    overlap = set(targets.keys()) & set(predictions.keys())
    prop_overlap = len(overlap) / len(targets)
    if prop_overlap < 0.95:
        warnings.warn(f'Only {prop_overlap * 100:.1f}% of target documents have predictions in the input file.')
    if prop_overlap <= 0.5:
        logging.error(f"Proportion of target documents represented in submitted predictions is less than 0.5!")
        sys.exit(-1)

    metrics = calculate_metrics(targets, predictions, args.ei_param_file, args.ei_model_dir, args.ei_use_unconditional)

    print(metrics)

    logging.info("Writing metrics to file...")
    with open(args.output, 'w') as outf:
        json.dump(metrics, outf, indent=4)
    logging.info("done.")


if __name__ == '__main__':
    main()