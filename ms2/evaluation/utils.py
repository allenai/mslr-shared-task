"""
Adapted from https://github.com/allenai/ms2
"""

import argparse
import itertools
import json
import math
import os
import re

import numpy as np
import torch

from sklearn.metrics import classification_report, confusion_matrix
from scipy.spatial.distance import jensenshannon
from ms2.models.evidence_inference_models import initialize_models
from ms2.models.utils import rouge_scores
from ms2.utils import get_tokenizer

from ms2.utils import (
    get_tokenizer,
    EXTRA_TOKENS,
    SEP_TOKEN,
    START_INTERVENTION,
    END_INTERVENTION,
    START_OUTCOME,
    END_OUTCOME,
)

TOKENS_SPLIT = '|'.join(EXTRA_TOKENS)

INTERVENTION_RE = START_INTERVENTION + '(.*?)' + END_INTERVENTION
OUTCOME_RE = START_OUTCOME + '(.*?)' + END_OUTCOME


def ios(preamble):
    # we know that the reference abstract is already space tokenized
    start_stop_words = EXTRA_TOKENS + ['<s>', '</s>']

    def clean_str(s):
        for w in start_stop_words:
            s = s.replace(w, '')
        return s

    outcomes = list(map(clean_str, re.findall(OUTCOME_RE, preamble)))
    interventions = list(map(clean_str, re.findall(INTERVENTION_RE, preamble)))
    return interventions, outcomes


def evidence_inference_score(model, evidence_inference_tokenizer, summary, preamble, use_ios):
    ret = []
    if use_ios:
        interventions, outcomes = ios(preamble)
        summary = evidence_inference_tokenizer(summary, return_tensors='pt')['input_ids']
        for i, o in itertools.product(interventions, outcomes):
            preamble = i + ' ' + evidence_inference_tokenizer.sep_token + ' ' + o
            ico = evidence_inference_tokenizer(preamble, return_tensors='pt')['input_ids']
            classes = model(ico, summary)
            classes = torch.softmax(classes, dim=-1).detach().cpu().squeeze().tolist()
            significance_distribution = dict(
                zip(["significantly decreased", "no significant difference", "significantly increased"], classes))
            ret.append(significance_distribution)
    else:
        preamble = ""
        ico = evidence_inference_tokenizer(preamble, return_tensors='pt')['input_ids']
        summary = evidence_inference_tokenizer(summary, return_tensors='pt')['input_ids']
        classes = model(ico, summary)
        classes = torch.softmax(classes, dim=-1).detach().cpu().squeeze().tolist()
        significance_distribution = dict(
            zip(["significantly decreased", "no significant difference", "significantly increased"], classes))
        ret.append(significance_distribution)

    return ret


def jsd(m1, m2):
    keys = list(set(m1.keys()) | set(m2.keys()))
    m1 = [m1.get(k, 0) for k in keys]
    m2 = [m2.get(k, 0) for k in keys]
    return jensenshannon(m1, m2, base=2)


def entailment_score(model, evidence_inference_tokenizer, generated, target, preamble, use_ios):
    generated_distributions = evidence_inference_score(model, evidence_inference_tokenizer, generated, preamble,
                                                       use_ios)
    summary_distributions = evidence_inference_score(model, evidence_inference_tokenizer, target, preamble, use_ios)
    jsds = []
    for generated_distribution, summary_distribution in zip(generated_distributions, summary_distributions):
        jsds.append(jsd(generated_distribution, summary_distribution))
    if len(jsds) == 0:
        return None
    return np.mean(jsds)


def f1_score(model, evidence_inference_tokenizer, generateds, targets, preambles, use_ios):
    summary_preds = []
    generated_preds = []
    in_doc_classifications = []
    labels = ["significantly decreased", "no significant difference", "significantly increased"]
    mapping = {x: i for (i, x) in enumerate(labels)}
    for generated, target, preamble in zip(generateds, targets, preambles):
        generated_distributions = evidence_inference_score(model, evidence_inference_tokenizer, generated, preamble,
                                                           use_ios)
        summary_distributions = evidence_inference_score(model, evidence_inference_tokenizer, target, preamble, use_ios)
        in_doc_generated = []
        in_doc_target = []
        for generated_distribution, summary_distribution in zip(generated_distributions, summary_distributions):
            generated_targets = sorted(generated_distribution.items(), key=lambda x: x[1])
            summary_targets = sorted(summary_distribution.items(), key=lambda x: x[1])
            best_summary_target = summary_targets[-1][0]
            in_doc_target.append(best_summary_target)
            summary_preds.append(best_summary_target)
            generated_target = generated_targets[-1][0]
            generated_preds.append(generated_target)
            in_doc_generated.append(generated_target)
        if len(in_doc_generated) == 0:
            continue
        in_doc_classifications.append(
            classification_report(
                np.array([mapping[x] for x in in_doc_target]),
                np.array([mapping[x] for x in in_doc_generated]),
                target_names=labels,
                labels=list(range(len(labels))),
                output_dict=True,
                digits=4
            )
        )
    res = classification_report(np.array([mapping[x] for x in summary_preds]),
                                np.array([mapping[x] for x in generated_preds]), target_names=labels, output_dict=True,
                                labels=list(range(len(labels))), digits=4)
    return res


def jsd_uniform(model, evidence_inference_tokenizer, target, preamble, use_ios):
    summary_distributions = evidence_inference_score(model, evidence_inference_tokenizer, target, preamble, use_ios)
    jsds = []
    # baseline distributions
    generated_distribution = {
        'significantly decreased': .134,
        'no significant difference': .570,
        'significantly increased': .296,
    }
    for summary_distribution in summary_distributions:
        jsds.append(jsd(generated_distribution, summary_distribution))
    if len(jsds) == 0:
        return None
    return np.mean(jsds)


def entailment_scores(model, evidence_inference_tokenizer, generateds, targets, preambles, use_ios):
    f1_scores = f1_score(model, evidence_inference_tokenizer, generateds, targets, preambles, use_ios)
    scores = list(map(lambda x: entailment_score(model, evidence_inference_tokenizer, *x, use_ios),
                      zip(generateds, targets, preambles)))
    scores = list(filter(lambda x: x is not None, scores))
    uniform_scores = list(
        map(lambda x: jsd_uniform(model, evidence_inference_tokenizer, *x, use_ios), zip(targets, preambles)))
    uniform_scores = list(filter(lambda x: x is not None, uniform_scores))
    assert len(scores) > 0
    avg = np.mean(scores)
    s = np.std(scores)
    uniform_score = np.mean(uniform_scores)
    return {
        'average': avg,
        'std': s,
        'uniform_preds': uniform_score,
        'f1_score': f1_scores,
    }


def clean(s):
    for t in EXTRA_TOKENS + ['<s>', '</s>']:
        s = s.replace(t, '')
        s = s.replace('  ', ' ')
    return s