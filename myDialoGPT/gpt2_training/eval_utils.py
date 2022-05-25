#  Copyright (c) Microsoft Corporation. 
#  Licensed under the MIT license. 
import torch
import logging

import numpy as np

from pycocoevalcap.bleu.bleu import Bleu
from collections import defaultdict

import json
from os.path import abspath, dirname, exists, join
import argparse
import logging
from tqdm import trange
import tqdm
import torch
import torch.nn.functional as F
import numpy as np
import socket
import os, sys
import re
import logging
from functools import partial
from demo_utils import download_model_folder
import argparse
import subprocess as sp
import pandas as pd

from pytorch_pretrained_bert import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from gpt2_training.train_utils import get_eval_list_same_length, load_model, boolean_string, fix_state_dict_namespace

from nlgeval import NLGEval

logger = logging.getLogger(__name__)

EOS_ID = 50256


def cal_BLEU_4(generated, reference, is_corpus=False):
    BLEUscore = [0.0, 0.0, 0.0, 0.0]
    for idx, g in enumerate(generated):
        if is_corpus:
            score, scores = Bleu(4).compute_score(reference, {0: [g]})
        else:
            score, scores = Bleu(4).compute_score({0: [reference[0][idx]]},
                                                  {0: [g]})
        for i, s in zip([0, 1, 2, 3], score):
            BLEUscore[i] += s
    BLEUscore[0] = BLEUscore[0]/len(generated)
    BLEUscore[1] = BLEUscore[1]/len(generated)
    BLEUscore[2] = BLEUscore[2]/len(generated)
    BLEUscore[3] = BLEUscore[3]/len(generated)
    return BLEUscore


def cal_entropy(generated):
    etp_score = [0.0, 0.0, 0.0, 0.0]
    div_score = [0.0, 0.0, 0.0, 0.0]
    counter = [defaultdict(int), defaultdict(int),
               defaultdict(int), defaultdict(int)]
    for gg in generated:
        g = gg.rstrip().split()
        for n in range(4):
            for idx in range(len(g)-n):
                ngram = ' '.join(g[idx:idx+n+1])
                counter[n][ngram] += 1
    for n in range(4):
        total = sum(counter[n].values()) + 1e-10
        for v in counter[n].values():
            etp_score[n] += - (v+0.0) / total * (np.log(v+0.0) - np.log(total))
        div_score[n] = (len(counter[n].values())+0.0) / total
    return etp_score, div_score


def eval_model_loss(model, tokenizer, eval_dataloader, epoch_id, args):
    # use the same signature with eval_model_generation
    logger.info('compute eval model loss, using eval mode, '
                'please change it back to train after calling this function')
    model.eval()
    tot_loss = []
    tot_ppl = []
    tot_sample = []
    with torch.no_grad():
        for step, batch in enumerate(eval_dataloader):
            batch = tuple(t.to(args.device) for t in batch)
            input_ids, position_ids, token_ids, label_ids, src_len, _ = batch
            if args.no_token_id:
                token_ids = None
            n_sample = input_ids.shape[0]
            loss, ppl = model(input_ids, position_ids, token_ids, label_ids)
            tot_loss.append(loss.mean().item() * n_sample)
            tot_ppl.append(ppl.mean().item() * n_sample)
            tot_sample.append(n_sample)
    print(f"\n Epoch {epoch_id}: Val loss {np.sum(tot_loss) / np.sum(tot_sample)} Val ppl {np.sum(tot_ppl) / np.sum(tot_sample)} ")
    return np.sum(tot_loss) / np.sum(tot_sample), np.sum(tot_ppl) / np.sum(tot_sample)


def cut_seq_to_eos(sentence, remove_id=[-1]):
    sent=[]
    for s in sentence:
        if s in remove_id:
            continue
        if s != EOS_ID:
            sent.append(s)
        else:
            break
    return sent


### FROM HUGGING FACE REPO
def top_filtering(logits, top_k=0, top_p=0.0, threshold=-float('Inf'), filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k, top-p (nucleus) and/or threshold filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k: <=0: no filtering, >0: keep only top k tokens with highest probability.
            top_p: <=0.0: no filtering, >0.0: keep only a subset S of candidates, where S is the smallest subset
                whose total probability mass is greater than or equal to the threshold top_p.
                In practice, we select the highest probability tokens whose cumulative probability mass exceeds
                the threshold top_p.
            threshold: a minimal threshold to keep logits
    """
    assert logits.dim() == 1  # Only work for batch size 1 for now - could update but it would obfuscate a bit the code
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        # Remove all tokens with a probability less than the last token in the top-k tokens
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        # Compute cumulative probabilities of sorted tokens
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probabilities = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probabilities > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Back to unsorted indices and set them to -infinity
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value

    indices_to_remove = logits < threshold
    logits[indices_to_remove] = filter_value
    return logits

def generate_next_token(model, input_ids, position_ids=None, token_type_ids=None, prev=None, temperature=1, top_k=0, top_p=0, past=None):
    with torch.no_grad():
        if not past:
            hidden_states, past = model.transformer(prev, position_ids, token_type_ids, past=past)
        else:
            hidden_states, past = model.transformer(prev, past=past)
        logits = model.lm_head(hidden_states)
        logits = logits[0, -1, :] / temperature
        logits = top_filtering(logits, top_k=top_k, top_p=top_p)
        probs = F.softmax(logits.unsqueeze(0), dim=-1)
        prev = torch.multinomial(probs, num_samples=1)
        return prev, probs[0][prev], past

def generate_sequence(model, input_ids, position_ids=None, token_type_ids=None, temperature=1, top_k=0, top_p=0, length=20, past=None, device='cuda'):
    output = input_ids.new_zeros([input_ids.size(0),0])
    prev = input_ids
    for i in range(length):
        prev, probs, past = generate_next_token(model, input_ids, position_ids, token_type_ids, prev, temperature, top_k, top_p, past)
        output = torch.cat((output, prev), dim=1)
    return output

def cut_seq_to_eos(sentence, remove_id=[-1]):
    sent=[]
    for s in sentence:
        if s in remove_id:
            continue
        if s != EOS_ID:
            sent.append(s)
        else:
            break
    return sent

def predict(model, enc, args, pred_filename):
    logger.info('predicting...')
    model.eval()
    df = pd.read_csv(args.pred_input_file, header = None, sep = '\t')
    source_list = df.iloc[:, 0].to_list()
    target_list = df.iloc[:, 1].to_list()
    print(source_list[0:5])
    print(target_list[0:5])

    pred_target_list = []

    for id, source in enumerate(source_list):
        print(id)
        context_tokens = enc.encode(source) + [EOS_ID]
        context_tokens = torch.tensor(context_tokens, device=args.device, dtype=torch.long).unsqueeze(0)
        position_ids = torch.arange(0, context_tokens.size(-1), dtype=torch.long, device=context_tokens.device)

        out = generate_sequence(model, context_tokens, position_ids=position_ids,
                                length=args.max_seq_length, temperature=args.pred_temperature, 
                                top_k=args.top_k, top_p= args.top_p) 

        out = out.tolist()                     
        pred_target = enc.decode(cut_seq_to_eos(out[0])).encode('ascii','ignore').decode('ascii')
        pred_target_list.append(pred_target)
    
    df['pred_target'] = pred_target_list
    df.to_csv(pred_filename + '.pred.tsv', sep = '\t', index = False)
    try:
        n = NLGEval(metrics_to_omit=['SkipThoughtCS', 'EmbeddingAverageCosineSimilarity', 'VectorExtremaCosineSimilarity','GreedyMatchingScore'])
        metrics_dict = n.compute_metrics([target_list], pred_target_list)

<<<<<<< HEAD
    n = NLGEval(metrics_to_omit=['METEOR', 'CIDEr', 'SkipThoughtCS', 'EmbeddingAverageCosineSimilarity', 'VectorExtremaCosineSimilarity','GreedyMatchingScore'])
    metrics_dict = n.compute_metrics([target_list], pred_target_list)
=======
        with open(pred_filename + '.nlgeval.txt', 'w') as f: 
            for key, value in metrics_dict.items(): 
                f.write('%s:%s\n' % (key, value))
    except:
        pass
>>>>>>> 359932b72f11c576c5b335593d4a2498e87d4a2d



