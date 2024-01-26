import re

import json
from nltk import ngrams
from nltk.translate.bleu_score import sentence_bleu

# year_pattern = re.compile(r'\(\d{4}\)')
slot_pattern = re.compile(r'<movie>')

from rouge import Rouge

def _cal_rouge(hypothesis,reference):
    """
    both hypothesis and reference are str
    """
    if hypothesis == '':
        return 0,0,0
    rouge = Rouge()
    try:
        scores = rouge.get_scores(hypothesis, reference)
        return scores[0]['rouge-1']['f'],scores[0]['rouge-2']['f'],scores[0]['rouge-l']['f']
    except:
        print("something wrong here! ",hypothesis,reference)
        return 0,0,0


class ConvEvaluator:
    def __init__(self, tokenizer, log_file_path):
        self.tokenizer = tokenizer

        self.reset_metric()
        if log_file_path:
            self.log_file = open(log_file_path, 'w', buffering=1)
            self.log_cnt = 0

    def evaluate(self, preds, labels, log=False):

        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_preds = [decoded_pred.replace('<pad>', '').replace('<|endoftext|>', '') for decoded_pred in
                         decoded_preds]       
        
        # print(decoded_preds[0])
        # assert 1==0

        decoded_preds = ["System: " + pred.strip() for pred in decoded_preds]
        #### for a fair comparision , we prepend the System token before the generated sequences.
        decoded_preds = [pred for pred in decoded_preds]

        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=False)
        decoded_labels = [decoded_label.replace('<pad>', '').replace('<|endoftext|>', '') for decoded_label in
                          decoded_labels]
        decoded_labels = [label.strip() for label in decoded_labels]
        if log and hasattr(self, 'log_file'):
            for pred, label in zip(decoded_preds, decoded_labels):
                self.log_file.write(json.dumps({
                    'pred': pred,
                    'label': label
                }, ensure_ascii=False) + '\n')

        self.collect_ngram(decoded_preds)
        self.compute_item_ratio(decoded_preds)
        self.compute_bleu(decoded_preds, decoded_labels)
        self.compute_rouge(decoded_preds, decoded_labels)
        self.sent_cnt += len([pred for pred in decoded_preds if len(pred) > 0])

    def collect_ngram(self, strs):
        for str in strs:
            str = str.split()
            for k in range(1, 5):
                dist_k = f'dist@{k}'
                for token in ngrams(str, k):
                    self.metric[dist_k].add(token)

    def compute_bleu(self, preds, labels):
        for pred, label in zip(preds, labels):
            pred, label = pred.split(), [label.split()]
            for k in range(4):
                weights = [0] * 4
                weights[k] = 1
                self.metric[f'bleu@{k + 1}'] += sentence_bleu(label, pred, weights)
        

    def compute_rouge(self, preds, labels):
        
        for pred, label in zip(preds, labels):
            rouge_1, rouge_2, rouge_l = _cal_rouge(pred, label)
            self.metric["rouge@1"] += rouge_1
            self.metric["rouge@2"] += rouge_2
            self.metric["rouge@L"] += rouge_l

    def compute_item_ratio(self, strs):
        for str in strs:
            # items = re.findall(year_pattern, str)
            # self.metric['item_ratio'] += len(items)
            items = re.findall(slot_pattern, str)
            self.metric['item_ratio'] += len(items)

    def report(self):
        report = {}
        for k, v in self.metric.items():
            if self.sent_cnt == 0:
                report[k] = 0
            else:
                if 'dist' in k:
                    v = len(v)
                report[k] = v / self.sent_cnt
        report['sent_cnt'] = self.sent_cnt
        return report

    def reset_metric(self):
        self.metric = {
            'bleu@1': 0,
            'bleu@2': 0,
            'bleu@3': 0,
            'bleu@4': 0,
            'rouge@1': 0,
            'rouge@2': 0,
            'rouge@L': 0,
            'dist@1': set(),
            'dist@2': set(),
            'dist@3': set(),
            'dist@4': set(),
            'item_ratio': 0,
        }
        self.sent_cnt = 0
