from nlgeval import compute_metrics
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='', help='.tsv result file path for evaluation')
args = parser.parse_args()
print(args.data)
assert args.data.endswith('.tsv')

df = pd.read_csv(args.data, sep = '\t')
true = df['1'].to_list()
pred = df['pred_target'].to_list()

ref_file = args.data[:-4] + '.ref.txt'
hyp_file = args.data[:-4] + '.hyp.txt'

with open(ref_file, 'w') as fw:
    fw.write('\n'.join(true))
with open(hyp_file, 'w') as fw:
    fw.write('\n'.join(pred))

# references = [['I am a cat.']]
# hypothesis = ['I am a dog.']
metrics_dict = compute_metrics(hypothesis = hyp_file, references = [ref_file])

with open(args.data[:-4] + '.nlgeval.txt', 'w') as f: 
    for key, value in metrics_dict.items(): 
        f.write('%s:%s\n' % (key, value))