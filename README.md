# EmotionalSupport

## Fine-tuning BART 

### Environment
- python 3.6
- pytorch 1.7
- transformers 3.3.1

Setup:
```
conda create -n bart python=3.6
pip install pytorch==1.7
pip install transformers==3.3.1
```

### Data
- 6 files. `.source` as input, `.target` as ground truth.
- Full data path (CRC): `/afs/crc/group/dmsquare/vol5/ltong2/EmotionalSupport/data/reddit/comments/bart_dataset/full_m`
- Toy data path (CRC): `/afs/crc/group/dmsquare/vol5/ltong2/EmotionalSupport/data/reddit/comments/bart_dataset/toy_m`
- Put all 6 files under `data/`

### RUN
```
conda activate bart
bash run_bart_parallel.sh
```
