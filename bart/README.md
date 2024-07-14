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
- `.source` as input, `.target` as ground truth.
- Put all files under `data/{dataset_name}/` (same as `--data_dir` in .sh)

### RUN
```
conda activate bart
bash run_bart_parallel.sh
```
