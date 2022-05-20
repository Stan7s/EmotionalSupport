# EmotionalSupport

## Fine-tuning DialoGPT

### Environment

Setup:
```
git clone https://github.com/microsoft/DialoGPT.git
cd DialoGPT
conda env create -f LSP-linux.yml -n LSP
conda activate LSP

# For evaluation
pip install git+https://github.com/Maluuba/nlg-eval.git@master
nlg-eval --setup
```

### Model
- DialoGPT model (medium): Loacated under `model/medium/` (need to copy `pytorch_model.bin` from BOX[1])
- Output models: will be located under `model/output_model/`

### Data
- Located under `data/`
- Dummy data for test: `data/dummy/`
- Full training data: `data/0518/` (need to copy from BOX[2])
  - full_0_-3
  - full_5_-3
  - full_10_-3
  - full_-5_-3
  - full_-10_-3

### RUN
```
conda activate LSP
bash scripts/demo_train.sh # Dummy data

bash scripts/0518/{SCRIPT_NAME}.sh # Full data
```

- [1] https://notredame.app.box.com/folder/163443565826
- [2] https://notredame.app.box.com/folder/163290643660
