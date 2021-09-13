# WMT21 Quality Estimation Entry

Entry of students of the TU Darmstadt for the [WMT21 quality estimation shared task (task 1 and 2)](http://statmt.org/wmt21/quality-estimation-task.html).

Language adapters and tokenizers for mulitlingual BERT & Sinhala/ Khmer provided by Jonas Pfeiffer.
Please cite [UNKs Everywhere: Adapting Multilingual Language Models to New Scripts](https://arxiv.org/abs/2012.15562) 
and [MAD-X: An Adapter-Based Framework for Multi-Task Cross-Lingual Transfer](https://arxiv.org/abs/2005.00052) if you use them.

## Installation
Since our implementation relies on the use of adapters, the framework must be present on the system. It can be installed by running
```
pip install adapter-transformers
```
Further information on installing the adaper framework can be found on https://adapterhub.ml/.


## Use Fine-Tuned Adapters
We release our trained adapters for XLM-R (base & large) and mBERT (trained with language adapters + additional embeddings) on [AdapterHub](https://adapterhub.ml/explore/quality_estimation/wmt21/)
and on [Hugging Face Hub](https://huggingface.co/models?other=adapterhub:quality_estimation/wmt21).


## Usage
Our model can be run by passing a config.yaml file containing all hyperparameters to the run.py's main method.
```
python run.py configs/config.yaml
```

An exemplary config.yaml can be found below:
```
do_train: True
model: bert-base-multilingual-cased
output_dir: results/qe-da/testing
max_seq_len: 50
task: qe_da
report_to: none
madx2: True
architecture: base
reduction_factor: 8
dropout: 0.1
no_lang: False
predict: False
debug: False
boosting: False
train:
  train_batchsize: 8 # should evenly divide 7000 for multi pair training to work with the current hack
  eval_batchsize: 50 # must evenly divide 1000 for multi pair training to work with the current hack
  max_steps: 6000
  logging_steps: 10
  eval_steps: 250
  gradient_accumulation_steps: 1
  save_total_limit: 2
  amp: True
  epochs: 1
  pair: # list of pairs or just a pair
    - [en, de]
    - [en, zh]
    - [et, en]
    - [ne, en]
    - [ro, en]
    - [ru, en]
    - [si, en]
test:
  batchsize: 32
  pairs:
    - [en, de]
    - [en, zh]
    - [et, en]
    - [ne, en]
    - [ro, en]
    - [ru, en]
    - [si, en]

```

