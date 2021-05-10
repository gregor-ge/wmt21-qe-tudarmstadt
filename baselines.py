import argparse
import json
import os
from datetime import datetime

import torch
import yaml
from torch.utils.data import DataLoader, Sampler
from transformers import XLMRobertaModelWithHeads, XLMRobertaConfig, XLMRobertaTokenizer, TrainingArguments, Trainer, \
    EvalPrediction, TrainerCallback, AutoConfig, AutoTokenizer, AutoModelWithHeads, AdapterConfig, AutoModelForSequenceClassification
import numpy as np
from scipy.stats import pearsonr
from datasets import load_dataset, Dataset, DatasetDict
import logging


logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

def main(config):
    os.environ["WANDB_WATCH"] = "false"
    if config.get("wandb_project", ""):
        os.environ["WANDB_PROJECT"] = config["wandb_project"]
    if config.get("do_train", True):
        train(config)
    else:
        test(config)

def train(config):
    logger.info(config)
    task_folder = f"train_{config.get('task_name', '')}{config['task']}_{datetime.now().strftime('%Y-%m-%d_%H-%M')}"
    output_dir = os.path.join(config["output_dir"], task_folder)
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Saving results in {output_dir}")
    yaml.dump(config, open(os.path.join(output_dir, "train_config.yaml"), "w"))

    model_config = AutoConfig.from_pretrained(config.get("model", "bert-base-multilingual-cased"), num_labels=1)
    model = AutoModelForSequenceClassification.from_pretrained(config.get('model', "bert-base-multilingual-cased"), config=model_config)

    task = config["task"]

    train_config = config["train"]
    is_multipair = not isinstance(train_config["pair"][0], str)
    logger.info(f"Training for {task} {train_config['pair']}")
    pairs = train_config["pair"] if is_multipair else [train_config["pair"]]

    dataset = load_data(train_config["pair"], task, config)

    training_args = TrainingArguments(
        learning_rate=train_config.get("learning_rate", 5e-5),
        num_train_epochs=train_config["epochs"],
        max_steps=train_config.get("max_steps", -1),
        per_device_train_batch_size=train_config.get("train_batchsize", 16),
        per_device_eval_batch_size=train_config.get("eval_batchsize", 32),
        logging_steps=train_config.get("logging_steps", 10),
        output_dir=output_dir,
        overwrite_output_dir=True,
        gradient_accumulation_steps=train_config.get("gradient_accumulation_steps", 1),
        fp16=train_config.get("amp", True),
        eval_steps=train_config.get("eval_steps", 250),
        evaluation_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="pearson",
        save_total_limit=train_config.get("save_total_limit", None),
        run_name=task_folder,
        report_to=config.get("report_to", "all"),
        skip_memory_metrics=config.get("skip_memory_metrics", True)
    )

    if is_multipair:
        trainer = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["dev"],
            compute_metrics=compute_pearson,
            do_save_adapters=True,
            do_save_full_model=False
        )
    else:
        train_lang1, train_lang2 = pairs[0]
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["dev"],
            compute_metrics=compute_pearson,
            do_save_adapters=True,
            do_save_full_model=False
        )
    trainer.train()
    best_checkpoint = os.path.join(output_dir, "best_checkpoint", task)
    os.makedirs(best_checkpoint, exist_ok=True)
    test(config, model, task_folder)

def test(config, model=None, task_folder=None):
    task = config["task"]
    if not model:
        logger.info(f"Loading task adapter from {config['adapter_path']}")
        model_config = AutoConfig.from_pretrained(config.get("model", "bert-base-multilingual-cased"), num_labels=1)
        model = AutoModelForSequenceClassification.from_pretrained(config.get('model', "bert-base-multilingual-cased"),
                                                                   config=model_config)
    if not task_folder:
        task_folder = f"test_{config.get('task_name', '')}_{config['task']}{datetime.now().strftime('%Y-%m-%d_%H-%M')}"
    output_dir = os.path.join(config["output_dir"], task_folder)
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Saving results in {output_dir}")
    yaml.dump(config, open(os.path.join(output_dir, "test_config.yaml"), "w"))
    results = {"dev": [], "test": [], "task": task}
    for pair in config["test"]["pairs"]:
        lang1, lang2 = pair
        dataset = load_data(pair, task, config)
        logger.info(f"Evaluation results for {task} {lang1}-{lang2}")

        dev_trainer = Trainer(
            model=model,
            args=TrainingArguments(output_dir=output_dir,
                                   remove_unused_columns=False,
                                   per_device_eval_batch_size=config["test"]["batchsize"],
                                   run_name=task_folder,
                                   report_to=config.get("report_to", "all"),
                                   skip_memory_metrics=config.get("skip_memory_metrics", True)),
            eval_dataset=dataset["dev"],
            compute_metrics=compute_pearson
        )
        dev_evaluation = dev_trainer.evaluate(metric_key_prefix="dev")
        dev_evaluation["pair"] = f"{lang1}_{lang2}"
        results["dev"].append(dev_evaluation)
        test_trainer = Trainer(
            model=model,
            args=TrainingArguments(output_dir=output_dir,
                                   remove_unused_columns=False,
                                   per_device_eval_batch_size=config["test"]["batchsize"],
                                   run_name=task_folder,
                                   report_to=config.get("report_to", "all"),
                                   skip_memory_metrics=config.get("skip_memory_metrics", True)),
            eval_dataset=dataset["test"],
            compute_metrics=compute_pearson
        )
        test_evaluation = test_trainer.evaluate(metric_key_prefix="test")
        test_evaluation["pair"] = f"{lang1}_{lang2}"
        results["test"].append(test_evaluation)
    logger.info(results)
    json.dump(results, open(os.path.join(output_dir, f"evaluation_{task}.json"), "w"), indent=2)

def load_data(lang_pairs, task, config):
    if isinstance(lang_pairs[0], str):
        lang_pairs = [lang_pairs]
    tokenizer = AutoTokenizer.from_pretrained(config.get("model", "bert-base-multilingual-cased"))

    if task == "qe_da":
        dataset = load_dataset("csv", delimiter="\t", quoting=3, data_files={
            "train": [f"data/data/direct-assessments/train/{lang1}-{lang2}-train/train.{lang1}{lang2}.df.short.tsv" for (lang1, lang2) in lang_pairs],
            "test": [f"data/data/direct-assessments/test/{lang1}-{lang2}/test20.{lang1}{lang2}.df.short.tsv" for (lang1, lang2) in lang_pairs],
            "dev": [f"data/data/direct-assessments/dev/{lang1}-{lang2}-dev/dev.{lang1}{lang2}.df.short.tsv" for (lang1, lang2) in lang_pairs]
        })
        # The transformers model expects the target class column to be named "labels"
        dataset = dataset.rename_column("z_mean", "label")

    def encode_batch(batch):
        """Encodes a batch of input data using the model tokenizer."""
        original = batch["original"]
        translation = batch["translation"]
        if "prompt" in config:
            prompt_orig, prompt_transl = config["prompt"]
            original = [f"{prompt_orig}: {o}" for o in original]
            translation = [f"{prompt_transl}: {t}" for t in translation]

        sen1 = tokenizer(original, max_length=config.get("max_seq_len", 50), truncation=True, padding="max_length")
        sen2 = tokenizer(translation, max_length=config.get("max_seq_len", 50), truncation=True, padding="max_length")

        for i1, i2 in zip(sen1["input_ids"], sen2["input_ids"]):
            i1.extend(i2[1:])

        for a1, a2 in zip(sen1["attention_mask"], sen2["attention_mask"]):
            a1.extend(a2[1:])

        for t1, t2 in zip(sen1["token_type_ids"], sen2["token_type_ids"]):
            t1.extend([1]*(len(t2)-1))

        return sen1
    # Encode the input data
    dataset = dataset.map(encode_batch, batched=True)
    # Transform to pytorch tensors and only output the required columns

    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "token_type_ids", "label"])
    return dataset

def compute_pearson(p: EvalPrediction):
    if len(p.predictions) == 1000:  # hard coded
        return {"pearson": pearsonr(p.predictions.reshape(-1), p.label_ids)[0]}
    else:
        predictions = p.predictions.reshape(-1)
        all_r = []
        for i in range(0, len(predictions), 1000):
            all_r.append(pearsonr(predictions[i:i + 1000], p.label_ids[i:i + 1000])[0])
        return {"pearson": np.mean(all_r)}

class CustomTrainer(Trainer):
    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        train_sampler = MultiSampler(self.train_dataset, self.args.train_batch_size)
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=train_sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

class MultiSampler(Sampler):
    def __init__(self, data_source, batchsize):
        self.data_source = data_source
        self.batchsize = batchsize
        self.max_rounds = len(self.data_source)//7000
        self.max_get = self.batchsize*(7000//self.batchsize)

    def __iter__(self):
        permutation = np.random.permutation(range(7000))
        idx = 0
        round = 0
        while idx < self.max_get:
            for i in range(self.batchsize):
                yield permutation[idx+i].item() + round*7000
            round += 1
            if round == self.max_rounds:
                idx += self.batchsize
                round = 0

    def __len__(self) -> int:
        return self.max_rounds * self.max_get

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    config = yaml.load(open(args.config), Loader=yaml.FullLoader)
    main(config)