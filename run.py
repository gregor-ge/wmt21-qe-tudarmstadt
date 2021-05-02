import argparse
import json
import os
from datetime import datetime

import yaml
import torch
from transformers import XLMRobertaModelWithHeads, XLMRobertaConfig, XLMRobertaTokenizer, TrainingArguments, Trainer, EvalPrediction
import numpy as np
from scipy.stats import pearsonr
import transformers.adapters.composition as ac
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

    model_config = XLMRobertaConfig.from_pretrained(config.get("model", "xlm-roberta-base"), num_labels=1)
    model = XLMRobertaModelWithHeads.from_pretrained(config.get("model", "xlm-roberta-base"), config=model_config)

    task = config["task"]
    assert task == "qe_da" or task == "qe_hter"
    model.add_adapter(task)
    model.add_classification_head(task, num_labels=1)

    train_config = config["train"]
    train_lang1, train_lang2 = train_config["pair"]
    logger.info(f"Training for {task} {train_lang1}-{train_lang2}")
    model.load_adapter(f"{train_lang1}/wiki@ukp")
    model.load_adapter(f"{train_lang2}/wiki@ukp")

    model.train_adapter([task])
    model.set_active_adapters([ac.Split(train_lang1, train_lang2, split_index=config.get("max_seq_len", 50)), task])

    dataset = load_data(train_lang1, train_lang2, task, config)

    training_args = TrainingArguments(
        learning_rate=train_config.get("learning_rate", 0.0001),
        num_train_epochs=train_config["epochs"],
        max_steps=train_config.get("max_steps", -1),
        per_device_train_batch_size=train_config.get("train_batchsize", 16),
        per_device_eval_batch_size=train_config.get("dev_batchsize", 32),
        logging_steps=train_config.get("logging_steps", 10),
        output_dir=output_dir,
        overwrite_output_dir=True,
        gradient_accumulation_steps=train_config.get("gradient_accumulation_steps", 1),
        fp16=train_config.get("amp", True),
        eval_steps=train_config.get("eval_steps", 250),
        evaluation_strategy="steps",
        load_best_model_at_end=True,
        save_total_limit=train_config.get("save_total_limit", None),
        run_name=task_folder,
        report_to=config.get("report_to", "all"),
        skip_memory_metrics=config.get("skip_memory_metrics", True)
    )

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
    model.save_adapter(best_checkpoint, task)
    test(config, model, task_folder)


def test(config, model=None, task_folder=None):
    task = config["task"]
    assert task == "qe_da" or task == "qe_hter"
    if not model:
        logger.info(f"Loading task adapter from {config['adapter_path']}")
        model_config = XLMRobertaConfig.from_pretrained(config.get("model", "xlm-roberta-base"), num_labels=1)
        model = XLMRobertaModelWithHeads.from_pretrained(config.get("model", "xlm-roberta-base"), config=model_config)
        model.load_adapter(config["adapter_path"], model_name=task)
    if not task_folder:
        task_folder = f"test_{config.get('task_name', '')}_{config['task']}{datetime.now().strftime('%Y-%m-%d_%H-%M')}"
    output_dir = os.path.join(config["output_dir"], task_folder)
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Saving results in {output_dir}")
    yaml.dump(config, open(os.path.join(output_dir, "test_config.yaml"), "w"))
    for pair in config["test"]["pairs"]:
        lang1 = pair[0]
        lang2 = pair[1]
        dataset = load_data(lang1, lang2, task, config)
        logger.info(f"Evaluation results for {task} {lang1}-{lang2}")
        model.load_adapter(f"{lang1}/wiki@ukp")
        model.load_adapter(f"{lang2}/wiki@ukp")

        model.set_active_adapters([ac.Split(lang1, lang2, split_index=config.get("max_seq_len", 50)), task])

        eval_trainer = Trainer(
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
        evaluation = eval_trainer.evaluate(metric_key_prefix="test")
        logger.info(evaluation)
        json.dump(evaluation, open(os.path.join(output_dir, f"da_{lang1}_{lang2}.json"), "w"))


def load_data(lang1, lang2, task, config):
    tokenizer = XLMRobertaTokenizer.from_pretrained(config.get("model", "xlm-roberta-base"))
    if config.get("model", "xlm-roberta-base") != "xlm-roberta-base":
        logger.warning("encode_batch not implemented for non-xlmr models!")

    if task == "qe_da":
        dataset = load_dataset("csv", delimiter="\t", quoting=3, data_files={
            "train": f"data/data/direct-assessments/train/{lang1}-{lang2}-train/train.{lang1}{lang2}.df.short.tsv",
            "test": f"data/data/direct-assessments/test/{lang1}-{lang2}/test20.{lang1}{lang2}.df.short.tsv",
            "dev": f"data/data/direct-assessments/dev/{lang1}-{lang2}-dev/dev.{lang1}{lang2}.df.short.tsv"
        })
        # The transformers model expects the target class column to be named "labels"
        dataset = dataset.rename_column("z_mean", "label")

    if task == "qe_hter":
        def read_f(f, dt):
            return [dt(l.strip()) for l in open(f, encoding="utf-8").readlines()]
        train_hter = read_f(f"data/data/post-editing/train/{lang1}-{lang2}-train/train.hter", float)
        dev_hter = read_f(f"data/data/post-editing/dev/{lang1}-{lang2}-dev/dev.hter", float)
        test_hter = read_f(f"data/data/post-editing/test/{lang1}-{lang2}-test20/test20.hter", float)
        train_src = read_f(f"data/data/post-editing/train/{lang1}-{lang2}-train/train.src", str)
        dev_src = read_f(f"data/data/post-editing/dev/{lang1}-{lang2}-dev/dev.src", str)
        test_src = read_f(f"data/data/post-editing/test/{lang1}-{lang2}-test20/test20.src", str)
        train_mt = read_f(f"data/data/post-editing/train/{lang1}-{lang2}-train/train.mt", str)
        dev_mt = read_f(f"data/data/post-editing/dev/{lang1}-{lang2}-dev/dev.mt", str)
        test_mt = read_f(f"data/data/post-editing/test/{lang1}-{lang2}-test20/test20.mt", str)
        train = Dataset.from_dict({"original": train_src, "translation": train_mt, "label": train_hter}, split="train")
        dev = Dataset.from_dict({"original": dev_src, "translation": dev_mt, "label": dev_hter}, split="dev")
        test = Dataset.from_dict({"original": test_src, "translation": test_mt, "label": test_hter}, split="test")
        dataset = DatasetDict({"train": train, "dev": dev, "test": test})

    def encode_batch(batch):
        """Encodes a batch of input data using the model tokenizer."""
        sen1 = tokenizer(batch["original"], max_length=config.get("max_seq_len", 50), truncation=True, padding="max_length")
        sen2 = tokenizer(batch["translation"], max_length=config.get("max_seq_len", 50), truncation=True, padding="max_length")

        for i1, i2 in zip(sen1["input_ids"], sen2["input_ids"]):
            i1.append(2)
            i1.extend(i2[1:])

        for a1, a2 in zip(sen1["attention_mask"], sen2["attention_mask"]):
            a1.extend(a2)

        return sen1
    # Encode the input data
    dataset = dataset.map(encode_batch, batched=True)
    # Transform to pytorch tensors and only output the required columns
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    return dataset


def compute_pearson(p: EvalPrediction):
    return {"pearson": pearsonr(p.predictions.reshape(-1), p.label_ids)[0]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    config = yaml.load(open(args.config), Loader=yaml.FullLoader)
    main(config)