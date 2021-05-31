import argparse
import collections
import json
import os
from datetime import datetime
from typing import Union, List, Optional

import torch
from torch.nn import Embedding
import yaml
from torch.utils.data import DataLoader, Sampler
from transformers import TrainingArguments, Trainer, \
    EvalPrediction, TrainerCallback, AutoConfig, AutoTokenizer, AutoModelWithHeads
import numpy as np
from scipy.stats import pearsonr
import transformers.adapters.composition as ac
from datasets import load_dataset, Dataset, DatasetDict
import logging
import random
from transformers.adapters.configuration import AdapterConfig
from transformers.adapters.utils import resolve_adapter_path
from transformers.trainer_pt_utils import nested_concat, DistributedTensorGatherer, SequentialDistributedSampler
from transformers.trainer_utils import PredictionOutput, denumpify_detensorize

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)

LANG_TO_OFFSET = {"si": 0, "km": 1}

def main(config):
    torch.manual_seed(config.get("seed", 400002021))
    np.random.seed(config.get("seed", 400002021))
    random.seed(config.get("seed", 400002021))

    os.environ["WANDB_WATCH"] = "false"
    if config.get("wandb_project", ""):
        os.environ["WANDB_PROJECT"] = config["wandb_project"]
    if config.get("do_train", True):
        train(config)
    else:
        test(config)


def train(config):
    logging.info(config)
    task_folder = f"train_{config.get('task_name', '')}{config['task']}_{datetime.now().strftime('%Y-%m-%d_%H-%M')}"
    output_dir = os.path.join(config["output_dir"], task_folder)
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Saving results in {output_dir}")
    yaml.dump(config, open(os.path.join(output_dir, "train_config.yaml"), "w"))

    model_config = AutoConfig.from_pretrained(config.get("model", "xlm-roberta-base"), num_labels=1, hidden_dropout_prob=config.get("dropout", 0.1))
    model = AutoModelWithHeads.from_pretrained(config.get("model", "xlm-roberta-base"), config=model_config)

    task = config["task"]
    assert task == "qe_da" or task == "qe_hter"
    adapter_config = AdapterConfig.load("pfeiffer", non_linearity="gelu", reduction_factor=config.get("reduction_factor", 16))
    if config.get("architecture", "base") == "split":
        model.add_adapter(task+"_original", config=adapter_config)
        model.add_adapter(task+"_translation", config=adapter_config)
        model.add_classification_head(task+"_original", num_labels=1)
    elif config.get("architecture", "base") == "tri":
        model.add_adapter(task+"_original", config=adapter_config)
        model.add_adapter(task+"_translation", config=adapter_config)
        model.add_adapter(task+"_tri", config=adapter_config)
        model.add_classification_head(task+"_tri", num_labels=1)
    else:
        model.add_adapter(task, config=adapter_config)
        model.add_classification_head(task, num_labels=1)

    train_config = config["train"]
    is_multipair = not isinstance(train_config["pair"][0], str)
    logging.info(f"Training for {task} {train_config['pair']}")
    pairs = train_config["pair"] if is_multipair else [train_config["pair"]]

    for train_lang1, train_lang2 in pairs:
        load_lang_adapter(model, train_lang1, config)
        load_lang_adapter(model, train_lang2, config)
    if config.get("architecture", "base") == "split":
        model.train_adapter([task+"_original", task+"_translation"])
    elif config.get("architecture", "base") == "tri":
        model.train_adapter([task+"_original", task+"_translation", task+"_tri"])
    else:
        model.train_adapter([task])
    dataset = load_data(train_config["pair"], task, config)

    training_args = TrainingArguments(
        learning_rate=train_config.get("learning_rate", 0.0001),
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

    skip_layer = []  # [11] if config.get("madx2", False) else []
    if is_multipair:
        if config.get("architecture", "base") == "split":
            task_adapter = [ac.Split(task+"_original", task+"_translation", split_index=config.get("max_seq_len", 50))]
        elif config.get("architecture", "base") == "tri":
            task_adapter = [ac.Split(task+"_original", task+"_translation", split_index=config.get("max_seq_len", 50)),
                            task+"_tri"]
        else:
            task_adapter = [task]
        trainer = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["dev"],
            compute_metrics=compute_pearson,
            do_save_adapters=True,
            do_save_full_model=False
        )
        trainer.add_callback(AdapterLangCallback(pairs, task_adapter, config.get("max_seq_len", 50), skip_layer, config.get("no_lang", False)))
    else:
        train_lang1, train_lang2 = pairs[0]
        if config.get("architecture", "base") == "split":
            task_adapter = [ac.Split(task+"_original", task+"_translation", split_index=config.get("max_seq_len", 50))]
        elif config.get("architecture", "base") == "tri":
            task_adapter = [ac.Split(task+"_original", task+"_translation", split_index=config.get("max_seq_len", 50)),
                            task+"_tri"]
        else:
            task_adapter = [task]
        if config.get("no_lang", False):
            setup = [*task_adapter]
        else:
            setup = [ac.Split(train_lang1, train_lang2, split_index=config.get("max_seq_len", 50)), *task_adapter]
        model.set_active_adapters(setup, skip_layers=skip_layer)
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
    if config.get("architecture", "base") == "split" or config.get("architecture", "base") == "tri":
        best_checkpoint = os.path.join(output_dir, "best_checkpoint")
        folder1 = os.path.join(best_checkpoint, task+"_original")
        folder2 = os.path.join(best_checkpoint, task+"_translation")
        os.makedirs(folder1, exist_ok=True)
        os.makedirs(folder2, exist_ok=True)
        model.save_adapter(folder1, task+"_original")
        model.save_adapter(folder2, task+"_translation")
        if config.get("architecture", "base") == "tri":
            folder3 = os.path.join(best_checkpoint, task+"_tri")
            os.makedirs(folder3, exist_ok=True)
            model.save_adapter(folder3, task+"_tri")
    else:
        best_checkpoint = os.path.join(output_dir, "best_checkpoint", task)
        os.makedirs(best_checkpoint, exist_ok=True)
        model.save_adapter(best_checkpoint, task)
    test(config, model, task_folder)


def test(config, model=None, task_folder=None):
    task = config["task"]
    assert task == "qe_da" or task == "qe_hter"
    if not model:
        if isinstance(config['adapter_path'], list):
            return test_ensemble(config)
        logging.info(f"Loading task adapter from {config['adapter_path']}")
        model_config = AutoConfig.from_pretrained(config.get("model", "xlm-roberta-base"), num_labels=1, hidden_dropout_prob=config.get("dropout", 0.1))
        model = AutoModelWithHeads.from_pretrained(config.get("model", "xlm-roberta-base"), config=model_config)
        if config.get("architecture", "base") == "split" or config.get("architecture", "base") == "tri":
            model.load_adapter(os.path.join(config["adapter_path"], task+"_original"), load_as=task+"_original")
            model.load_adapter(os.path.join(config["adapter_path"], task+"_translation"), load_as=task+"_translation")
            if config.get("architecture", "base") == "tri":
                model.load_adapter(os.path.join(config["adapter_path"], task+"_tri"), load_as=task+"_tri")
        else:
            model.load_adapter(os.path.join(config["adapter_path"], task), load_as=task)
    if not task_folder:
        task_folder = f"test_{config.get('task_name', '')}_{config['task']}{datetime.now().strftime('%Y-%m-%d_%H-%M')}"
    output_dir = os.path.join(config["output_dir"], task_folder)
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Saving results in {output_dir}")
    yaml.dump(config, open(os.path.join(output_dir, "test_config.yaml"), "w"))
    results = {"dev": [], "test": [], "task": task}
    for pair in config["test"]["pairs"]:
        lang1, lang2 = pair
        logging.info(f"Evaluation results for {task} {lang1}-{lang2}")
        load_lang_adapter(model, lang1, config)
        load_lang_adapter(model, lang2, config)
        dataset = load_data(pair, task, config)

        skip_layer = [] # [11] if config.get("madx2", False) else []
        if config.get("architecture", "base") == "split":
            task_adapter = [ac.Split(task+"_original", task+"_translation", split_index=config.get("max_seq_len", 50))]
        elif config.get("architecture", "base") == "tri":
            task_adapter = [ac.Split(task+"_original", task+"_translation", split_index=config.get("max_seq_len", 50)),
                            task+"_tri"]
        else:
            task_adapter = [task]
        if config.get("no_lang", False):
            setup = [*task_adapter]
        else:
            setup = [ac.Split(lang1, lang2, split_index=config.get("max_seq_len", 50)), *task_adapter]
        model.set_active_adapters(setup, skip_layers=skip_layer)
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
    logging.info(results)
    json.dump(results, open(os.path.join(output_dir, f"evaluation_{task}.json"), "w"), indent=2)


def load_lang_adapter(model, language, config):
    if not config.get("no_lang", False):
        download_langs = set(["ro", "si", "km", "ps", "ne"])
        if language in download_langs:
            model.load_adapter(
                f"https://public.ukp.informatik.tu-darmstadt.de/AdapterHub/text_lang/{language}/bert-base-multilingual-cased/pfeiffer/{language}.zip",
                with_head=False)
        else:
            model.load_adapter(f"{language}/wiki@ukp", with_head=False)
        if config.get("madx2", False):
            try:
                del model.base_model.encoder.layer._modules['11'].output.adapters[language]
                del model.base_model.encoder.layer._modules['11'].attention.output.adapters[language]
            except KeyError:
                pass
    if language in ["si", "km"] and config.get("extend_embeddings", False):
        if "token_offset" not in config:
            config["token_offset"] = [-1, -1]
        path = resolve_adapter_path(f"https://public.ukp.informatik.tu-darmstadt.de/AdapterHub/text_lang/{language}/bert-base-multilingual-cased/pfeiffer/{language}.zip")
        added_embeddings = torch.load(os.path.join(path, "pytorch_model_embeddings.bin"))["bert.embeddings.word_embeddings.weight"].to(model.device)
        current_emb_len = model.bert.embeddings.word_embeddings.num_embeddings
        config["token_offset"][LANG_TO_OFFSET[language]] = current_emb_len
        new_embedding = Embedding(current_emb_len+added_embeddings.shape[0], added_embeddings.shape[1], padding_idx=0)
        new_embedding.weight.data = torch.cat([model.bert.embeddings.word_embeddings.weight.data, added_embeddings])
        model.bert.embeddings.word_embeddings = new_embedding


def load_data(lang_pairs, task, config):
    if isinstance(lang_pairs[0], str):
        lang_pairs = [lang_pairs]
    tokenizer = AutoTokenizer.from_pretrained(config.get("model", "xlm-roberta-base"))

    if task == "qe_da":
        dataset = load_dataset("csv", delimiter="\t", quoting=3, data_files={
            "train": [f"data/data/direct-assessments/train/{lang1}-{lang2}-train/train.{lang1}{lang2}.df.short.tsv" for (lang1, lang2) in lang_pairs],
            "test": [f"data/data/direct-assessments/test/{lang1}-{lang2}/test20.{lang1}{lang2}.df.short.tsv" for (lang1, lang2) in lang_pairs],
            "dev": [f"data/data/direct-assessments/dev/{lang1}-{lang2}-dev/dev.{lang1}{lang2}.df.short.tsv" for (lang1, lang2) in lang_pairs]
        })
        # The transformers model expects the target class column to be named "labels"
        dataset = dataset.rename_column("z_mean", "label")

    if task == "qe_hter":
        def read_f(f, dt):
            return [dt(l.strip()) for l in open(f, encoding="utf-8").readlines()]
        train_hter, dev_hter, test_hter, train_src, dev_src, test_src, train_mt, dev_mt, test_mt = [], [], [], [], [], [], [], [], []
        for (lang1, lang2) in lang_pairs:
            train_hter.extend(read_f(f"data/data/post-editing/train/{lang1}-{lang2}-train/train.hter", float))
            dev_hter.extend(read_f(f"data/data/post-editing/dev/{lang1}-{lang2}-dev/dev.hter", float))
            test_hter.extend(read_f(f"data/data/post-editing/test/{lang1}-{lang2}-test20/test20.hter", float))
            train_src.extend(read_f(f"data/data/post-editing/train/{lang1}-{lang2}-train/train.src", str))
            dev_src.extend(read_f(f"data/data/post-editing/dev/{lang1}-{lang2}-dev/dev.src", str))
            test_src.extend(read_f(f"data/data/post-editing/test/{lang1}-{lang2}-test20/test20.src", str))
            train_mt.extend(read_f(f"data/data/post-editing/train/{lang1}-{lang2}-train/train.mt", str))
            dev_mt.extend(read_f(f"data/data/post-editing/dev/{lang1}-{lang2}-dev/dev.mt", str))
            test_mt.extend(read_f(f"data/data/post-editing/test/{lang1}-{lang2}-test20/test20.mt", str))
        train = Dataset.from_dict({"original": train_src, "translation": train_mt, "label": train_hter}, split="train")
        dev = Dataset.from_dict({"original": dev_src, "translation": dev_mt, "label": dev_hter}, split="dev")
        test = Dataset.from_dict({"original": test_src, "translation": test_mt, "label": test_hter}, split="test")
        dataset = DatasetDict({"train": train, "dev": dev, "test": test})

    def encode(is_train, lang_pairs):
        def _encode(all_data):
            """Encodes a batch of input data using the model tokenizer."""
            original = all_data["original"]
            translation = all_data["translation"]
            if "prompt" in config:
                prompt_orig, prompt_transl = config["prompt"]
                original = [f"{prompt_orig}: {o}" for o in original]
                translation = [f"{prompt_transl}: {t}" for t in translation]

            sen1 = tokenizer(original, max_length=config.get("max_seq_len", 50), truncation=True, padding="max_length")
            if config.get("extend_embeddings", False):
                factor = 7000 if is_train else 1000
                for i, (lang1, lang2) in enumerate(lang_pairs):
                    if lang1 in ["si", "km"]:
                        alt_tokenizer = AutoTokenizer.from_pretrained(f"data/{lang1}-tokenizer")
                        sen1_alt = alt_tokenizer(original[i*factor:i*factor+factor], max_length=config.get("max_seq_len", 50), truncation=True, padding="max_length")
                        offset = config["token_offset"][LANG_TO_OFFSET[lang1]]
                        sen1["input_ids"][i*factor:i*factor+factor] = [[i+offset for i in sen] for sen in sen1_alt["input_ids"]]

            sen2 = tokenizer(translation, max_length=config.get("max_seq_len", 50), truncation=True, padding="max_length")

            if "xlm" in config.get("model", "xlm-roberta-base"):
                for i1, i2 in zip(sen1["input_ids"], sen2["input_ids"]):
                    i1.append(2)
                    i1.extend(i2[1:])

                for a1, a2 in zip(sen1["attention_mask"], sen2["attention_mask"]):
                    a1.extend(a2)
            else:
                for i1, i2 in zip(sen1["input_ids"], sen2["input_ids"]):
                    i1.extend(i2[1:])

                for a1, a2 in zip(sen1["attention_mask"], sen2["attention_mask"]):
                    a1.extend(a2[1:])

                for t1, t2 in zip(sen1["token_type_ids"], sen2["token_type_ids"]):
                    t1.extend([1]*(len(t2)-1))

            return sen1
        return _encode
    # Encode the input data
    dataset["train"] = dataset["train"].map(encode(True, lang_pairs), batched=True, batch_size=7000*len(lang_pairs))
    dataset["test"] = dataset["test"].map(encode(False, lang_pairs), batched=True, batch_size=1000*len(lang_pairs))
    dataset["dev"] = dataset["dev"].map(encode(False, lang_pairs), batched=True, batch_size=1000*len(lang_pairs))
    # Transform to pytorch tensors and only output the required columns
    if "xlm" in config.get("model", "xlm-roberta-base"):
        dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    else:
        dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "token_type_ids", "label"])
    return dataset


def compute_pearson(p: EvalPrediction):
    if len(p.predictions) == 1000: #hard coded
        return {"pearson": pearsonr(p.predictions.reshape(-1), p.label_ids)[0]}
    else:
        predictions = p.predictions.reshape(-1)
        all_r = []
        for i in range(0, len(predictions), 1000):
            all_r.append(pearsonr(predictions[i:i+1000], p.label_ids[i:i+1000])[0])
        return {"pearson": np.mean(all_r)}


def test_ensemble(config):
    task = config["task"]
    assert task == "qe_da" or task == "qe_hter"
    logging.info(f"Loading task adapter from {config['adapter_path']}")
    model_config = AutoConfig.from_pretrained(config.get("model", "xlm-roberta-base"), num_labels=1, hidden_dropout_prob=config.get("dropout", 0.1))
    model = AutoModelWithHeads.from_pretrained(config.get("model", "xlm-roberta-base"), config=model_config)
    all_task_adapters = []
    architectures = config.get("architecture", "base")
    for i, adapter_path in enumerate(config['adapter_path']):
        if isinstance(architectures, list):
            architecture = architectures[i]
        else:
            architecture = architectures
        if architecture == "split" or architecture == "tri":
            model.load_adapter(os.path.join(adapter_path, task+"_original"), load_as=task+"_original"+str(i))
            model.load_adapter(os.path.join(adapter_path, task+"_translation"), load_as=task+"_translation"+str(i))
            if architecture == "tri":
                model.load_adapter(os.path.join(adapter_path, task+"_tri"), load_as=task+"_tri"+str(i))
        else:
            model.load_adapter(os.path.join(config["adapter_path"], task), load_as=task+str(i))

        if architecture == "split":
            task_adapter = ac.Split(task+"_original"+str(i), task+"_translation"+str(i), split_index=config.get("max_seq_len", 50))
        elif architecture == "tri":
            task_adapter = [ac.Split(task+"_original"+str(i), task+"_translation"+str(i), split_index=config.get("max_seq_len", 50)),
                            task+"_tri"+str(i)]
        else:
            task_adapter = task+str(i)
        all_task_adapters.append(task_adapter)

    task_folder = f"test_{config.get('task_name', '')}_{config['task']}{datetime.now().strftime('%Y-%m-%d_%H-%M')}"
    output_dir = os.path.join(config["output_dir"], task_folder)
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Saving results in {output_dir}")
    yaml.dump(config, open(os.path.join(output_dir, "test_config.yaml"), "w"))
    results = {"dev": [], "test": [], "task": task}
    for pair in config["test"]["pairs"]:
        lang1, lang2 = pair
        logging.info(f"Evaluation results for {task} {lang1}-{lang2}")
        load_lang_adapter(model, lang1, config)
        load_lang_adapter(model, lang2, config)
        dataset = load_data(pair, task, config)

        skip_layer = [] # [11] if config.get("madx2", False) else []
        if config.get("no_lang", False):
            setup = all_task_adapters
        else:
            setup = [[ac.Split(lang1, lang2, split_index=config.get("max_seq_len", 50)), a] for a in all_task_adapters]
        #model.set_active_adapters(setup, skip_layers=skip_layer)
        dev_trainer = EnsembleTrainer(
            model=model,
            args=TrainingArguments(output_dir=output_dir,
                                   remove_unused_columns=False,
                                   per_device_eval_batch_size=config["test"]["batchsize"],
                                   run_name=task_folder,
                                   report_to=config.get("report_to", "all"),
                                   skip_memory_metrics=config.get("skip_memory_metrics", True)),
            eval_dataset=dataset["dev"],
            compute_metrics=compute_pearson,
            adapter_setup=setup
        )
        dev_evaluation = dev_trainer.evaluate(metric_key_prefix="dev")
        dev_evaluation["pair"] = f"{lang1}_{lang2}"
        results["dev"].append(dev_evaluation)
        test_trainer = EnsembleTrainer(
            model=model,
            args=TrainingArguments(output_dir=output_dir,
                                   remove_unused_columns=False,
                                   per_device_eval_batch_size=config["test"]["batchsize"],
                                   run_name=task_folder,
                                   report_to=config.get("report_to", "all"),
                                   skip_memory_metrics=config.get("skip_memory_metrics", True)),
            eval_dataset=dataset["test"],
            compute_metrics=compute_pearson,
            adapter_setup=setup
        )
        test_evaluation = test_trainer.evaluate(metric_key_prefix="test")
        test_evaluation["pair"] = f"{lang1}_{lang2}"
        results["test"].append(test_evaluation)
    logging.info(results)
    json.dump(results, open(os.path.join(output_dir, f"evaluation_{task}.json"), "w"), indent=2)



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

    def prediction_loop(
            self,
            dataloader: DataLoader,
            description: str,
            prediction_loss_only: Optional[bool] = None,
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "eval",
    ) -> PredictionOutput:
        """
        Prediction/evaluation loop, shared by :obj:`Trainer.evaluate()` and :obj:`Trainer.predict()`.

        Works both with or without labels.
        """
        if not isinstance(dataloader.dataset, collections.abc.Sized):
            raise ValueError("dataset must implement __len__")
        prediction_loss_only = (
            prediction_loss_only if prediction_loss_only is not None else self.args.prediction_loss_only
        )

        if self.args.deepspeed and not self.args.do_train:
            # no harm, but flagging to the user that deepspeed config is ignored for eval
            # flagging only for when --do_train wasn't passed as only then it's redundant
            logging.info("Detected the deepspeed argument but it will not be used for evaluation")

        model = self._wrap_model(self.model, training=False)

        # if full fp16 is wanted on eval and this ``evaluation`` or ``predict`` isn't called while
        # ``train`` is running, half it first and then put on device
        if not self.is_in_train and self.args.fp16_full_eval:
            model = model.half().to(self.args.device)

        batch_size = dataloader.batch_size
        num_examples = self.num_examples(dataloader)
        logging.info(f"***** Running {description} *****")
        logging.info(f"  Num examples = {num_examples}")
        logging.info(f"  Batch size = {batch_size}")
        losses_host: torch.Tensor = None
        preds_host: Union[torch.Tensor, List[torch.Tensor]] = None
        labels_host: Union[torch.Tensor, List[torch.Tensor]] = None

        world_size = max(1, self.args.world_size)

        eval_losses_gatherer = DistributedTensorGatherer(world_size, num_examples, make_multiple_of=batch_size)
        if not prediction_loss_only:
            # The actual number of eval_sample can be greater than num_examples in distributed settings (when we pass
            # a batch size to the sampler)
            make_multiple_of = None
            if hasattr(dataloader, "sampler") and isinstance(dataloader.sampler, SequentialDistributedSampler):
                make_multiple_of = dataloader.sampler.batch_size
            preds_gatherer = DistributedTensorGatherer(world_size, num_examples, make_multiple_of=make_multiple_of)
            labels_gatherer = DistributedTensorGatherer(world_size, num_examples, make_multiple_of=make_multiple_of)

        model.eval()

        if self.args.past_index >= 0:
            self._past = None

        self.callback_handler.eval_dataloader = dataloader

        for step, inputs in enumerate(dataloader):
            ### THIS IS A DIRTY HACK
            if (step * batch_size)%1000 == 0:
                self.callback_handler.callbacks[-1].next_test_adapter(model)
            loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
            if loss is not None:
                losses = loss.repeat(batch_size)
                losses_host = losses if losses_host is None else torch.cat((losses_host, losses), dim=0)
            if logits is not None:
                preds_host = logits if preds_host is None else nested_concat(preds_host, logits, padding_index=-100)
            if labels is not None:
                labels_host = labels if labels_host is None else nested_concat(labels_host, labels, padding_index=-100)
            self.control = self.callback_handler.on_prediction_step(self.args, self.state, self.control)

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            if self.args.eval_accumulation_steps is not None and (step + 1) % self.args.eval_accumulation_steps == 0:
                eval_losses_gatherer.add_arrays(self._gather_and_numpify(losses_host, "eval_losses"))
                if not prediction_loss_only:
                    preds_gatherer.add_arrays(self._gather_and_numpify(preds_host, "eval_preds"))
                    labels_gatherer.add_arrays(self._gather_and_numpify(labels_host, "eval_label_ids"))

                # Set back to None to begin a new accumulation
                losses_host, preds_host, labels_host = None, None, None

        if self.args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        eval_losses_gatherer.add_arrays(self._gather_and_numpify(losses_host, "eval_losses"))
        if not prediction_loss_only:
            preds_gatherer.add_arrays(self._gather_and_numpify(preds_host, "eval_preds"))
            labels_gatherer.add_arrays(self._gather_and_numpify(labels_host, "eval_label_ids"))

        eval_loss = eval_losses_gatherer.finalize()
        preds = preds_gatherer.finalize() if not prediction_loss_only else None
        label_ids = labels_gatherer.finalize() if not prediction_loss_only else None

        if self.compute_metrics is not None and preds is not None and label_ids is not None:
            metrics = self.compute_metrics(EvalPrediction(predictions=preds, label_ids=label_ids))
        else:
            metrics = {}

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if eval_loss is not None:
            metrics[f"{metric_key_prefix}_loss"] = eval_loss.mean().item()

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return PredictionOutput(predictions=preds, label_ids=label_ids, metrics=metrics)


class EnsembleTrainer(Trainer):
    def __init__(self, adapter_setup=[], **kwargs):
        super(EnsembleTrainer, self).__init__(**kwargs)
        self.adapter_setup = adapter_setup

    def prediction_loop(
            self,
            dataloader: DataLoader,
            description: str,
            prediction_loss_only: Optional[bool] = None,
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "eval"
    ) -> PredictionOutput:
        """
        Prediction/evaluation loop, shared by :obj:`Trainer.evaluate()` and :obj:`Trainer.predict()`.

        Works both with or without labels.
        """
        if not isinstance(dataloader.dataset, collections.abc.Sized):
            raise ValueError("dataset must implement __len__")
        prediction_loss_only = (
            prediction_loss_only if prediction_loss_only is not None else self.args.prediction_loss_only
        )

        if self.args.deepspeed and not self.args.do_train:
            # no harm, but flagging to the user that deepspeed config is ignored for eval
            # flagging only for when --do_train wasn't passed as only then it's redundant
            logging.info("Detected the deepspeed argument but it will not be used for evaluation")

        model = self._wrap_model(self.model, training=False)

        # if full fp16 is wanted on eval and this ``evaluation`` or ``predict`` isn't called while
        # ``train`` is running, half it first and then put on device
        if not self.is_in_train and self.args.fp16_full_eval:
            model = model.half().to(self.args.device)

        batch_size = dataloader.batch_size
        num_examples = self.num_examples(dataloader)
        logging.info(f"***** Running {description} *****")
        logging.info(f"  Num examples = {num_examples}")
        logging.info(f"  Batch size = {batch_size}")
        losses_host: torch.Tensor = None
        preds_host: Union[torch.Tensor, List[torch.Tensor]] = None
        labels_host: Union[torch.Tensor, List[torch.Tensor]] = None

        world_size = max(1, self.args.world_size)

        eval_losses_gatherer = DistributedTensorGatherer(world_size, num_examples, make_multiple_of=batch_size)
        if not prediction_loss_only:
            # The actual number of eval_sample can be greater than num_examples in distributed settings (when we pass
            # a batch size to the sampler)
            make_multiple_of = None
            if hasattr(dataloader, "sampler") and isinstance(dataloader.sampler, SequentialDistributedSampler):
                make_multiple_of = dataloader.sampler.batch_size
            preds_gatherer = DistributedTensorGatherer(world_size, num_examples, make_multiple_of=make_multiple_of)
            labels_gatherer = DistributedTensorGatherer(world_size, num_examples, make_multiple_of=make_multiple_of)

        model.eval()

        if self.args.past_index >= 0:
            self._past = None

        self.callback_handler.eval_dataloader = dataloader

        for setup in self.adapter_setup:
            model.set_active_adapters(setup)
            for step, inputs in enumerate(dataloader):
                loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
                if loss is not None:
                    losses = loss.repeat(batch_size)
                    losses_host = losses if losses_host is None else torch.cat((losses_host, losses), dim=0)
                if logits is not None:
                    preds_host = logits if preds_host is None else nested_concat(preds_host, logits, padding_index=-100)
                if labels is not None:
                    labels_host = labels if labels_host is None else nested_concat(labels_host, labels, padding_index=-100)
                self.control = self.callback_handler.on_prediction_step(self.args, self.state, self.control)

                # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
                if self.args.eval_accumulation_steps is not None and (step + 1) % self.args.eval_accumulation_steps == 0:
                    eval_losses_gatherer.add_arrays(self._gather_and_numpify(losses_host, "eval_losses"))
                    if not prediction_loss_only:
                        preds_gatherer.add_arrays(self._gather_and_numpify(preds_host, "eval_preds"))
                        labels_gatherer.add_arrays(self._gather_and_numpify(labels_host, "eval_label_ids"))

                    # Set back to None to begin a new accumulation
                    losses_host, preds_host, labels_host = None, None, None

            if self.args.past_index and hasattr(self, "_past"):
                # Clean the state at the end of the evaluation loop
                delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        eval_losses_gatherer.add_arrays(self._gather_and_numpify(losses_host.reshape(len(self.adapter_setup), -1).mean(dim=0), "eval_losses"))
        if not prediction_loss_only:
            preds_gatherer.add_arrays(self._gather_and_numpify(preds_host.reshape(len(self.adapter_setup), -1).mean(dim=0), "eval_preds"))
            labels_gatherer.add_arrays(self._gather_and_numpify(labels_host[:1000], "eval_label_ids"))

        eval_loss = eval_losses_gatherer.finalize()
        preds = preds_gatherer.finalize() if not prediction_loss_only else None
        label_ids = labels_gatherer.finalize() if not prediction_loss_only else None

        if self.compute_metrics is not None and preds is not None and label_ids is not None:
            metrics = self.compute_metrics(EvalPrediction(predictions=preds, label_ids=label_ids))
        else:
            metrics = {}

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if eval_loss is not None:
            metrics[f"{metric_key_prefix}_loss"] = eval_loss.mean().item()

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return PredictionOutput(predictions=preds, label_ids=label_ids, metrics=metrics)



class AdapterLangCallback(TrainerCallback):
    def __init__(self, pairs, task_adapters, split_index, skip_layers, no_lang):
        self.pairs = pairs
        self.task_adapters = task_adapters
        self.split_index = split_index
        self.skip_layers = skip_layers
        self.no_lang = no_lang
        self.train_idx = 0
        self.test_idx = 0

    def on_step_begin(self, args: TrainingArguments, state, control, **kwargs):
        """
        Event called at the beginning of a training step. If using gradient accumulation, one training step might take
        several inputs.
        """
        model = kwargs["model"]
        train_lang1, train_lang2 = self.pairs[self.train_idx]
        if self.no_lang:
            setup = [*self.task_adapters]
        else:
            setup = [ac.Split(train_lang1, train_lang2, split_index=self.split_index), *self.task_adapters]
        model.set_active_adapters(setup, skip_layers=self.skip_layers)
        self.train_idx += 1
        self.train_idx = self.train_idx % len(self.pairs)

    def next_test_adapter(self, model):
        test_lang1, test_lang2 = self.pairs[self.test_idx]
        if self.no_lang:
            setup = [*self.task_adapters]
        else:
            setup = [ac.Split(test_lang1, test_lang2, split_index=self.split_index), *self.task_adapters]
        model.set_active_adapters(setup, skip_layers=self.skip_layers)
        self.test_idx += 1
        self.test_idx = self.test_idx % len(self.pairs)


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