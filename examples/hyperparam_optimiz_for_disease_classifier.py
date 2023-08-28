#!/usr/bin/env python
# coding: utf-8

# hyperparameter optimization with raytune for disease classification

# imports
import os
import subprocess
GPU_NUMBER = [0,1,2,3]
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(s) for s in GPU_NUMBER])
os.environ["NCCL_DEBUG"] = "INFO"
os.environ["CONDA_OVERRIDE_GLIBC"] = "2.56"
os.environ["LD_LIBRARY_PATH"] = "/path/to/miniconda3/lib:/path/to/sw/lib:/path/to/sw/lib"

# initiate runtime environment for raytune
import pyarrow # must occur prior to ray import
import ray
from ray import tune
from ray.tune import ExperimentAnalysis
from ray.tune.suggest.hyperopt import HyperOptSearch
ray.shutdown() #engage new ray session
runtime_env = {"conda": "base",
               "env_vars": {"LD_LIBRARY_PATH": "/path/to/miniconda3/lib:/path/to/sw/lib:/path/to/sw/lib"}}
ray.init(runtime_env=runtime_env)

def initialize_ray_with_check(ip_address):
    """
    Initialize Ray with a specified IP address and check its status and accessibility.

    Args:
    - ip_address (str): The IP address (with port) to initialize Ray.

    Returns:
    - bool: True if initialization was successful and dashboard is accessible, False otherwise.
    """
    try:
        ray.init(address=ip_address)
        print(ray.nodes())
        
        services = ray.get_webui_url()
        if not services:
            raise RuntimeError("Ray dashboard is not accessible.")
        else:
            print(f"Ray dashboard is accessible at: {services}")
        return True
    except Exception as e:
        print(f"Error initializing Ray: {e}")
        return False

# Usage:
ip = 'your_ip:xxxx'  # Replace with your actual IP address and port
if initialize_ray_with_check(ip):
    print("Ray initialized successfully.")
else:
    print("Error during Ray initialization.")
    
import datetime
import numpy as np
import pandas as pd
import random
import seaborn as sns; sns.set()
from collections import Counter
from datasets import load_from_disk
from scipy.stats import ranksums
from sklearn.metrics import accuracy_score
from transformers import BertForSequenceClassification
from transformers import Trainer
from transformers.training_args import TrainingArguments

from geneformer import DataCollatorForCellClassification

# number of CPU cores
num_proc=30

# load train dataset with columns: 
    # cell_type (annotation of each cell's type)
    # disease (healthy or disease state)
    # individual (unique ID for each patient)
    # length (length of that cell's rank value encoding)
train_dataset=load_from_disk("/path/to/disease_train_data.dataset")

# filter dataset for given cell_type
def if_cell_type(example):
    return example["cell_type"].startswith("Cardiomyocyte")

trainset_v2 = train_dataset.filter(if_cell_type, num_proc=num_proc)

# create dictionary of disease states : label ids
target_names = ["healthy", "disease1", "disease2"]
target_name_id_dict = dict(zip(target_names,[i for i in range(len(target_names))]))

trainset_v3 = trainset_v2.rename_column("disease","label")

# change labels to numerical ids
def classes_to_ids(example):
    example["label"] = target_name_id_dict[example["label"]]
    return example

trainset_v4 = trainset_v3.map(classes_to_ids, num_proc=num_proc)

# separate into train, validation, test sets
indiv_set = set(trainset_v4["individual"])
random.seed(42)
train_indiv = random.sample(indiv_set,round(0.7*len(indiv_set)))
eval_indiv = [indiv for indiv in indiv_set if indiv not in train_indiv]
valid_indiv = random.sample(eval_indiv,round(0.5*len(eval_indiv)))
test_indiv = [indiv for indiv in eval_indiv if indiv not in valid_indiv]

def if_train(example):
    return example["individual"] in train_indiv

classifier_trainset = trainset_v4.filter(if_train,num_proc=num_proc).shuffle(seed=42)

def if_valid(example):
    return example["individual"] in valid_indiv

classifier_validset = trainset_v4.filter(if_valid,num_proc=num_proc).shuffle(seed=42)

# define output directory path
current_date = datetime.datetime.now()
datestamp = f"{str(current_date.year)[-2:]}{current_date.month:02d}{current_date.day:02d}"
output_dir = f"/path/to/models/{datestamp}_geneformer_DiseaseClassifier/"

# ensure not overwriting previously saved model
saved_model_test = os.path.join(output_dir, f"pytorch_model.bin")
if os.path.isfile(saved_model_test) == True:
    raise Exception("Model already saved to this directory.")

# make output directory
subprocess.call(f'mkdir {output_dir}', shell=True)

# set training parameters
# how many pretrained layers to freeze
freeze_layers = 2
# batch size for training and eval
geneformer_batch_size = 12
# number of epochs
epochs = 1
# logging steps
logging_steps = round(len(classifier_trainset)/geneformer_batch_size/10)

# define function to initiate model
def model_init():
    model = BertForSequenceClassification.from_pretrained("/path/to/pretrained_model/",
                                                          num_labels=len(target_names),
                                                          output_attentions = False,
                                                          output_hidden_states = False)
    if freeze_layers is not None:
        modules_to_freeze = model.bert.encoder.layer[:freeze_layers]
        for module in modules_to_freeze:
            for param in module.parameters():
                param.requires_grad = False

    model = model.to("cuda:0")
    return model

# define metrics
# note: macro f1 score recommended for imbalanced multiclass classifiers
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    # calculate accuracy using sklearn's function
    acc = accuracy_score(labels, preds)
    return {
      'accuracy': acc,
    }

# set training arguments
training_args = {
    "do_train": True,
    "do_eval": True,
    "evaluation_strategy": "steps",
    "eval_steps": logging_steps,
    "logging_steps": logging_steps,
    "group_by_length": True,
    "length_column_name": "length",
    "disable_tqdm": True,
    "skip_memory_metrics": True, # memory tracker causes errors in raytune
    "per_device_train_batch_size": geneformer_batch_size,
    "per_device_eval_batch_size": geneformer_batch_size,
    "num_train_epochs": epochs,
    "load_best_model_at_end": True,
    "output_dir": output_dir,
}

training_args_init = TrainingArguments(**training_args)

# create the trainer
trainer = Trainer(
    model_init=model_init,
    args=training_args_init,
    data_collator=DataCollatorForCellClassification(),
    train_dataset=classifier_trainset,
    eval_dataset=classifier_validset,
    compute_metrics=compute_metrics,
)

# specify raytune hyperparameter search space
ray_config = {
    "num_train_epochs": tune.choice([epochs]),
    "learning_rate": tune.loguniform(1e-6, 1e-3),
    "weight_decay": tune.uniform(0.0, 0.3),
    "lr_scheduler_type": tune.choice(["linear","cosine","polynomial"]),
    "warmup_steps": tune.uniform(100, 2000),
    "seed": tune.uniform(0,100),
    "per_device_train_batch_size": tune.choice([geneformer_batch_size])
}

hyperopt_search = HyperOptSearch(
    metric="eval_accuracy", mode="max")

# optimize hyperparameters
trainer.hyperparameter_search(
    direction="maximize",
    backend="ray",
    resources_per_trial={"cpu":8,"gpu":1},
    hp_space=lambda _: ray_config,
    search_alg=hyperopt_search,
    n_trials=100, # number of trials
    progress_reporter=tune.CLIReporter(max_report_frequency=600,
                                                   sort_by_metric=True,
                                                   max_progress_rows=100,
                                                   mode="max",
                                                   metric="eval_accuracy",
                                                   metric_columns=["loss", "eval_loss", "eval_accuracy"])
)