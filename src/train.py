"""Distributed training with Ray Train (train_step, eval_step, train_loop_per_worker, train_model)."""
import os
from datetime import datetime
import numpy as np
import ray
import torch
import torch.nn as nn
from transformers import BertModel
# [nb cell 89, lines 1-12]
# Ray 2.58: ray.air.Checkpoint / ray.air.session were removed.
# Everything now lives under ray.train (use train.report / train.get_context /
# train.get_dataset_shard / train.get_checkpoint in place of `session.*`).
import tempfile
import ray.train as train
from ray.train import Checkpoint, CheckpointConfig, DataConfig, RunConfig, ScalingConfig
from ray.train.torch import TorchTrainer
import torch.nn.functional as F
from torch.nn.parallel.distributed import DistributedDataParallel

# [nb cell 101, line 2]
from ray.data import ExecutionOptions
# [nb cell 102, line 3]
import gc

from madewithml.config import TEST_SIZE, TRAIN_LOOP_CONFIG
from madewithml.data import CustomPreprocessor, load_data, stratify_split
from madewithml.models import FinetunedLLM
from madewithml.utils import collate_fn, set_seeds


# [nb cell 90, lines 1-14]
def train_step(ds, batch_size, model, num_classes, loss_fn, optimizer):
    
    model.train()
    loss = 0.0
    ds_generator = ds.iter_torch_batches(batch_size=batch_size, collate_fn=collate_fn) #lazy iterator
    for i, batch in enumerate(ds_generator):
        optimizer.zero_grad()  # reset gradients
        z = model(batch)  # forward pass
        targets =batch["targets"] # one-hot (for loss_fn if BCE , CE doesnt need One hot for single label multiclass)
        J = loss_fn(z, targets)  # define loss
        J.backward()  # backward pass
        optimizer.step()  # update weights
        loss += (J.detach().item() - loss) / (i + 1)  # cumulative loss
    return loss


# [nb cell 91, lines 1-14]
def eval_step(ds, batch_size, model, num_classes, loss_fn):
    model.eval()
    loss = 0.0
    y_trues, y_preds = [], []
    ds_generator = ds.iter_torch_batches(batch_size=batch_size, collate_fn=collate_fn)
    with torch.inference_mode():
        for i, batch in enumerate(ds_generator):
            z = model(batch)
            targets = batch["targets"]
            J = loss_fn(z, targets).item()
            loss += (J - loss) / (i + 1) #better compute wise  to keep running average, we could do append and then average at the end 
            y_trues.extend(batch["targets"].cpu().numpy())
            y_preds.extend(torch.argmax(z, dim=1).cpu().numpy())
    return loss, np.vstack(y_trues), np.vstack(y_preds)


# [nb cell 92, lines 1-21] (markdown cell, copied as comments)
#                      trainer.fit()
#                ┌──────────┴──────────┐
#         Worker 0 (cuda:0)     Worker 1 (cuda:1)
#         shard 0 of data       shard 1 of data
#         same code ↓           same code ↓
#         load BERT             load BERT
#         prepare_model ◄──sync──► prepare_model
#         epoch loop:           epoch loop:
#           each step ◄──grad avg──► each step
#           report    ◄──wait────►  report
#
# Step	                           What happens                                	Why
# Read config       	            Get hyperparameters	                   One function, many configs
# set_seeds	                      Seed this process	                     Reproducible runs
# get_dataset_shard 	            Get this worker's data                 	Workers don't overlap
# from_pretrained + FinetunedLLM	SciBERT + random head     	            Start of fine-tuning
# prepare_model	                  To GPU + DDP + sync weights	            All copies start equal and stay in sync
# Adam + scheduler	               Update rule + lr decay	                Learn nd take smaller steps when stuck
# batch_size // world_size	       Split the batch	                      Same mini batch for any worker count
# train_step / eval_step	          Learn / measure	                       One epoch each
# save + report	                    Save weights + log metrics	          Pick the best model later and load it for inference

# [nb cell 93, lines 1-44]
# Training loop
def train_loop_per_worker(config):
    # Hyperparameters
    dropout_p = config["dropout_p"]
    lr = config["lr"]
    lr_factor = config["lr_factor"]
    lr_patience = config["lr_patience"]
    num_epochs = config["num_epochs"]
    batch_size = config["batch_size"]
    num_classes = config["num_classes"]

    # Get datasets
    set_seeds()
    train_ds = train.get_dataset_shard("train")
    val_ds = train.get_dataset_shard("val")

    # Model
    llm = BertModel.from_pretrained("allenai/scibert_scivocab_uncased", return_dict=False)
    model = FinetunedLLM(llm=llm, dropout_p=dropout_p, embedding_dim=llm.config.hidden_size, num_classes=num_classes)
    model = train.torch.prepare_model(model) #now the model is DDP wrapper . FinetunedLLM is model.module

    # Training components always after prepare model so that it holds the paramter on already on the GPU  
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=lr_factor, patience=lr_patience)

    # Training
    batch_size_per_worker = batch_size // train.get_context().get_world_size() 
    for epoch in range(num_epochs):
        # Step
        train_loss = train_step(train_ds, batch_size_per_worker, model, num_classes, loss_fn, optimizer)
        val_loss, _, _ = eval_step(val_ds, batch_size_per_worker, model, num_classes, loss_fn)
        scheduler.step(val_loss)

        # Checkpoint
        # DDP keeps weights identical on every rank, so only rank 0 saves. If every rank uploads,
        # they all copy model.pt into the same checkpoint dir and Windows fails with WinError 32
        # (file in use). Every rank must still call train.report -- it is a sync barrier.
        metrics = dict(epoch=epoch, lr=optimizer.param_groups[0]["lr"], train_loss=train_loss, val_loss=val_loss)
        if train.get_context().get_world_rank() == 0:
            with tempfile.TemporaryDirectory() as dp:
                if isinstance(model, DistributedDataParallel):  # cpu
                    model.module.save(dp=dp) #Finetuned LLm's save method
                else:
                    model.save(dp=dp)
                train.report(metrics, checkpoint=Checkpoint.from_directory(dp))
        else:
            train.report(metrics, checkpoint=None)

# [nb cell 94, lines 1-11]
#if imbalanced
#  # Class weights
# batch_counts = []
# for batch in train_ds.iter_torch_batches(batch_size=256, collate_fn=collate_fn):
#     batch_counts.append(np.bincount(batch["targets"].cpu().numpy()))
# counts = [sum(count) for count in zip(*batch_counts)]
# class_weights = np.array([1.0/count for i, count in enumerate(counts)])
# class_weights_tensor = torch.Tensor(class_weights).to(get_device())

# # Training components
# loss_fn = nn.BCEWithLogitsLoss(weight=class_weights_tensor)


def train_model(num_samples=None, num_workers_override=None, **config_overrides):
    """Notebook cells 2, 31, 69 (last line) and 95-105 as one function.

    Defaults reproduce the notebook run. For a quick check:
        train_model(num_samples=100, num_workers_override=1, num_epochs=1)
    """
    # [nb cell 2, lines 1-2]
    if ray.is_initialized():
        ray.shutdown()
    # [nb cell 31, lines 1-2]
    ray.data.DatasetContext.get_current().execution_options.preserve_order=True #deterministic
    ray.init()
    # [nb cell 69, line 11]
    set_seeds()
    # [nb cell 99, line 1] (modified, see CELL_MAP.md)
    ds = load_data(num_samples=num_samples)
    train_ds, val_ds = stratify_split(ds, stratify="tag", test_size=TEST_SIZE)

    # [nb cell 100, lines 1-6]
    # Preprocess
    preprocessor = CustomPreprocessor()
    train_ds =  preprocessor.fit_transform(train_ds)
    val_ds = preprocessor.transform(val_ds)
    train_ds = train_ds.materialize() # run the pipeline once and cache it; otherwise it re-executes every epoch
    val_ds = val_ds.materialize()
    num_classes = len(preprocessor.class_to_index)  # [added] the notebook took this from EDA cell 21

    # [nb cell 95, lines 1-2]
    # Train loop config
    train_loop_config = {
        **TRAIN_LOOP_CONFIG,  # [added] the hyperparameter values now live in madewithml/config.py
    # [nb cell 95, lines 9-10]
        "num_classes": num_classes,
    }
    train_loop_config.update(config_overrides)  # [added] e.g. num_epochs=1 for a quick run

    # [nb cell 96, lines 1-12]
    # Workers -- this machine is CPU-only (torch.cuda.is_available() is False) with 4 physical
    # cores and ~17 GB RAM, of which only ~5 GB is typically free while the kernel is alive.
    #
    # Measured peak RSS for one worker doing BERT-base fwd+bwd+Adam at seq_len 40:
    #   batch/worker  16 -> 2.29 GB | 32 -> 2.65 GB | 64 -> 3.39 GB | 128 -> 4.93 GB
    # The previous setting (num_workers=2, batch_size=256 => 128/worker) needed ~9.9 GB and
    # died with "MemoryError: Unable to allocate internal buffer" in the Ray deserializer.
    #
    # One worker holding all 4 physical cores is also no slower here: 2 workers x 2 threads
    # splits the same cores and adds DDP gradient sync on top.
    num_workers = 2
    resources_per_worker = {"CPU": 4, "GPU": 0}
    if num_workers_override:  # [added] e.g. 1 worker for a quick run
        num_workers = num_workers_override

    # [nb cell 97, lines 1-7]
    # Scaling config
    scaling_config = ScalingConfig(
        num_workers=num_workers,
        use_gpu=bool(resources_per_worker["GPU"]),
        resources_per_worker=resources_per_worker,
    )

    # [nb cell 98, lines 1-8]
    # Run config
    checkpoint_config = CheckpointConfig(num_to_keep=1, checkpoint_score_attribute="val_loss", checkpoint_score_order="min") # remove num to keep if we want all checkpoint to save. checkpoint is configured per epoch
    run_config = RunConfig(
        name=f"llm_{datetime.now():%Y%m%d_%H%M%S}",  # unique per run so Ray doesn't reload an old run's checkpoints
        checkpoint_config=checkpoint_config,
        storage_path=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ray_results")),  # `local_dir` was removed in Ray 2.x; must be an absolute path
    )

    # [nb cell 101, line 1]
    # Dataset config
    # [nb cell 101, lines 3-8]

    dataset_config = DataConfig(
        datasets_to_split=["train", "val"],
        execution_options=ExecutionOptions(preserve_order=True), #default ray handes out blocks in order they arrive, for reproducability 
    )

    # [nb cell 102, lines 5-17]
    gc.collect()
    torch.cuda.empty_cache() #on gpu 


    # Trainer
    trainer = TorchTrainer(
        train_loop_per_worker=train_loop_per_worker,
        train_loop_config=train_loop_config,
        scaling_config=scaling_config,
        run_config=run_config,
        datasets={"train": train_ds, "val": val_ds},
        dataset_config=dataset_config
    )

    # [nb cell 103, line 1]
    results = trainer.fit()
    # [nb cell 104, line 1] (modified, see CELL_MAP.md)
    print(results.metrics_dataframe)
    # [nb cell 105, line 1] (modified, see CELL_MAP.md)
    print(results.best_checkpoints)
    return results


if __name__ == "__main__":
    train_model()
