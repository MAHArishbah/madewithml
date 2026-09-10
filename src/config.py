"""Constants, training hyperparameters and logging setup (from notebook cells 1, 3, 42, 95)."""
# [nb cell 1, line 20]
import sys
# [nb cell 1, lines 22-26]
import logging
logging.getLogger("httpx").setLevel(logging.WARNING)

TEST_SIZE=0.2
RANDOM_SEED=1234

# [nb cell 3, line 1]
DATASET_LOC = "https://raw.githubusercontent.com/GokuMohandas/Made-With-ML/main/datasets/dataset.csv"
# [nb cell 42, line 1]
HOLDOUT_LOC = "https://raw.githubusercontent.com/GokuMohandas/Made-With-ML/main/datasets/holdout.csv"

# Training hyperparameters used by madewithml/train.py (num_classes is added there, from the data)
TRAIN_LOOP_CONFIG = {
# [nb cell 95, lines 3-8]
    "dropout_p": 0.5,
    "lr": 1e-4,
    "lr_factor": 0.8,
    "lr_patience": 3,
    "num_epochs": 5,
    "batch_size": 32,  # was 256; BERT fwd+bwd on CPU peaks at ~4.9 GB RSS at 128/worker
}
